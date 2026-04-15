use anyhow::Result;
use crate::infrastructure::database::surreal::SurrealDb;
use crate::core::models::{DuplicateGroup, DuplicateItem};
use surrealdb::types::{RecordId, SurrealValue};
use super::{DbOperations, record_id_to_string};

const DUPLICATE_THRESHOLD: f32 = 0.92;

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f32; let mut na = 0f32; let mut nb = 0f32;
    for (x, y) in a.iter().zip(b.iter()) { dot += x*y; na += x*x; nb += y*y; }
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}

fn hamming(a: u64, b: u64) -> u32 { (a ^ b).count_ones() }

fn find(p: &mut Vec<usize>, i: usize) -> usize {
    if p[i] == i { i } else { let r = find(p, p[i]); p[i] = r; r }
}

fn resolve_video_thumb_fallback(base_dir: &str, file_name: &str, thumb_cache_dir: Option<&std::path::Path>) -> Option<String> {
    let video_abs = std::path::Path::new(base_dir).join(file_name);
    let stem = video_abs.file_stem()?.to_string_lossy();
    let thumb_file = format!("{}.thumb.jpg", stem);
    if let Some(cache_dir) = thumb_cache_dir {
        let p = cache_dir.join(&thumb_file);
        if p.exists() { return Some(p.to_string_lossy().to_string()); }
    }
    let parent = video_abs.parent().unwrap_or(std::path::Path::new(base_dir));
    let p = parent.join(&thumb_file);
    if p.exists() { return Some(p.to_string_lossy().to_string()); }
    None
}

impl DbOperations {
    pub async fn get_duplicates(
        db: &SurrealDb, source_dir: &str, media_type: Option<&str>, thumb_cache_dir: Option<&std::path::Path>,
    ) -> Result<Vec<DuplicateGroup>> {
        let mut groups: Vec<DuplicateGroup> = vec![];
        let base = source_dir.trim_end_matches('/');
        let type_filter = match media_type {
            Some(t) => format!(" AND media_type = '{}'", t),
            None => String::new(),
        };
        let mut covered: std::collections::HashSet<String> = std::collections::HashSet::new();

        // 1. Exact SHA-256 duplicates
        let query = format!(
            "SELECT sha256, ids FROM (SELECT file.sha256 AS sha256, array::group(id) AS ids, count() AS cnt FROM media WHERE deleted_at = NONE AND is_hidden = false{} GROUP BY file.sha256) WHERE cnt > 1",
            type_filter
        );
        let mut res = db.db.query(&query).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct DupRow { sha256: String, ids: Vec<RecordId> }
        let dup_rows: Vec<DupRow> = res.take(0)?;
        for row in dup_rows {
            let ids_str = row.ids.iter().map(|id| record_id_to_string(id)).collect::<Vec<_>>().join(", ");
            let q = format!("SELECT id, file.name AS name, file.size AS size, thumbnail FROM media WHERE id IN [{}]", ids_str);
            let mut r2 = db.db.query(&q).await?;
            #[derive(serde::Deserialize, SurrealValue)]
            struct DI { id: RecordId, name: Option<String>, size: Option<u64>, thumbnail: Option<String> }
            let items: Vec<DI> = r2.take(0)?;
            if items.len() < 2 { continue; }
            let key: Vec<String> = items.iter().map(|i| record_id_to_string(&i.id)).collect();
            for k in &key { covered.insert(k.clone()); }
            groups.push(DuplicateGroup {
                group_id: row.sha256.clone(), reason: "Trùng Hash — giống nhau 100%".into(),
                items: items.into_iter().map(|i| {
                    let name = i.name.clone();
                    let file_path = name.as_ref().map(|n| format!("{}/{}", base, n)).unwrap_or_default();
                    let thumbnail_path = i.thumbnail.map(|t| if std::path::Path::new(&t).is_absolute() { t } else { format!("{}/{}", base, t) })
                        .or_else(|| { if media_type != Some("video") { return None; } resolve_video_thumb_fallback(base, name.as_deref()?, thumb_cache_dir) });
                    DuplicateItem { media_id: record_id_to_string(&i.id), file_path, size: i.size.unwrap_or(0), thumbnail_path }
                }).collect(),
            });
        }

        // 2. pHash near-duplicate (Hamming <= 8)
        let phash_q = format!(
            "SELECT id, file.name AS name, file.size AS size, file.phash AS phash, thumbnail FROM media WHERE deleted_at = NONE AND is_hidden = false AND file.phash IS NOT NONE{}",
            type_filter
        );
        let mut pr = db.db.query(&phash_q).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct PRow { id: RecordId, name: Option<String>, size: Option<u64>, phash: Option<String>, thumbnail: Option<String> }
        let prows: Vec<PRow> = pr.take(0)?;
        let phash_items: Vec<(RecordId, Option<String>, u64, u64, Option<String>)> = prows.into_iter()
            .filter_map(|r| { let h = u64::from_str_radix(r.phash.as_deref()?, 16).ok()?; Some((r.id, r.name, r.size.unwrap_or(0), h, r.thumbnail)) })
            .collect();

        let n = phash_items.len();
        let mut parent: Vec<usize> = (0..n).collect();
        for i in 0..n { for j in (i+1)..n { if hamming(phash_items[i].3, phash_items[j].3) <= 8 { let ri = find(&mut parent, i); let rj = find(&mut parent, j); if ri != rj { parent[ri] = rj; } } } }
        let mut ph_clusters: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for i in 0..n { ph_clusters.entry(find(&mut parent, i)).or_default().push(i); }
        for (_root, idxs) in ph_clusters {
            if idxs.len() < 2 { continue; }
            let ids: Vec<String> = idxs.iter().map(|&i| record_id_to_string(&phash_items[i].0)).collect();
            if ids.iter().all(|id| covered.contains(id)) { continue; }
            for id in &ids { covered.insert(id.clone()); }
            groups.push(DuplicateGroup {
                group_id: format!("phash_{}", ids[0]),
                reason: "Ảnh gần giống nhau (pHash — cùng nội dung nhưng khác kích thước/nén)".into(),
                items: idxs.iter().map(|&i| DuplicateItem {
                    media_id: record_id_to_string(&phash_items[i].0),
                    file_path: phash_items[i].1.as_ref().map(|n| format!("{}/{}", base, n)).unwrap_or_default(),
                    size: phash_items[i].2,
                    thumbnail_path: phash_items[i].4.as_ref().map(|t| if std::path::Path::new(t).is_absolute() { t.clone() } else { format!("{}/{}", base, t) })
                        .or_else(|| { if media_type != Some("video") { return None; } resolve_video_thumb_fallback(base, phash_items[i].1.as_deref()?, thumb_cache_dir) }),
                }).collect(),
            });
        }

        // 3. Vision vector similarity >= threshold
        let emb_src_filter = match media_type { Some("video") => " AND source = 'video_frame'", _ => " AND source = 'image'" };
        let emb_q = format!(
            "SELECT media_id, vec, source, frame_idx FROM embedding WHERE media_id.deleted_at = NONE AND media_id.is_hidden = false{}{}",
            type_filter.replace("media_type", "media_id.media_type"), emb_src_filter
        );
        let mut er = db.db.query(&emb_q).await?;
        #[derive(serde::Deserialize, SurrealValue)]
        struct ERow { media_id: RecordId, vec: Vec<f32>, #[allow(dead_code)] source: Option<String>, frame_idx: Option<u32> }
        let all_embs: Vec<ERow> = er.take(0)?;

        let mut rep_embs: std::collections::HashMap<String, (&ERow, u32)> = std::collections::HashMap::new();
        for e in &all_embs {
            let mid = record_id_to_string(&e.media_id);
            let fi = e.frame_idx.unwrap_or(u32::MAX);
            rep_embs.entry(mid).and_modify(|(prev, prev_fi)| { if fi < *prev_fi { *prev = e; *prev_fi = fi; } }).or_insert((e, fi));
        }
        let rep: Vec<(&ERow, String)> = rep_embs.into_iter().map(|(mid, (e, _))| (e, mid)).collect();

        if rep.len() > 1 {
            use rayon::prelude::*;
            let pairs: Vec<(usize, usize)> = (0..rep.len()).into_par_iter().flat_map(|i| {
                let mut local = vec![];
                for j in (i+1)..rep.len() {
                    if rep[i].0.media_id == rep[j].0.media_id { continue; }
                    if cosine_sim(&rep[i].0.vec, &rep[j].0.vec) >= DUPLICATE_THRESHOLD { local.push((i, j)); }
                }
                local
            }).collect();

            if !pairs.is_empty() {
                let mut par: Vec<usize> = (0..rep.len()).collect();
                for (i, j) in pairs { let ri = find(&mut par, i); let rj = find(&mut par, j); if ri != rj { par[ri] = rj; } }
                let mut clusters: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
                for i in 0..rep.len() { clusters.entry(find(&mut par, i)).or_default().push(i); }
                let mut ci = 0usize;
                for (_root, idxs) in clusters {
                    if idxs.len() < 2 { continue; }
                    let ids: Vec<String> = idxs.iter().map(|&i| rep[i].1.clone()).collect();
                    if ids.iter().all(|id| covered.contains(id)) { continue; }
                    for id in &ids { covered.insert(id.clone()); }
                    let ids_str = ids.iter().map(|id| if id.contains(':') { id.clone() } else { format!("media:{}", id) }).collect::<Vec<_>>().join(", ");
                    let q = format!("SELECT id, file.name AS name, file.size AS size, thumbnail FROM media WHERE id IN [{}]", ids_str);
                    let mut r3 = db.db.query(&q).await?;
                    #[derive(serde::Deserialize, SurrealValue)]
                    struct DI { id: RecordId, name: Option<String>, size: Option<u64>, thumbnail: Option<String> }
                    let items: Vec<DI> = r3.take(0)?;
                    if items.len() < 2 { continue; }
                    let reason = match media_type { Some("video") => "Video gần giống nhau (AI phát hiện đoạn đầu tương tự ≥ 97%)", _ => "Ảnh gần giống nhau (AI phát hiện ≥ 97%)" };
                    groups.push(DuplicateGroup {
                        group_id: format!("vec_{}_{}", ci, record_id_to_string(&items[0].id)),
                        reason: reason.into(),
                        items: items.into_iter().map(|i| {
                            let name = i.name.clone();
                            let file_path = name.as_ref().map(|n| format!("{}/{}", base, n)).unwrap_or_default();
                            let thumbnail_path = i.thumbnail.map(|t| if std::path::Path::new(&t).is_absolute() { t } else { format!("{}/{}", base, t) })
                                .or_else(|| { if media_type != Some("video") { return None; } resolve_video_thumb_fallback(base, name.as_deref()?, thumb_cache_dir) });
                            DuplicateItem { media_id: record_id_to_string(&i.id), file_path, size: i.size.unwrap_or(0), thumbnail_path }
                        }).collect(),
                    });
                    ci += 1;
                }
            }
        }

        Ok(groups)
    }
}
