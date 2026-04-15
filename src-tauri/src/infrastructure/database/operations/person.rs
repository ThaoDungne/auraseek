use anyhow::Result;
use crate::infrastructure::database::surreal::SurrealDb;
use crate::infrastructure::database::models::PersonDoc;
use crate::core::models::PersonGroup;
use surrealdb::types::{RecordId, SurrealValue};
use super::{DbOperations, record_id_to_string};

impl DbOperations {
    pub async fn upsert_person(db: &SurrealDb, person: PersonDoc) -> Result<()> {
        let fid = person.face_id.clone();
        let name = person.name.clone();
        let thumb = person.thumbnail.clone();
        let conf = person.conf;
        let bbox = person.face_bbox.clone();
        db.db.query(
            "INSERT INTO person { face_id: $fid, name: $name, thumbnail: $thumb, conf: $conf, face_bbox: $bbox }
             ON DUPLICATE KEY UPDATE
                name = $input.name ?? name,
                conf = IF $input.conf IS NOT NONE AND (conf IS NONE OR $input.conf > conf) THEN $input.conf ELSE conf END,
                thumbnail = IF $input.conf IS NOT NONE AND (conf IS NONE OR $input.conf > conf) THEN $input.thumbnail ELSE thumbnail END,
                face_bbox = IF $input.conf IS NOT NONE AND (conf IS NONE OR $input.conf > conf) THEN $input.face_bbox ELSE face_bbox END"
        )
        .bind(("fid", fid)).bind(("name", name)).bind(("thumb", thumb)).bind(("conf", conf)).bind(("bbox", bbox))
        .await?.check().map_err(|e| anyhow::anyhow!("upsert_person failed: {}", e))?;
        Ok(())
    }

    pub async fn name_person(db: &SurrealDb, face_id: &str, name: &str) -> Result<()> {
        let fid = face_id.to_string();
        let n = name.to_string();
        db.db.query("UPDATE person SET name = $name WHERE face_id = $fid")
            .bind(("name", n.clone())).bind(("fid", fid.clone())).await?;
        db.db.query("UPDATE media SET faces[WHERE face_id = $fid].name = $name WHERE faces.*.face_id CONTAINS $fid")
            .bind(("fid", fid)).bind(("name", n)).await?;
        Ok(())
    }

    pub async fn get_people(db: &SurrealDb, source_dir: &str) -> Result<Vec<PersonGroup>> {
        let mut res = db.db.query(
            "SELECT face_id, name, thumbnail, conf, face_bbox,
                (SELECT count() FROM media WHERE faces.*.face_id CONTAINS $parent.face_id AND deleted_at = NONE AND is_hidden = false GROUP ALL)[0].count AS photo_count,
                (SELECT file.name FROM media WHERE faces.*.face_id CONTAINS $parent.face_id AND deleted_at = NONE AND is_hidden = false LIMIT 1)[0].file.name AS cover_name
            FROM person ORDER BY photo_count DESC"
        ).await?;

        #[derive(serde::Deserialize, SurrealValue)]
        struct PRow {
            face_id: String, name: Option<String>, thumbnail: Option<String>,
            conf: Option<f32>, face_bbox: Option<crate::infrastructure::database::models::Bbox>,
            photo_count: Option<u64>, cover_name: Option<String>,
        }
        let rows: Vec<PRow> = res.take(0)?;
        let base = source_dir.trim_end_matches('/');
        Ok(rows.into_iter().map(|r| {
            let cover_path = r.cover_name.as_ref().map(|n| format!("{}/{}", base, n));
            let thumbnail = r.thumbnail.map(|t| {
                if std::path::Path::new(&t).is_absolute() { t } else { format!("{}/{}", base, t) }
            });
            PersonGroup {
                face_id: r.face_id, name: r.name,
                photo_count: r.photo_count.unwrap_or(0) as u32,
                cover_path, thumbnail, conf: r.conf,
                face_bbox: r.face_bbox.map(|b| crate::core::models::BboxInfo { x: b.x, y: b.y, w: b.w, h: b.h }),
            }
        }).collect())
    }

    pub async fn merge_people(db: &SurrealDb, target_face_id: &str, source_face_id: &str) -> Result<()> {
        let src = source_face_id.to_string();
        let tgt = target_face_id.to_string();
        let src_id = if src.contains(':') { src.clone() } else { format!("person:{}", src) };
        let tgt_id = if tgt.contains(':') { tgt.clone() } else { format!("person:{}", tgt) };

        #[derive(serde::Deserialize, SurrealValue)]
        struct NameRow { name: Option<String> }
        let mut res = db.db.query("SELECT name FROM type::record($id)").bind(("id", src_id.clone())).await?;
        let source_name: Option<NameRow> = res.take(0)?;
        let mut res2 = db.db.query("SELECT name FROM type::record($id)").bind(("id", tgt_id.clone())).await?;
        let target_name_row: Option<NameRow> = res2.take(0)?;
        let final_name: Option<String> = target_name_row.and_then(|r| r.name).or(source_name.and_then(|r| r.name));

        db.db.query("UPDATE media SET faces[WHERE face_id = $src_raw].face_id = $tgt_raw, faces[WHERE face_id = $src_raw].name = $nm WHERE faces.*.face_id CONTAINS $src_raw")
            .bind(("src_raw", src)).bind(("tgt_raw", tgt)).bind(("nm", final_name.clone())).await?;
        db.db.query("DELETE type::record($id)").bind(("id", src_id)).await?.check()?;
        if let Some(ref n) = final_name {
            db.db.query("UPDATE type::record($id) SET name = $name").bind(("id", tgt_id)).bind(("name", n.clone())).await?.check()?;
        }
        Ok(())
    }

    pub async fn delete_person(db: &SurrealDb, face_id: &str) -> Result<()> {
        let fid = face_id.to_string();
        db.db.query("UPDATE media SET faces = faces.filter(|$f| $f.face_id != $fid) WHERE faces.*.face_id CONTAINS $fid")
            .bind(("fid", fid.clone())).await?;
        let table_id = if fid.contains(':') { fid } else { format!("person:{}", fid) };
        db.db.query("DELETE type::record($id)").bind(("id", table_id)).await?.check()?;
        Ok(())
    }

    pub async fn remove_face_from_person(db: &SurrealDb, media_id: &str, face_id: &str) -> Result<()> {
        let fid = face_id.to_string();
        let query = format!("UPDATE {} SET faces = faces.filter(|$f| $f.face_id != $fid)", media_id);
        db.db.query(&query).bind(("fid", fid)).await?.check()?;
        Ok(())
    }
}
