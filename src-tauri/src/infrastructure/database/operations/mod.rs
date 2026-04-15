/// Database operations ? SurrealDB v3 edition
/// Split by domain responsibility.
pub mod media;
pub mod embedding;
pub mod person;
pub mod trash;
pub mod search;
pub mod config;
pub mod album;
pub mod duplicates;

use anyhow::Result;
use std::collections::HashMap;
use crate::infrastructure::database::surreal::SurrealDb;
use crate::core::models::{
    SearchResult, SearchResultMeta, TimelineItem, TimelineGroup,
    DetectedObject, BboxInfo, DetectedFace,
};
use crate::infrastructure::database::models::MediaRow;
use surrealdb::types::{RecordId, RecordIdKey};
use chrono::Datelike;

pub struct DbOperations;

pub fn record_id_to_string(id: &RecordId) -> String {
    let key_str = match &id.key {
        RecordIdKey::String(s) => s.clone(),
        RecordIdKey::Number(n) => n.to_string(),
        _ => "unknown".to_string(),
    };
    format!("{}:{}", id.table, key_str)
}

#[allow(dead_code)]
fn strip_table_prefix(id: &str) -> &str {
    id.find(':').map(|i| &id[i+1..]).unwrap_or(id)
}

pub fn row_to_search_result(row: &MediaRow, score: f32, source_dir: &str) -> SearchResult {
    let id_str = record_id_to_string(&row.id);
    let base   = source_dir.trim_end_matches('/');
    SearchResult {
        media_id:         id_str,
        similarity_score: score,
        file_path:        format!("{}/{}", base, row.file.name),
        media_type:       row.media_type.clone(),
        width:            row.metadata.width,
        height:           row.metadata.height,
        detected_objects: row.objects.iter().map(|o| DetectedObject {
            class_name: o.class_name.clone(), conf: o.conf,
            bbox: BboxInfo { x: o.bbox.x, y: o.bbox.y, w: o.bbox.w, h: o.bbox.h },
            mask_rle: o.mask_rle.clone(),
        }).collect(),
        detected_faces: row.faces.iter().map(|f| DetectedFace {
            face_id: f.face_id.clone(), name: f.name.clone(), conf: f.conf,
            bbox: BboxInfo { x: f.bbox.x, y: f.bbox.y, w: f.bbox.w, h: f.bbox.h },
        }).collect(),
        metadata: SearchResultMeta {
            width: row.metadata.width, height: row.metadata.height,
            created_at: row.metadata.created_at.as_ref().map(|dt: &surrealdb::types::Datetime| dt.to_string()),
            objects: row.objects.iter().map(|o| o.class_name.clone()).collect(),
            faces: row.faces.iter().filter_map(|f| f.name.clone()).collect(),
        },
        thumbnail_path: row.thumbnail.clone(),
    }
}

impl DbOperations {
    pub(crate) fn group_rows_into_timeline(rows: Vec<MediaRow>, source_dir: &str) -> Result<Vec<TimelineGroup>> {
        let base = source_dir.trim_end_matches('/');
        let mut groups: HashMap<(i32, u32), TimelineGroup> = HashMap::new();

        let mut sorted_rows = rows;
        sorted_rows.sort_by(|a, b| {
            let (ay, am) = parse_ym(a);
            let (by, bm) = parse_ym(b);
            if ay != by { by.cmp(&ay) } else { bm.cmp(&am) }
        });

        for row in sorted_rows {
            let (year, month) = parse_ym(&row);
            let label = format_month_label(year, month);
            let file_path = format!("{}/{}", base, row.file.name);
            let thumbnail_path = row.thumbnail.clone();
            let item = TimelineItem {
                media_id:   record_id_to_string(&row.id),
                file_path, media_type: row.media_type.clone(),
                width: row.metadata.width, height: row.metadata.height,
                created_at: row.metadata.created_at.as_ref().map(|dt: &surrealdb::types::Datetime| dt.to_string()),
                objects:  row.objects.iter().map(|o| o.class_name.clone()).collect(),
                faces:    row.faces.iter().filter_map(|f| f.name.clone()).collect(),
                face_ids: row.faces.iter().map(|f| f.face_id.clone()).collect(),
                favorite: row.favorite,
                deleted_at: row.deleted_at.as_ref().map(|dt: &surrealdb::types::Datetime| dt.to_string()),
                is_hidden: row.is_hidden,
                thumbnail_path,
                detected_objects: row.objects.iter().map(|o| DetectedObject {
                    class_name: o.class_name.clone(), conf: o.conf,
                    bbox: BboxInfo { x: o.bbox.x, y: o.bbox.y, w: o.bbox.w, h: o.bbox.h },
                    mask_rle: o.mask_rle.clone(),
                }).collect(),
                detected_faces: row.faces.iter().map(|f| DetectedFace {
                    face_id: f.face_id.clone(), name: f.name.clone(), conf: f.conf,
                    bbox: BboxInfo { x: f.bbox.x, y: f.bbox.y, w: f.bbox.w, h: f.bbox.h },
                }).collect(),
            };
            groups.entry((year, month)).or_insert_with(|| TimelineGroup {
                label, year, month, day: None, items: vec![],
            }).items.push(item);
        }

        let mut result: Vec<TimelineGroup> = groups.into_values().collect();
        result.sort_by(|a, b| b.year.cmp(&a.year).then(b.month.cmp(&a.month)));
        Ok(result)
    }
}

fn parse_ym(row: &MediaRow) -> (i32, u32) {
    if let Some(ref dt) = row.metadata.created_at { return (dt.year(), dt.month() as u32); }
    if let Some(ref dt) = row.metadata.modified_at { return (dt.year(), dt.month() as u32); }
    (1970, 1)
}

fn format_month_label(year: i32, month: u32) -> String {
    let months = ["Th?ng 1","Th?ng 2","Th?ng 3","Th?ng 4","Th?ng 5","Th?ng 6",
                  "Th?ng 7","Th?ng 8","Th?ng 9","Th?ng 10","Th?ng 11","Th?ng 12"];
    let m = months.get((month.saturating_sub(1)) as usize).unwrap_or(&"");
    format!("{} {}", m, year)
}

pub(crate) fn parse_year_month_from_str(s: &str) -> Option<(i32, u32)> {
    use chrono::Datelike;
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) { return Some((dt.year(), dt.month())); }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") { return Some((dt.year(), dt.month())); }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") { return Some((dt.year(), dt.month())); }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") { return Some((dt.year(), dt.month())); }
    if let Ok(dt) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") { return Some((dt.year(), dt.month())); }
    if s.len() >= 7 {
        let parts: Vec<&str> = s.split(|c: char| c == '-' || c == '/').collect();
        if parts.len() >= 2 {
            if let (Ok(y), Ok(m)) = (parts[0].parse::<i32>(), parts[1].parse::<u32>()) {
                if (1900..=2100).contains(&y) && (1..=12).contains(&m) { return Some((y, m)); }
            }
        }
    }
    None
}
