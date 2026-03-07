# Surrealist – Queries xem data AuraSeek

**Kết nối Surrealist:**
- URL: `ws://127.0.0.1:8000` (hoặc địa chỉ SurrealDB của bạn)
- Namespace: `auraseek`
- Database: `auraseek`
- User/Pass: `root` / `root` (nếu dùng auth)

---

## 1. Thống kê nhanh

```sql
-- Số bản ghi từng bảng
SELECT count() FROM media GROUP ALL;
SELECT count() FROM embedding GROUP ALL;
SELECT count() FROM person GROUP ALL;
SELECT count() FROM search_history GROUP ALL;
```

---

## 2. Bảng `media` (ảnh/video đã ingest)

```sql
-- Xem tất cả media (id, path, type, ngày tạo)
SELECT id, file.path, media_type, metadata.created_at, processed
FROM media
ORDER BY metadata.created_at DESC
LIMIT 50;
```

```sql
-- Xem chi tiết 1 bản ghi (bỏ LIMIT 1 để xem nhiều)
SELECT * FROM media LIMIT 1;
```

```sql
-- Chỉ path + objects + faces (để đọc nhanh)
SELECT id, file.path, objects[*].class_name, faces[*].face_id, faces[*].name
FROM media
ORDER BY metadata.created_at DESC
LIMIT 20;
```

```sql
-- Media chưa AI xử lý
SELECT id, file.path FROM media WHERE processed = false;
```

```sql
-- Media có khuôn mặt
SELECT id, file.path, faces[*].face_id, faces[*].name
FROM media
WHERE array::len(faces) > 0
LIMIT 20;
```

---

## 3. Bảng `embedding` (vector tìm kiếm)

```sql
-- Số embedding, không trả về vector (vec rất dài)
SELECT count() FROM embedding GROUP ALL;
```

```sql
-- Id media + source, không select vec
SELECT id, media_id, source, frame_ts, frame_idx, created_at
FROM embedding
ORDER BY created_at DESC
LIMIT 20;
```

```sql
-- Độ dài vector (ví dụ 1024 chiều)
SELECT id, media_id, array::len(vec) AS dim FROM embedding LIMIT 5;
```

---

## 4. Bảng `person` (nhóm khuôn mặt)

```sql
-- Tất cả person
SELECT * FROM person ORDER BY created_at DESC;
```

```sql
-- Person đã đặt tên
SELECT * FROM person WHERE name != NONE;
```

---

## 5. Bảng `search_history`

```sql
-- Lịch sử tìm kiếm gần nhất
SELECT * FROM search_history
ORDER BY created_at DESC
LIMIT 20;
```

```sql
-- Chỉ query text và thời gian
SELECT query, image_path, created_at
FROM search_history
ORDER BY created_at DESC
LIMIT 15;
```

---

## 6. Query kết hợp (optional)

```sql
-- Media theo tháng (nhóm)
SELECT
    time::format(metadata.created_at, '%Y-%m') AS month,
    count() AS total
FROM media
WHERE metadata.created_at != NONE
GROUP BY time::format(metadata.created_at, '%Y-%m')
ORDER BY month DESC;
```

```sql
-- Person có số ảnh nhiều nhất (cần subquery tương tự app)
SELECT
    p.face_id,
    p.name,
    (SELECT count() FROM media WHERE faces.*.face_id CONTAINS p.face_id GROUP ALL)[0].count AS photo_count
FROM person AS p
ORDER BY photo_count DESC;
```

---

Chạy từng khối trong Surrealist (Query tab), nhớ chọn đúng **Namespace** và **Database** là `auraseek`.
