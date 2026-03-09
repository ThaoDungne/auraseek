import { invoke, convertFileSrc } from "@tauri-apps/api/core";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface BboxInfo {
    x: number;
    y: number;
    w: number;
    h: number;
}

export interface DetectedObject {
    class_name: string;
    conf: number;
    bbox: BboxInfo;
    /** RLE mask: each [offset, length] means pixels at row-major indices [offset..offset+length) are set */
    mask_rle?: [number, number][];
}

export interface DetectedFace {
    face_id: string;
    name: string | null;
    conf: number;
    bbox: BboxInfo;
}

export interface SearchResultMeta {
    width: number | null;
    height: number | null;
    created_at: string | null;
    objects: string[];
    faces: string[];
}

export interface SearchResult {
    media_id:          string;
    similarity_score:  number;
    file_path:         string;
    media_type:        string;
    metadata:          SearchResultMeta;
    /** Full detection data for hover overlays */
    detected_objects:  DetectedObject[];
    detected_faces:    DetectedFace[];
    width:             number | null;
    height:            number | null;
}

export interface TimelineItem {
    media_id: string;
    file_path: string;
    media_type: string;
    width: number | null;
    height: number | null;
    created_at: string | null;
    objects: string[];
    faces: string[];
    face_ids: string[];
    favorite: boolean;
    detected_objects: DetectedObject[];
    detected_faces: DetectedFace[];
}

export interface TimelineGroup {
    label: string;
    year: number;
    month: number;
    day: number | null;
    items: TimelineItem[];
}

export interface PersonGroup {
    face_id: string;
    name: string | null;
    photo_count: number;
    cover_path: string | null;
    thumbnail: string | null;
    conf: number | null;
    face_bbox: BboxInfo | null;
}

export interface DuplicateItem {
    media_id: string;
    file_path: string;
    size: number;
}

export interface DuplicateGroup {
    group_id: string;
    reason: string;
    items: DuplicateItem[];
}

export interface IngestSummary {
    total_found: number;
    newly_added: number;
    skipped_dup: number;
    errors: number;
}

export interface SearchFilters {
    object?: string;
    face?: string;
    month?: number;
    year?: number;
    media_type?: string;
}

export interface AppStatus {
    engine_ready: boolean;
    db_ready: boolean;
    vector_count: number;
    source_dir?: string;
}

export interface SyncStatus {
    state: "idle" | "syncing" | "done" | "error";
    processed: number;
    total: number;
    message: string;
}

// ─── API ─────────────────────────────────────────────────────────────────────

export const AuraSeekApi = {
    /** Returns true if all AI model files are present in the app data dir. */
    async checkModels(): Promise<boolean> {
        return invoke<boolean>("cmd_check_models");
    },

    /**
     * Start downloading missing model files in the background.
     * Progress is reported via `"model-download-progress"` events.
     * Listen for `{ done: true }` before calling `init()`.
     */
    async downloadModels(): Promise<void> {
        return invoke<void>("cmd_download_models");
    },

    async init(): Promise<string> {
        return invoke<string>("cmd_init");
    },

    async getStatus(): Promise<AppStatus> {
        return invoke<AppStatus>("cmd_get_status");
    },

    async scanFolder(sourcePath: string): Promise<IngestSummary> {
        return invoke<IngestSummary>("cmd_scan_folder", { sourcePath });
    },

    async searchText(query: string, filters?: SearchFilters): Promise<SearchResult[]> {
        return invoke<SearchResult[]>("cmd_search_text", { query, filters });
    },

    async searchImage(imagePath: string, filters?: SearchFilters): Promise<SearchResult[]> {
        return invoke<SearchResult[]>("cmd_search_image", { imagePath, filters });
    },

    async searchCombined(text: string, imagePath: string, filters?: SearchFilters): Promise<SearchResult[]> {
        return invoke<SearchResult[]>("cmd_search_combined", { text, imagePath, filters });
    },

    async searchObject(className: string, filters?: SearchFilters): Promise<SearchResult[]> {
        return invoke<SearchResult[]>("cmd_search_object", { className, filters });
    },

    async searchFace(name: string, filters?: SearchFilters): Promise<SearchResult[]> {
        return invoke<SearchResult[]>("cmd_search_face", { name, filters });
    },

    async searchFilterOnly(filters?: SearchFilters): Promise<SearchResult[]> {
        return invoke<SearchResult[]>("cmd_search_filter_only", { filters });
    },

    async getTimeline(limit?: number): Promise<TimelineGroup[]> {
        return invoke<TimelineGroup[]>("cmd_get_timeline", { limit });
    },

    async getPeople(): Promise<PersonGroup[]> {
        return invoke<PersonGroup[]>("cmd_get_people");
    },

    async namePerson(faceId: string, name: string): Promise<void> {
        return invoke<void>("cmd_name_person", { faceId, name });
    },

    async getDuplicates(): Promise<DuplicateGroup[]> {
        return invoke<DuplicateGroup[]>("cmd_get_duplicates");
    },

    async getSearchHistory(limit?: number): Promise<any[]> {
        return invoke<any[]>("cmd_get_search_history", { limit });
    },

    async toggleFavorite(mediaId: string): Promise<boolean> {
        return invoke<boolean>("cmd_toggle_favorite", { mediaId });
    },

    async getDistinctObjects(): Promise<string[]> {
        return invoke<string[]>("cmd_get_distinct_objects");
    },

    async setDbConfig(addr: string, user: string, pass: string): Promise<void> {
        return invoke<void>("cmd_set_db_config", { addr, user, pass });
    },

    async getSourceDir(): Promise<string> {
        return invoke<string>("cmd_get_source_dir");
    },

    async setSourceDir(dir: string): Promise<void> {
        return invoke<void>("cmd_set_source_dir", { dir });
    },

    async getSyncStatus(): Promise<SyncStatus> {
        return invoke<SyncStatus>("cmd_get_sync_status");
    },

    async autoScan(): Promise<string> {
        return invoke<string>("cmd_auto_scan");
    },

    async ingestFiles(filePaths: string[]): Promise<IngestSummary> {
        return invoke<IngestSummary>("cmd_ingest_files", { filePaths });
    },

    /** Send raw image bytes (base64) to backend — used for clipboard paste without a file path */
    async ingestImageData(data: string, ext: string): Promise<IngestSummary> {
        return invoke<IngestSummary>("cmd_ingest_image_data", { data, ext });
    },
    // ─── Trash & Hidden ──────────────────────────────────────────────────────

    async moveToTrash(mediaId: string): Promise<void> {
        return invoke<void>("cmd_move_to_trash", { mediaId });
    },

    async restoreFromTrash(mediaId: string): Promise<void> {
        return invoke<void>("cmd_restore_from_trash", { mediaId });
    },

    async getTrash(): Promise<TimelineGroup[]> {
        return invoke<TimelineGroup[]>("cmd_get_trash");
    },

    async emptyTrash(): Promise<void> {
        return invoke<void>("cmd_empty_trash");
    },

    async hidePhoto(mediaId: string): Promise<void> {
        return invoke<void>("cmd_hide_photo", { mediaId });
    },

    async unhidePhoto(mediaId: string): Promise<void> {
        return invoke<void>("cmd_unhide_photo", { mediaId });
    },

    async getHiddenPhotos(): Promise<TimelineGroup[]> {
        return invoke<TimelineGroup[]>("cmd_get_hidden_photos");
    },

    async authenticateOs(): Promise<boolean> {
        return invoke<boolean>("cmd_authenticate_os");
    },

    /** Dọn dẹp các ảnh/video không còn tồn tại trên đĩa. Trả về số lượng đã xóa. */
    async cleanupDatabase(): Promise<number> {
        return invoke<number>("cmd_cleanup_database");
    },

    /** Đặt lại database: xóa toàn bộ dữ liệu (ảnh trên đĩa không bị xóa). */
    async resetDatabase(): Promise<void> {
        return invoke<void>("cmd_reset_database");
    },
};

export function localFileUrl(filePath: string): string {
    if (!filePath) return "";
    // Use Tauri's convertFileSrc which correctly handles path encoding and
    // sets up the asset:// URL with proper streaming support for video.
    try {
        return convertFileSrc(filePath);
    } catch {
        // Fallback for non-Tauri environments (browser dev)
        const encoded = encodeURIComponent(filePath).replace(/%2F/g, "/").replace(/%5C/g, "/");
        return `asset://localhost/${encoded}`;
    }
}
