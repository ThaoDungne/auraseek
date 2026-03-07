import { invoke } from "@tauri-apps/api/core";

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
    media_id: string;
    similarity_score: number;
    file_path: string;
    media_type: string;
    metadata: SearchResultMeta;
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
    sha256: string;
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
}

// ─── API ─────────────────────────────────────────────────────────────────────

export const AuraSeekApi = {
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
};

export function localFileUrl(filePath: string): string {
    if (!filePath) return "";
    const encoded = encodeURIComponent(filePath).replace(/%2F/g, "/").replace(/%5C/g, "/");
    return `asset://localhost/${encoded}`;
}
