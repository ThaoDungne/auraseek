import { useState, useEffect, useCallback, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import { TooltipProvider } from "@/components/ui/tooltip";
import { NewLayout } from "./components/layout/NewLayout";
import { SelectionProvider } from "@/contexts/SelectionContext";
import { TimelineView } from "@/views/timeline";
import { PeopleView } from "@/views/people/PeopleView";
import { DuplicatesView } from "@/views/duplicates/DuplicatesView";
import { AlbumsView } from "@/views/albums/AlbumsView";
import { TrashView } from "@/views/trash/TrashView";
import { HiddenView } from "@/views/hidden/HiddenView";
import { FilteredGalleryView } from "@/views/gallery/FilteredGalleryView";
import { SearchResultsView } from "@/views/search/SearchResultsView";
import { FirstRunModal } from "@/components/common/FirstRunModal";
import ModelDownloadScreen, { type ModelDownloadEvent } from "@/components/common/ModelDownloadScreen";
import { AuraSeekApi, localFileUrl, streamFileUrl, warmStreamPortCache, type SearchResult, type TimelineGroup, type PersonGroup, type SearchFilters as ApiFilters, type SyncStatus } from "@/lib/api";
import type { Photo } from "@/types/photo.type";

type AppRoute = {
  view: string;
  payload?: any;
};

export type ActiveFilters = {
  object?: string;
  face?: string;
  month?: number;
  year?: number;
  mediaType?: string;
};

function App() {
  const [route, setRoute] = useState<AppRoute>({ view: "timeline" });
  const [searchQuery, setSearchQuery] = useState("");
  const [searchImagePath, setSearchImagePath] = useState<string | null>(null);
  const searchTempPathRef = useRef<string | null>((window as any).__AURASEEK_SEARCH_TMP_PATH__ || null);
  const [activeFilters, setActiveFilters] = useState<ActiveFilters>({});
  const [downloadProgress, setDownloadProgress] = useState<ModelDownloadEvent | null>(null);
  const [needsDownload, setNeedsDownload] = useState(false);

  // Data state
  const [timelineGroups, setTimelineGroups] = useState<TimelineGroup[]>([]);
  const [people, setPeople] = useState<PersonGroup[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [selectionMode, setSelectionMode] = useState(false);

  // First-run and source dir
  const [showFirstRun, setShowFirstRun] = useState(false);
  const [sourceDir, setSourceDir] = useState("");
  const [isDragOver, setIsDragOver] = useState(false);

  // Sync status
  const [syncStatus, setSyncStatus] = useState<SyncStatus | null>(null);
  const syncPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    const initialize = async () => {
      console.log("[AuraSeek] 🚀 Initializing app...");

      if (!('__TAURI_INTERNALS__' in window)) {
        const msg = "App phải chạy trong Tauri WebView. Dùng lệnh: cargo tauri dev";
        console.warn("[AuraSeek] ⚠️", msg);
        setInitError(msg);
        setIsInitialized(true);
        return;
      }

      try {
        // 1. Check if model files are present
        const modelsReady = await AuraSeekApi.checkModels();

        if (!modelsReady) {
          setNeedsDownload(true);
          // Wait for download to complete via events
          await new Promise<void>((resolve, reject) => {
            let unlistenDownload: (() => void) | null = null;
            listen<ModelDownloadEvent>("model-download-progress", (event) => {
              setDownloadProgress(event.payload);
              if (event.payload.done) {
                unlistenDownload?.();
                resolve();
              }
              if (event.payload.error) {
                unlistenDownload?.();
                reject(new Error(event.payload.error));
              }
            }).then((fn) => { unlistenDownload = fn; });

            // Kick off the download
            AuraSeekApi.downloadModels().catch(reject);
          });
          setNeedsDownload(false);
          // Keep screen up while engine loads models into RAM
          setDownloadProgress({
            file: "", progress: 1.0,
            message: "Đang chuẩn bị hệ thống...",
            done: false, error: "",
            file_index: 0, file_total: 0,
            bytes_done: 0, bytes_total: 0,
          });
        }

        // 2. Normal initialization (engine + DB)
        const msg = await AuraSeekApi.init();
        console.log("[AuraSeek] ✅ Engine + DB ready:", msg);
        setInitError(null);
        setDownloadProgress(null); // hide loading screen now that engine is ready

        // Warm stream port cache so `streamFileUrlSync()` can serve thumbnails reliably
        await warmStreamPortCache();

        // Get source_dir from backend
        const dir = await AuraSeekApi.getSourceDir();
        setSourceDir(dir);

        if (!dir) {
          // First run: show folder picker
          setShowFirstRun(true);
        } else {
          // Auto-scan in background
          await loadTimeline();
          triggerAutoScan();
        }
      } catch (err: any) {
        console.warn("[AuraSeek] ⚠️ Init warning:", err);
        setInitError(String(err));
      } finally {
        setIsInitialized(true);
      }
    };
    initialize();
    return () => {
      if (syncPollRef.current) clearInterval(syncPollRef.current);
      if (unlisten) unlisten();
    };
  }, []);

  const triggerAutoScan = useCallback(async () => {
    try {
      await AuraSeekApi.autoScan();
      setSyncStatus({ state: "syncing", processed: 0, total: 0, message: "Đang đồng bộ dữ liệu..." });
      // Poll sync status
      if (syncPollRef.current) clearInterval(syncPollRef.current);
      syncPollRef.current = setInterval(async () => {
        try {
          const st = await AuraSeekApi.getSyncStatus();
          setSyncStatus(st);
          if (st.state === "done" || st.state === "error") {
            if (syncPollRef.current) clearInterval(syncPollRef.current);
            if (st.state === "done") {
              await loadTimeline();
              window.dispatchEvent(new Event("refresh_photos"));
            }
          }
        } catch { }
      }, 2000);
    } catch (e) {
      console.warn("[AuraSeek] ⚠️ Auto-scan failed:", e);
      setSyncStatus({ state: "error", processed: 0, total: 0, message: String(e) });
    }
  }, []);

  const handleFirstRunComplete = useCallback(async (dir: string) => {
    setSourceDir(dir);
    setShowFirstRun(false);
    await loadTimeline();
    triggerAutoScan();
  }, [triggerAutoScan]);

  const loadTimeline = useCallback(async () => {
    try {
      console.log("[AuraSeek] 📅 Loading timeline...");
      const groups = await AuraSeekApi.getTimeline();
      setTimelineGroups(groups);
      console.log("[AuraSeek] 📅 Timeline loaded:", groups.length, "groups");

      const allPhotos: Photo[] = await Promise.all(groups.flatMap(g =>
        g.items.map(async item => {
          // Video thumbnails are absolute paths in the thumbnails cache dir.
          // Serve them via the local Axum HTTP server to bypass WebKit asset:// restrictions.
          let thumbnailUrl: string | undefined;
          if (item.thumbnail_path) {
            if (item.thumbnail_path.startsWith("/") || item.thumbnail_path.match(/^[A-Za-z]:\\/)) {
              thumbnailUrl = await streamFileUrl(item.thumbnail_path);
            } else {
              thumbnailUrl = localFileUrl(item.thumbnail_path);
            }
          }
          return {
            id: item.media_id,
            url: localFileUrl(item.file_path),
            takenAt: item.created_at || new Date().toISOString(),
            createdAt: item.created_at || new Date().toISOString(),
            sizeBytes: 0,
            width: item.width || 0,
            height: item.height || 0,
            objects: item.objects,
            faces: item.faces,
            faceIds: item.face_ids,
            type: item.media_type as "photo" | "video",
            labels: item.objects,
            favorite: item.favorite,
            detectedObjects: item.detected_objects,
            detectedFaces: item.detected_faces,
            thumbnailUrl,
            filePath: item.file_path,
          };
        })
      ));
      setPhotos(allPhotos);
    } catch (err) {
      console.warn("[AuraSeek] ⚠️ Timeline load failed:", err);
    }
  }, []);

  const loadPeople = useCallback(async () => {
    try {
      const p = await AuraSeekApi.getPeople();
      setPeople(p);
    } catch (err) {
      console.warn("[AuraSeek] ⚠️ People load failed:", err);
    }
  }, []);

  // ── Drag-drop / paste images into the app ─────────────────────────
  useEffect(() => {
    const ALLOWED_TYPES = [
      "image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp",
      "image/tiff", "image/heic", "image/avif",
      "video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska",
      "video/webm", "video/x-m4v", "video/x-flv", "video/x-ms-wmv",
    ];
    const ALLOWED_EXTS = [
      "jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif", "heic", "avif",
      "mp4", "mov", "avi", "mkv", "webm", "m4v", "flv", "wmv",
    ];

    /** Convert a File's MIME type to an extension string */
    const mimeToExt = (mime: string): string => {
      if (mime === "image/jpeg") return "jpg";
      if (mime === "image/png") return "png";
      if (mime === "image/webp") return "webp";
      return mime.split("/")[1] ?? "jpg";
    };

    /**
     * Process a list of File objects from drag-drop or paste.
     * - If the File has a `.path` (Tauri WebView extension) → use ingest_files (faster, no base64 round-trip).
     * - Otherwise → read as ArrayBuffer, encode to base64, send via ingest_image_data.
     */
    const processFiles = async (files: File[]) => {
      const validFiles = files.filter(f =>
        ALLOWED_TYPES.includes(f.type) ||
        ALLOWED_EXTS.includes(f.name.split(".").pop()?.toLowerCase() ?? "")
      );
      if (validFiles.length === 0) return;

      const withPath: string[] = [];
      const withBlob: File[] = [];

      for (const f of validFiles) {
        const p = (f as any).path as string | undefined;
        if (p) withPath.push(p);
        else withBlob.push(f);
      }

      let newCount = 0;

      if (withPath.length > 0) {
        console.log("[AuraSeek] 📂 Ingesting", withPath.length, "files by path");
        try {
          const s = await AuraSeekApi.ingestFiles(withPath);
          newCount += s.newly_added;
        } catch (e) { console.warn("[AuraSeek] ingestFiles failed:", e); }
      }

      for (const f of withBlob) {
        console.log("[AuraSeek] 📋 Ingesting blob:", f.name || "(unnamed)", f.type);
        try {
          const buf = await f.arrayBuffer();
          const bytes = new Uint8Array(buf);
          const b64 = btoa(String.fromCharCode(...bytes));
          const ext = mimeToExt(f.type);
          const s = await AuraSeekApi.ingestImageData(b64, ext);
          newCount += s.newly_added;
        } catch (e) { console.warn("[AuraSeek] ingestImageData failed:", e); }
      }

      if (newCount > 0) {
        console.log("[AuraSeek] ✅ Ingested", newCount, "new images — refreshing timeline");
        await loadTimeline();
      }
    };

    const handleDrop = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);
      const files = Array.from(e.dataTransfer?.files ?? []);
      if (files.length > 0) processFiles(files);
    };

    const handlePaste = (e: ClipboardEvent) => {
      // Skip if the user is typing in an input field
      const active = document.activeElement;
      if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA")) return;

      const files = Array.from(e.clipboardData?.files ?? []);
      if (files.length > 0) {
        e.preventDefault();
        processFiles(files);
      }
    };

    const handleDragOver = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(true);
    };

    const handleDragLeave = (e: DragEvent) => {
      if (e.relatedTarget === null) setIsDragOver(false);
    };

    document.addEventListener("drop", handleDrop);
    document.addEventListener("dragover", handleDragOver);
    document.addEventListener("dragleave", handleDragLeave);
    document.addEventListener("paste", handlePaste);
    return () => {
      document.removeEventListener("drop", handleDrop);
      document.removeEventListener("dragover", handleDragOver);
      document.removeEventListener("dragleave", handleDragLeave);
      document.removeEventListener("paste", handlePaste);
    };
  }, [loadTimeline]);

  // ── Search ────────────────────────────────────────────────────────
  const handleSearch = useCallback(async (text: string, imagePath?: string | null) => {
    const hasFilters = activeFilters && Object.values(activeFilters).some(v => v !== undefined);
    if (!text.trim() && !imagePath && !hasFilters) {
      setSearchResults([]);
      return;
    }
    setIsSearching(true);
    const filters: ApiFilters = {
      object: activeFilters.object,
      face: activeFilters.face,
      month: activeFilters.month,
      year: activeFilters.year,
      media_type: activeFilters.mediaType,
    };
    try {
      let results: SearchResult[];
      console.log("[AuraSeek] 🔍 Searching:", { text, imagePath, filters });
      if (text && imagePath) {
        results = await AuraSeekApi.searchCombined(text, imagePath, filters);
      } else if (imagePath) {
        results = await AuraSeekApi.searchImage(imagePath, filters);
      } else if (text.trim()) {
        results = await AuraSeekApi.searchText(text, filters);
      } else if (filters.object) {
        results = await AuraSeekApi.searchObject(filters.object, filters);
      } else if (filters.face) {
        results = await AuraSeekApi.searchFace(filters.face, filters);
      } else {
        results = await AuraSeekApi.searchFilterOnly(filters);
      }
      console.log("[AuraSeek] 🔍 Found:", results.length, "results");
      setSearchResults(results);
      setRoute({ view: "search_results" });
    } catch (err) {
      console.error("[AuraSeek] ❌ Search failed:", err);
      // Dù lỗi, vẫn chuyển sang màn kết quả rỗng để người dùng thấy trạng thái.
      setSearchResults([]);
      setRoute({ view: "search_results" });
    } finally {
      // Nếu có file tạm cho search image, xoá sau khi search xong
      if (searchTempPathRef.current) {
        AuraSeekApi.deleteFile(searchTempPathRef.current).catch(() => {});
        searchTempPathRef.current = null;
      }
      setIsSearching(false);
    }
  }, [activeFilters]);

  const handleSearchSubmit = useCallback(() => {
    handleSearch(searchQuery, searchImagePath);
  }, [searchQuery, searchImagePath, handleSearch]);

  const handleFiltersChange = useCallback((filters: ActiveFilters) => {
    setActiveFilters(filters);
    const hasFilters = Object.values(filters).some(v => v !== undefined);
    if (hasFilters) {
      setTimeout(() => {
        document.getElementById("search-submit-btn")?.click();
      }, 100);
    }
  }, []);

  // ── Navigation ────────────────────────────────────────────────────
  const handleNavClick = useCallback((key: string) => {
    setRoute({ view: key });
    setSearchResults([]);
    setSearchQuery("");
    setSearchImagePath(null);
    if (key === "timeline") loadTimeline();
    if (key === "people") { loadPeople(); loadTimeline(); }
  }, [loadTimeline, loadPeople]);

  useEffect(() => {
    const handler = () => loadTimeline();
    window.addEventListener("refresh_photos", handler);

    // When backend ingest pipeline reports per-file progress (auto-scan or manual),
    // refresh the timeline so new photos/videos appear progressively.
    const unlistenPromise = listen("ingest-progress", () => {
      loadTimeline();
    });

    // Optimistic favorite handler for instant UI feedback
    const favoriteHandler = (e: any) => {
      const { id } = e.detail;
      setPhotos(prev => prev.map(p =>
        p.id === id ? { ...p, favorite: !p.favorite } : p
      ));
      setTimelineGroups(prev => prev.map(group => ({
        ...group,
        items: group.items.map(item =>
          item.media_id === id ? { ...item, favorite: !item.favorite } : item
        )
      })));
    };
    window.addEventListener("photo_toggle_favorite", favoriteHandler);

    return () => {
      window.removeEventListener("refresh_photos", handler);
      window.removeEventListener("photo_toggle_favorite", favoriteHandler);
      unlistenPromise.then(unlisten => unlisten()).catch(() => {});
    };
  }, [loadTimeline]);

  // ── Render ────────────────────────────────────────────────────────
  const renderView = () => {
    switch (route.view) {
      case "search_results":
        return (
          <SearchResultsView
            results={searchResults}
            query={searchQuery}
            isLoading={isSearching}
            onBack={() => setRoute({ view: "timeline" })}
          />
        );
      case "albums":
        return (
          <AlbumsView
            photos={photos}
            onNavigate={(payload) => setRoute({ view: "filtered", payload: { type: "album", ...payload } })}
          />
        );
      case "people":
        return (
          <PeopleView
            people={people}
            onNavigate={(payload) => setRoute({ view: "filtered", payload: { type: "person", ...payload } })}
          />
        );
      case "duplicates":
      case "duplicate_images":
        return <DuplicatesView mediaType="image" />;
      case "duplicate_videos":
        return <DuplicatesView mediaType="video" />;

      case "filtered":
        return (
          <FilteredGalleryView
            title={route.payload?.title || "Danh sách ảnh"}
            filterType={route.payload?.type}
            filterPayload={route.payload?.id}
            photos={photos}
            onBack={() => setRoute({ view: route.payload?.type === "album" ? "albums" : "people" })}
          />
        );
      case "favorite_photos":
        return (
          <FilteredGalleryView
            title="Ảnh yêu thích"
            filterType="favorites"
            filterPayload="photos"
            photos={photos}
            onBack={() => setRoute({ view: "timeline" })}
          />
        );
      case "favorite_videos":
        return (
          <FilteredGalleryView
            title="Video yêu thích"
            filterType="favorites"
            filterPayload="videos"
            photos={photos}
            onBack={() => setRoute({ view: "timeline" })}
          />
        );
      case "videos":
        return (
          <TimelineView
            timelineGroups={timelineGroups}
            photos={photos}
            searchQuery={searchQuery}
            isLoading={!isInitialized}
            selectionMode={selectionMode}
            mediaType="video"
          />
        );
      case "trash":
        return <TrashView />;
      case "hidden":
        return <HiddenView />;
      case "timeline":
      default:
        return (
          <TimelineView
            timelineGroups={timelineGroups}
            photos={photos}
            searchQuery={searchQuery}
            isLoading={!isInitialized}
            selectionMode={selectionMode}
            mediaType="photo"
          />
        );
    }
  };

  return (
    <SelectionProvider>
      <TooltipProvider>
        {showFirstRun && <FirstRunModal onComplete={handleFirstRunComplete} />}

        {/* Global drag-over indicator */}
        {isDragOver && (
          <div className="fixed inset-0 z-[100] pointer-events-none flex items-center justify-center bg-indigo-500/10 backdrop-blur-[2px] border-4 border-dashed border-indigo-400/60 rounded-xl m-2">
            <div className="bg-background/90 rounded-2xl px-8 py-5 shadow-2xl border border-indigo-400/30 text-center">
              <p className="text-lg font-semibold text-indigo-400">Thả ảnh vào đây</p>
              <p className="text-sm text-muted-foreground mt-1">Ảnh sẽ được lưu vào thư mục nguồn và xử lý AI tự động</p>
            </div>
          </div>
        )}

        {/* Model download / first-run loading screen */}
        {(needsDownload || (downloadProgress && !isInitialized)) && (
          <ModelDownloadScreen event={downloadProgress} />
        )}

        <NewLayout
          activeKey={route.view}
          onNavClick={handleNavClick}
          sourceDir={sourceDir}
          onSourceDirChange={setSourceDir}
          searchQuery={searchQuery}
          onSearchQueryChange={setSearchQuery}
          searchImagePath={searchImagePath}
          onSearchImageChange={setSearchImagePath}
          onSearchSubmit={handleSearchSubmit}
          isSearching={isSearching}
          onFiltersChange={handleFiltersChange}
          activeFilters={activeFilters}
          initError={initError}
          selectionMode={selectionMode}
          onSelectionModeChange={setSelectionMode}
          syncStatus={syncStatus}
          totalImages={photos.length}
        >
          {renderView()}
        </NewLayout>
      </TooltipProvider>
    </SelectionProvider>
  );
}

export default App;
