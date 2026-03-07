import { useState, useEffect, useCallback } from "react";
import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "./components/layout/AppSidebar";
import { AppTopbar } from "./components/layout/AppTopbar";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SelectionProvider } from "@/contexts/SelectionContext";
import { TimelineView } from "@/views/timeline";
import { PeopleView } from "@/views/people/PeopleView";
import { DuplicatesView } from "@/views/duplicates/DuplicatesView";
import { AlbumsView } from "@/views/albums/AlbumsView";
import { FilteredGalleryView } from "@/views/gallery/FilteredGalleryView";
import { SearchResultsView } from "@/views/search/SearchResultsView";
import { AuraSeekApi, localFileUrl, type SearchResult, type TimelineGroup, type PersonGroup, type SearchFilters as ApiFilters } from "@/lib/api";
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
  const [activeFilters, setActiveFilters] = useState<ActiveFilters>({});

  // Data state
  const [timelineGroups, setTimelineGroups] = useState<TimelineGroup[]>([]);
  const [people, setPeople] = useState<PersonGroup[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [selectionMode, setSelectionMode] = useState(false);

  // ── Init ──────────────────────────────────────────────────────────
  useEffect(() => {
    const initialize = async () => {
      console.log("[AuraSeek] 🚀 Initializing app...");
      setIsInitialized(true);

      if (!('__TAURI_INTERNALS__' in window)) {
        const msg = "App phải chạy trong Tauri WebView. Dùng lệnh: cargo tauri dev";
        console.warn("[AuraSeek] ⚠️", msg);
        setInitError(msg);
        return;
      }

      try {
        // Race init against a 15s timeout to prevent permanent hang
        const initPromise = AuraSeekApi.init();
        const timeoutPromise = new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("Init timeout: backend không phản hồi sau 15s. Kiểm tra SurrealDB server.")), 15000)
        );
        const msg = await Promise.race([initPromise, timeoutPromise]);
        console.log("[AuraSeek] ✅ Engine + DB ready:", msg);
        setInitError(null);
        await loadTimeline();
      } catch (err: any) {
        console.warn("[AuraSeek] ⚠️ Init warning (app still usable):", err);
        setInitError(String(err));
      }
    };
    initialize();
  }, []);

  const loadTimeline = useCallback(async () => {
    try {
      console.log("[AuraSeek] 📅 Loading timeline...");
      const groups = await AuraSeekApi.getTimeline();
      setTimelineGroups(groups);
      console.log("[AuraSeek] 📅 Timeline loaded:", groups.length, "groups");

      // Convert to legacy Photo[] format
      const allPhotos: Photo[] = groups.flatMap(g =>
        g.items.map(item => ({
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
        }))
      );
      setPhotos(allPhotos);
    } catch (err) {
      console.warn("[AuraSeek] ⚠️ Timeline load failed:", err);
    }
  }, []);

  const loadPeople = useCallback(async () => {
    try {
      console.log("[AuraSeek] 👤 Loading people...");
      const p = await AuraSeekApi.getPeople();
      setPeople(p);
      console.log("[AuraSeek] 👤 People loaded:", p.length, "groups");
    } catch (err) {
      console.warn("[AuraSeek] ⚠️ People load failed:", err);
    }
  }, []);

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
    } finally {
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
    if (key === "people") loadPeople();
  }, [loadTimeline, loadPeople]);

  useEffect(() => {
    const handler = () => loadTimeline();
    window.addEventListener("refresh_photos", handler);
    return () => window.removeEventListener("refresh_photos", handler);
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
        return <DuplicatesView />;
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
      case "favorites":
        return (
          <FilteredGalleryView
            title="Ảnh yêu thích"
            filterType="favorites"
            photos={photos}
            onBack={() => setRoute({ view: "timeline" })}
          />
        );
      case "recent":
        return (
          <FilteredGalleryView
            title="Tin mới"
            subtitle="Trong vòng 7 ngày qua"
            filterType="recent"
            photos={photos}
            onBack={() => setRoute({ view: "timeline" })}
          />
        );
      case "timeline":
      default:
        return (
          <TimelineView
            timelineGroups={timelineGroups}
            photos={photos}
            searchQuery={searchQuery}
            isLoading={!isInitialized}
            selectionMode={selectionMode}
          />
        );
    }
  };

  return (
    <SelectionProvider>
      <TooltipProvider>
        <SidebarProvider>
          <AppSidebar activeKey={route.view} onNavClick={handleNavClick} />

          <main className="flex flex-col flex-1 h-screen overflow-hidden">
            <AppTopbar
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
            />
            {renderView()}
          </main>
        </SidebarProvider>
      </TooltipProvider>
    </SelectionProvider>
  );
}

export default App;
