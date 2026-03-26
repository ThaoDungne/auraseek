import { SidebarTrigger } from "@/components/ui/sidebar";
import {
  Search, Filter, Moon, Sun, X, Share2, Plus, Trash2,
  History, Image as ImageIcon, Upload, AlertCircle, MousePointerClick,
  RefreshCw, CheckCircle2
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "../theme-provider";
import { useSelection } from "@/contexts/SelectionContext";
import { FilterPanel } from "@/components/common/FilterPanel";
import { useState, useRef, useCallback, useEffect } from "react";
import type { ActiveFilters } from "@/App";
import { AuraSeekApi, type SyncStatus } from "@/lib/api";

interface AppTopbarProps {
  totalImages?: number;
  searchQuery?: string;
  onSearchQueryChange?: (q: string) => void;
  searchImagePath?: string | null;
  onSearchImageChange?: (path: string | null) => void;
  onSearchSubmit?: () => void;
  isSearching?: boolean;
  onFiltersChange?: (filters: ActiveFilters) => void;
  activeFilters?: ActiveFilters;
  initError?: string | null;
  selectionMode?: boolean;
  onSelectionModeChange?: (mode: boolean) => void;
  syncStatus?: SyncStatus | null;
  onReload?: () => void;
}

export function AppTopbar({
  totalImages = 0,
  searchQuery = "",
  onSearchQueryChange,
  searchImagePath,
  onSearchImageChange,
  onSearchSubmit,
  isSearching = false,
  onFiltersChange,
  activeFilters,
  initError,
  selectionMode = false,
  onSelectionModeChange,
  syncStatus,
  onReload,
}: AppTopbarProps) {
  const { theme, setTheme } = useTheme();
  const { selectedIds, clearSelection } = useSelection();
  const [showFilters, setShowFilters] = useState(false);
  const [searchFocused, setSearchFocused] = useState(false);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const composingRef = useRef(false);

  // Sync external search query clears into the uncontrolled input
  useEffect(() => {
    if (searchInputRef.current && searchInputRef.current.value !== (searchQuery || "")) {
      searchInputRef.current.value = searchQuery || "";
    }
  }, [searchQuery]);

  const hasActiveFilters = activeFilters && Object.values(activeFilters).some(v => v !== undefined);

  const syncValue = useCallback(() => {
    const val = searchInputRef.current?.value ?? "";
    onSearchQueryChange?.(val);
  }, [onSearchQueryChange]);

  const handleFocus = useCallback(async () => {
    setSearchFocused(true);
    try {
      const history = await AuraSeekApi.getSearchHistory(8);
      setSearchHistory(history.map((h: any) => h.query).filter(Boolean));
    } catch {
      setSearchHistory(["Chó chạy trên cỏ", "Biển đà nẵng", "Gia đình"]);
    }
  }, []);

  const handleCompositionStart = () => {
    composingRef.current = true;
  };

  const handleCompositionEnd = () => {
    composingRef.current = false;
    syncValue();
  };

  const handleInput = () => {
    if (!composingRef.current) {
      syncValue();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (composingRef.current || e.nativeEvent.isComposing || e.keyCode === 229) return;
    if (e.key === "Enter") {
      syncValue();
      onSearchSubmit?.();
      setSearchFocused(false);
    }
  };

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      // Đọc bytes từ File (browser API)
      const arrayBuffer = await file.arrayBuffer();
      const bytes = new Uint8Array(arrayBuffer);

      // Gửi bytes sang backend để lưu file tạm, backend trả về đường dẫn tuyệt đối.
      const ext = file.name.split(".").pop() || "jpg";
      const tmpPath = await AuraSeekApi.saveSearchImageTmp(Array.from(bytes), ext);

      onSearchImageChange?.(tmpPath);
      (window as any).__AURASEEK_SEARCH_TMP_PATH__ = tmpPath;
    } catch (err) {
      console.error("[AuraSeek] ❌ Error saving temp search image:", err);
    }
  };

  const clearSearch = () => {
    if (searchInputRef.current) searchInputRef.current.value = "";
    onSearchQueryChange?.("");
    onSearchImageChange?.(null);
  };

  if (selectedIds.size > 0) {
    return (
      <div className="flex items-center gap-4 px-4 h-16 shrink-0 top-0 sticky z-10 bg-primary/10 saturate-150 backdrop-blur-2xl supports-[backdrop-filter]:bg-primary/20 transition-colors">
        <Button variant="ghost" size="icon" onClick={clearSelection} className="rounded-full text-primary hover:bg-primary/20">
          <X className="w-5 h-5" />
        </Button>
        <span className="font-bold text-[1.2rem] tracking-tight text-primary">{selectedIds.size} đã chọn</span>
        <div className="flex-1" />
        <div className="flex items-center gap-2 text-primary">
          <Button variant="ghost" size="icon" className="rounded-full hover:bg-primary/20 text-primary"><Share2 className="w-5 h-5" /></Button>
          <Button variant="ghost" size="icon" className="rounded-full hover:bg-primary/20 text-primary"><Plus className="w-5 h-5" /></Button>
          <Button variant="ghost" size="icon" className="rounded-full hover:bg-primary/20 text-primary"><Trash2 className="w-5 h-5" /></Button>
        </div>
      </div>
    );
  }

  const imageFileName = searchImagePath
    ? searchImagePath.split(/[/\\]/).pop() || searchImagePath
    : null;

  const currentInputValue = searchInputRef.current?.value || searchQuery || "";

  return (
    <div className="flex items-center gap-4 px-4 h-16 shrink-0 top-0 sticky z-[40] bg-background/50 saturate-150 backdrop-blur-2xl supports-[backdrop-filter]:bg-background/20 transition-colors">

      <SidebarTrigger className="shrink-0 rounded-full w-10 h-10 ml-1 hover:bg-muted text-muted-foreground" />

      {initError && (
        <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400 text-[11px] font-extrabold uppercase tracking-widest">
          <AlertCircle className="w-3.5 h-3.5" />
          <span>DB offline</span>
        </div>
      )}

      <div className="flex-1 max-w-3xl flex items-center gap-2 mx-auto">

        {imageFileName && (
          <div className="flex items-center gap-1.5 pl-2 pr-1 py-1 rounded-full bg-primary/10 border border-primary/20 text-xs text-primary shrink-0 max-w-[140px]">
            <ImageIcon className="w-3 h-3 shrink-0" />
            <span className="truncate">{imageFileName}</span>
            <button
              onClick={() => onSearchImageChange?.(null)}
              className="rounded-full hover:bg-primary/20 p-0.5 ml-0.5 shrink-0"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        )}

        {/* Search Input — uncontrolled for IME compatibility (Vietnamese Telex/VNI) */}
        <div className="relative flex-1 group z-50">
          <Search className={`absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 transition-colors ${searchFocused ? 'text-primary' : 'text-muted-foreground'}`} />
          <input
            ref={searchInputRef}
            type="text"
            id="search-input"
            defaultValue={searchQuery}
            onInput={handleInput}
            onCompositionStart={handleCompositionStart}
            onCompositionEnd={handleCompositionEnd}
            onFocus={handleFocus}
            onBlur={() => { syncValue(); setTimeout(() => setSearchFocused(false), 200); }}
            onKeyDown={handleKeyDown}
            placeholder={searchImagePath ? "Thêm mô tả (tuỳ chọn)..." : "Tìm theo văn bản, đối tượng, tháng..."}
            className={`w-full h-12 bg-muted/60 hover:bg-muted focus:bg-background border border-transparent focus:border-border/50 border-input pl-12 pr-12 text-[15px] font-medium outline-none transition-all ${searchFocused ? 'rounded-t-2xl shadow-lg border-b-border/20 bg-background' : 'rounded-full shadow-sm focus:shadow-md'}`}
          />

          <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1">
            {(currentInputValue || searchImagePath) && (
              <button onClick={clearSearch} className="rounded-full p-1 hover:bg-muted text-muted-foreground">
                <X className="w-4 h-4" />
              </button>
            )}
            <button
              onClick={() => fileInputRef.current?.click()}
              className="rounded-full p-1 hover:bg-muted text-muted-foreground hover:text-primary transition-colors"
              title="Tìm kiếm bằng ảnh"
            >
              <Upload className="w-4 h-4" />
            </button>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleImageUpload}
          />

          {searchFocused && !currentInputValue && !searchImagePath && searchHistory.length > 0 && (
            <div className="absolute top-full left-0 right-0 bg-background border border-t-0 border-border/50 rounded-b-2xl shadow-2xl overflow-hidden py-2 px-2 animate-in fade-in slide-in-from-top-2 duration-200">
              <div className="text-[11px] font-extrabold text-muted-foreground/50 px-4 py-2 uppercase tracking-[0.15em]">Tìm kiếm gần đây</div>
              {searchHistory.map((q, i) => (
                <button
                  key={i}
                  onClick={() => {
                    if (searchInputRef.current) searchInputRef.current.value = q;
                    onSearchQueryChange?.(q);
                    onSearchSubmit?.();
                  }}
                  className="w-full flex items-center gap-3 px-3 py-2 hover:bg-muted/50 rounded-xl cursor-pointer text-sm text-left"
                >
                  <History className="w-4 h-4 text-muted-foreground shrink-0" />
                  <span className="flex-1 truncate">{q}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        <Button
          id="search-submit-btn"
          onClick={() => { syncValue(); onSearchSubmit?.(); setSearchFocused(false); }}
          disabled={isSearching || (!currentInputValue.trim() && !searchImagePath && !hasActiveFilters)}
          className="h-10 px-6 rounded-full shrink-0 font-bold text-[13px]"
          size="sm"
        >
          {isSearching ? (
            <span className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Đang tìm...
            </span>
          ) : (
            "Tìm kiếm"
          )}
        </Button>

        <Button
          variant="ghost"
          size="icon"
          onClick={() => setShowFilters(true)}
          className={`shrink-0 rounded-full w-10 h-10 hover:bg-muted text-muted-foreground hover:text-foreground relative ${hasActiveFilters ? 'text-primary' : ''}`}
        >
          <Filter className="w-5 h-5" />
          {hasActiveFilters && (
            <div className="absolute top-2 right-2.5 w-2 h-2 bg-primary rounded-full ring-2 ring-background" />
          )}
        </Button>

        <Button
          variant={selectionMode ? "default" : "ghost"}
          size="icon"
          onClick={() => {
            if (selectionMode) clearSelection();
            onSelectionModeChange?.(!selectionMode);
          }}
          className={`shrink-0 rounded-full w-10 h-10 ${selectionMode ? '' : 'hover:bg-muted text-muted-foreground hover:text-foreground'}`}
          title={selectionMode ? "Thoát chọn ảnh" : "Chọn ảnh"}
        >
          <MousePointerClick className="w-5 h-5" />
        </Button>
      </div>

      <div className="flex items-center gap-3 shrink-0 mr-2 text-sm text-muted-foreground">
        {/* Sync status indicator */}
        {syncStatus && syncStatus.state === "syncing" && (
          <div className="hidden sm:flex items-center gap-1.5 px-2 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-500 text-xs animate-pulse">
            <RefreshCw className="w-3 h-3 animate-spin" />
            <span>Đang đồng bộ dữ liệu</span>
          </div>
        )}
        {syncStatus && syncStatus.state === "done" && (
          <div className="hidden sm:flex items-center gap-1.5 px-2 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 text-xs">
            <CheckCircle2 className="w-3 h-3" />
            <span>Đã đồng bộ</span>
          </div>
        )}
        {syncStatus && syncStatus.state === "error" && (
          <div className="hidden sm:flex items-center gap-1.5 px-2 py-1 rounded-full bg-red-500/10 border border-red-500/20 text-red-500 text-xs" title={syncStatus.message}>
            <AlertCircle className="w-3 h-3" />
            <span>Lỗi đồng bộ</span>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          className="rounded-full w-8 h-8 ml-1"
          onClick={onReload}
          title="Tải lại dữ liệu"
        >
          <RefreshCw className="w-4 h-4 text-muted-foreground" />
        </Button>
        <span className="hidden sm:inline-block font-medium">{totalImages > 0 ? `${totalImages} Ảnh & Video` : ""}</span>
        <Button
          variant="ghost"
          size="icon"
          className="rounded-full w-10 h-10 ml-2"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        >
          {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </Button>
      </div>

      <FilterPanel
        open={showFilters}
        onOpenChange={setShowFilters}
        activeFilters={activeFilters}
        onFiltersChange={onFiltersChange}
      />
    </div>
  );
}
