import React, { useRef, useState, useCallback, useEffect } from "react";
import { Search, Image as ImageIcon, Upload, X, History, Filter, MousePointerClick, RefreshCw, CheckCircle2, AlertCircle, Film, Library, Star, Users, CopySlash, Lock, Trash2, Grid3X3, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useSelection } from "@/contexts/SelectionContext";
import { FilterPanel } from "@/components/common/FilterPanel";
import { SettingsModal } from "@/components/common/SettingsModal";
import type { ActiveFilters } from "@/App";
import { AuraSeekApi, type SyncStatus } from "@/lib/api";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface NewLayoutProps {
  children: React.ReactNode;
  activeKey?: string;
  onNavClick?: (key: string) => void;
  sourceDir?: string;
  onSourceDirChange?: (dir: string) => void;
  // Topbar props
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
}

export function NewLayout({
  children,
  activeKey = "timeline",
  onNavClick,
  sourceDir = "",
  onSourceDirChange,
  // Topbar props
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
}: NewLayoutProps) {
  const { selectedIds, clearSelection } = useSelection();
  const [showFilters, setShowFilters] = useState(false);
  const [searchFocused, setSearchFocused] = useState(false);
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  const [showSettings, setShowSettings] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const composingRef = useRef(false);

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

  const handleCompositionStart = () => { composingRef.current = true; };
  const handleCompositionEnd = () => { composingRef.current = false; syncValue(); };
  const handleInput = () => { if (!composingRef.current) syncValue(); };

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
      const arrayBuffer = await file.arrayBuffer();
      const bytes = new Uint8Array(arrayBuffer);
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

  const imageFileName = searchImagePath ? searchImagePath.split(/[/\\]/).pop() || searchImagePath : null;
  const currentInputValue = searchInputRef.current?.value || searchQuery || "";

  // Menu items config (similar to old sidebar)
  const menuItems = [
    { title: "Ảnh", icon: ImageIcon, key: "timeline" },
    { title: "Video", icon: Film, key: "videos" },
    { title: "Bộ sưu tập", icon: Library, key: "albums" },
    { title: "Người", icon: Users, key: "people" },
    { title: "Thùng rác", icon: Trash2, key: "trash" },
    { title: "Kiểm tra trùng lặp", icon: CopySlash, key: "duplicates" },
  ];

  return (
    <div className="relative w-full h-screen overflow-hidden flex flex-col bg-[#f4f3ff] dark:bg-background transition-colors duration-300">
      
      {/* ── Background Wave Header ── */}
      <div className="absolute inset-x-0 top-0 h-[250px] sm:h-[300px] overflow-hidden pointer-events-none z-0">
        <div className="absolute inset-0 bg-[#0B0C10] dark:bg-[#050510]" />
        
        {/* Figma blur ellipses */}
        <div className="absolute w-[469px] h-[469px] rounded-full bg-[#ff2225] opacity-40 blur-[130px] sm:blur-[275px] -left-1/4 -top-1/2 mix-blend-screen" />
        <div className="absolute w-[477px] h-[477px] rounded-full bg-[#3e53f7] opacity-60 sm:opacity-80 blur-[130px] sm:blur-[225px] right-0 -top-1/2 mix-blend-screen mix-blend-color-dodge" />
        
        {/* Wave Separator Path */}
        <svg
          className="absolute bottom-0 w-full h-auto text-[#f4f3ff] dark:text-background"
          viewBox="0 0 1440 211"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          preserveAspectRatio="none"
        >
          <path
            d="M0 83.3138C265.5 137.892 481 213.91 762 171.691C1043 129.472 1311 26.3155 1440 0V211H0V83.3138Z"
            fill="currentColor"
          />
        </svg>
      </div>

      {/* ── Header Content (Z-INDEX 10) ── */}
      <div className="relative z-10 w-full h-[200px] sm:h-[250px] shrink-0 flex flex-col items-center justify-center pt-8">
        
        {/* Logo Text */}
        <div className="absolute top-8 sm:top-14 mx-auto w-full text-center px-4">
          <h1 
            className="text-white text-3xl sm:text-[45px] font-['Montserrat'] tracking-[0.8em] sm:tracking-[1.2em] uppercase font-light mr-[-1.2em]"
            style={{ textShadow: "0 0 40px rgba(255,255,255,0.4)" }}
          >
            AURASEEK
          </h1>
        </div>

        {/* Floating Search Bar */}
        <div className="w-full max-w-2xl px-6 mt-16 sm:mt-24 flex items-center relative z-20">
          <div className={`flex-1 relative flex items-center bg-black/15 dark:bg-black/40 backdrop-blur-md rounded-full shadow-lg border border-white/10 transition-all ${searchFocused ? 'ring-2 ring-primary bg-black/30' : 'hover:bg-black/20'}`}>
            
            {imageFileName ? (
              <div className="flex items-center gap-1.5 pl-4 pr-2 py-2 text-sm text-white shrink-0 max-w-[150px]">
                <ImageIcon className="w-4 h-4 shrink-0 opacity-70" />
                <span className="truncate">{imageFileName}</span>
                <button onClick={() => onSearchImageChange?.(null)} className="rounded-full hover:bg-white/20 p-1 shrink-0 ml-1">
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            ) : (
              <Search className="w-5 h-5 text-white/50 ml-5 shrink-0" />
            )}

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
              placeholder={searchImagePath ? "Thêm mô tả (tuỳ chọn)..." : "Tìm kiếm..."}
              className="flex-1 h-14 bg-transparent border-none text-white placeholder-white/50 px-4 font-['Roboto'] text-[18px] sm:text-[22px] outline-none"
            />

            <div className="flex items-center gap-1 pr-3">
              {(currentInputValue || searchImagePath) && (
                <button onClick={clearSearch} className="rounded-full p-2 text-white/50 hover:text-white hover:bg-white/10 transition">
                  <X className="w-4 h-4" />
                </button>
              )}
              <button onClick={() => fileInputRef.current?.click()} className="rounded-full p-2 text-white/50 hover:text-white hover:bg-white/10 transition">
                <Upload className="w-5 h-5" />
              </button>
              <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
              
              <Button
                variant="ghost" 
                size="icon"
                onClick={() => setShowFilters(true)}
                className={`rounded-full w-10 h-10 text-white/50 hover:text-white hover:bg-white/10 relative ml-1 ${hasActiveFilters ? 'text-white' : ''}`}
              >
                <Filter className="w-5 h-5" />
                {hasActiveFilters && <div className="absolute top-2 right-2 w-2 h-2 bg-primary rounded-full" />}
              </Button>
            </div>
            
            {/* Search History Dropdown */}
            {searchFocused && !currentInputValue && !searchImagePath && searchHistory.length > 0 && (
              <div className="absolute top-full mt-2 left-0 right-0 bg-white dark:bg-card border border-border/50 rounded-2xl shadow-2xl overflow-hidden py-2 px-2 animate-in fade-in slide-in-from-top-2 duration-200 z-[100]">
                <div className="text-[11px] font-extrabold text-muted-foreground/50 px-4 py-2 uppercase tracking-[0.15em]">Tìm kiếm gần đây</div>
                {searchHistory.map((q, i) => (
                  <button
                    key={i}
                    onMouseDown={(e) => {
                      e.preventDefault(); // prevent blur
                      if (searchInputRef.current) searchInputRef.current.value = q;
                      onSearchQueryChange?.(q);
                      onSearchSubmit?.();
                      setSearchFocused(false);
                    }}
                    className="w-full flex items-center gap-3 px-3 py-2 hover:bg-muted/50 rounded-xl cursor-pointer text-sm text-left text-foreground"
                  >
                    <History className="w-4 h-4 text-muted-foreground shrink-0" />
                    <span className="flex-1 truncate">{q}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Sync/Status Tags */}
        <div className="absolute bottom-4 left-6 flex items-center gap-3 text-xs sm:text-sm font-['Roboto'] font-medium text-[#322e2e]/60 dark:text-white/40">
           {initError && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-400">
              <AlertCircle className="w-3.5 h-3.5" /> <span>DB offline</span>
            </div>
           )}
           {syncStatus?.state === "syncing" && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-indigo-500/10 text-indigo-500 animate-pulse">
              <RefreshCw className="w-3.5 h-3.5 animate-spin" /> <span>Đang đồng bộ...</span>
            </div>
           )}
           {syncStatus?.state === "done" && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-600">
              <CheckCircle2 className="w-3.5 h-3.5" /> <span>Đã đồng bộ</span>
            </div>
           )}
           {syncStatus?.state === "error" && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-red-500/10 text-red-500">
              <AlertCircle className="w-3.5 h-3.5" /> <span>Lỗi đồng bộ</span>
            </div>
           )}
        </div>

        {/* Floating Menu Button (Top Right) */}
        <div className="absolute top-6 right-6 z-50">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="w-14 h-14 rounded-full bg-white/20 dark:bg-black/30 hover:bg-white/30 dark:hover:bg-black/50 backdrop-blur-md shadow-lg border border-white/20">
                <Grid3X3 className="w-6 h-6 text-white" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-64 bg-white/70 dark:bg-black/70 backdrop-blur-2xl border-white/20 dark:border-white/10 shadow-2xl rounded-2xl p-2 font-['Roboto']">
              <DropdownMenuLabel className="px-4 py-2 text-xs font-bold uppercase tracking-wider text-muted-foreground">Các chức năng</DropdownMenuLabel>
              <DropdownMenuSeparator className="bg-white/20 dark:bg-white/10" />
              {menuItems.map(item => (
                <DropdownMenuItem 
                  key={item.key} 
                  onClick={() => onNavClick?.(item.key)}
                  className={`rounded-xl px-4 py-3 cursor-pointer mb-1 transition-colors ${activeKey === item.key ? 'bg-primary/20 text-primary font-medium' : 'hover:bg-black/5 dark:hover:bg-white/10 text-zinc-700 dark:text-zinc-200 font-medium'}`}
                >
                  <item.icon className="w-5 h-5 mr-3 opacity-70" />
                  <span className="text-[16px]">{item.title}</span>
                </DropdownMenuItem>
              ))}
              <DropdownMenuSeparator className="bg-white/20 dark:bg-white/10" />
              <DropdownMenuItem 
                onClick={() => setShowSettings(true)}
                className="rounded-xl px-4 py-3 cursor-pointer text-zinc-700 dark:text-zinc-200 hover:bg-black/5 dark:hover:bg-white/10"
              >
                <Settings className="w-5 h-5 mr-3 opacity-70" />
                <span className="text-[16px]">Cài đặt</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {/* Selection mode indicator overlay */}
        {selectedIds.size > 0 && (
          <div className="absolute inset-x-0 w-full h-16 flex items-center justify-between px-6 bg-primary/20 backdrop-blur-md z-40 shadow-xl border-b border-primary/30 top-0">
             <div className="flex items-center gap-4">
                <Button variant="ghost" size="icon" onClick={clearSelection} className="rounded-full text-white hover:bg-white/20"><X className="w-5 h-5" /></Button>
                <span className="font-bold text-lg text-white">{selectedIds.size} đã chọn</span>
             </div>
          </div>
        )}
      </div>

      {/* ── Main Content Area ── */}
      <main className="flex flex-col flex-1 w-full overflow-hidden relative z-10">
        {children}
      </main>

      <FilterPanel open={showFilters} onOpenChange={setShowFilters} activeFilters={activeFilters} onFiltersChange={onFiltersChange} />
      <SettingsModal open={showSettings} onOpenChange={setShowSettings} currentSourceDir={sourceDir} onSourceDirChange={onSourceDirChange} />
    </div>
  );
}
