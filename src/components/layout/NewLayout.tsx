import React, { useRef, useState, useCallback, useEffect } from "react";
import {
  Search, Image as ImageIcon, Upload, X, History, Filter,
  RefreshCw, CheckCircle2, AlertCircle, Film, Library, Users,
  CopySlash, Trash2, Grid3X3, Settings, LayoutGrid,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useSelection } from "@/contexts/SelectionContext";
import { FilterPanel } from "@/components/common/FilterPanel";
import { SettingsModal } from "@/components/common/SettingsModal";
import { SpaceNodes } from "@/components/common/SpaceNodes";
import type { ActiveFilters } from "@/App";
import { cn } from "@/lib/utils";
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
      setSearchHistory(["Sinh nhật 15/06/1995", "Kỷ niệm 20/12/1988", "Ngày sinh 05/02/2000"]);
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

  // Menu items config (Floating charcoal style)
  const menuItems = [
    { title: "Tất cả", icon: LayoutGrid, key: "all" },
    { title: "Ảnh", icon: ImageIcon, key: "timeline" },
    { title: "Video", icon: Film, key: "videos" },
    { title: "Bộ sưu tập", icon: Library, key: "albums" },
    { title: "Thùng rác", icon: Trash2, key: "trash" },
    { title: "Kiểm tra trùng lặp", icon: CopySlash, key: "duplicates" },
  ];

  return (
    <div className="relative w-full h-screen overflow-hidden flex flex-col bg-white">
      
      {/* ═══════════════ DECORATIVE GLOW BLOBS ═══════════════ */}
      <div className="absolute top-0 left-0 w-[400px] h-[400px] bg-red-500/10 blur-[120px] rounded-full -translate-x-1/2 -translate-y-1/2 pointer-events-none z-0" />
      <div className="absolute top-0 right-0 w-[400px] h-[400px] bg-blue-500/10 blur-[120px] rounded-full translate-x-1/2 -translate-y-1/2 pointer-events-none z-0" />

      {/* ═══════════════ HEADER (260px) ═══════════════ */}
      <div className="relative shrink-0 overflow-visible flex flex-col justify-between" style={{ height: "280px", zIndex: 10 }}>

        {/* Premium Dark Gradient */}
        <div className="absolute inset-0" style={{
          background: "linear-gradient(135deg, #020205 0%, #080a1a 40%, #1a0808 100%)",
        }} />

        {/* Cosmic Nodes Network Canvas */}
        <SpaceNodes />

        {/* Aura Blobs for High-Tech feel */}
        <div className="absolute w-[500px] h-[500px] rounded-full opacity-30 blur-[100px] -left-40 -top-40 pointer-events-none"
             style={{ background: "radial-gradient(circle, #3e53f7 0%, transparent 70%)" }} />
        <div className="absolute w-[400px] h-[400px] rounded-full opacity-20 blur-[80px] right-0 top-0 pointer-events-none"
             style={{ background: "radial-gradient(circle, #ff2225 0%, transparent 70%)" }} />

        {/* Brand Row */}
        <div className="relative z-10 flex items-center h-26 px-3 pt-4">
          {/* Logo: Logo.png */}
          <div className="flex items-center gap-4 mt-18">
            <div className="relative w-[200px] h-[200px] flex items-center justify-center shrink-0">
               <img src="/logo/Logo.png" alt="AuraSeek Logo" className="w-full h-full object-contain drop-shadow-[0_0_15px_rgba(255,255,255,0.2)]" />
            </div>

            {/* Typography: A U R A S E E K (Premium kerning) */}
            <h1 className="text-white font-['Montserrat'] text-[50px] tracking-[1.2em] uppercase font-light translate-x-[0.1em]"
                style={{ textShadow: "0 0 20px rgba(255,255,255,0.4)" }}>
              AURASEEK
            </h1>
          </div>

          {/* Sync Status Overlay (Right) */}
          <div className="ml-auto hidden sm:flex items-center gap-4 text-[10px] font-bold tracking-widest text-white/50 uppercase">
             {syncStatus?.state === "syncing" && (
                <div className="flex items-center gap-2 text-blue-400">
                   <RefreshCw className="w-3 h-3 animate-spin" /> <span>Đang đồng bộ</span>
                </div>
             )}
             {syncStatus?.state === "done" && (
                <div className="flex items-center gap-2 text-emerald-400">
                   <CheckCircle2 className="w-3 h-3" /> <span>Hệ thống sẵn sàng</span>
                </div>
             )}
          </div>
        </div>

        {/* Wave Divider — Tall and distinct wave */}
        <div className="absolute bottom-0 left-0 w-full" style={{ height: "100px", zIndex: 30 }}>
            <svg 
               viewBox="0 0 1440 100" 
               preserveAspectRatio="none" 
               className="block w-full h-full"
               xmlns="http://www.w3.org/2000/svg"
               style={{ filter: "drop-shadow(0 -4px 16px rgba(0,0,0,0.1))" }}
            >
                <path 
                  d="M0 60 C240 20, 480 90, 720 50 C960 10, 1200 80, 1440 40 L1440 100 L0 100 Z" 
                  fill="white"
                />
            </svg>
        </div>

        {/* Selection mode indicator overlay */}
        {selectedIds.size > 0 && (
          <div className="absolute inset-x-0 w-full h-20 flex items-center justify-between px-8 bg-primary/40 backdrop-blur-xl z-[100] shadow-2xl top-0">
             <div className="flex items-center gap-6">
                <Button variant="ghost" size="icon" onClick={clearSelection} className="rounded-full text-white hover:bg-white/20"><X className="w-6 h-6" /></Button>
                <span className="font-bold text-xl text-white tracking-wide">{selectedIds.size} mục đã chọn</span>
             </div>
          </div>
        )}
      </div>

      {/* ═══════════════ SEARCH & ACTIONS ═══════════════ */}
      <div className="relative z-20 w-full flex justify-center px-8 py-4 bg-white">
        <div className="w-full max-w-4xl flex items-center gap-4">
          
          {/* Centered Pill Search Bar */}
          <div className="flex-1 relative group">
            <div className={cn(
              "flex items-center rounded-full border transition-all duration-500 px-5 cursor-text",
              searchFocused
                ? "bg-gray-200 border-primary/40 ring-[6px] ring-primary/10 shadow-[0_12px_40px_rgba(0,0,0,0.1)] scale-[1.01]"
                : "bg-gray-200 border-zinc-200 shadow-[0_4px_16px_rgba(0,0,0,0.04)] hover:border-primary/30 hover:shadow-[0_8px_32px_rgba(0,0,0,0.08)] hover:-translate-y-0.5"
            )}
            onClick={() => searchInputRef.current?.focus()}>
              <Search className={cn("w-5 h-5 transition-colors duration-300", searchFocused ? "text-primary" : "text-zinc-400 group-hover:text-primary/70")} />
              
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
                placeholder="Tìm kiếm..."
                className="flex-1 h-14 bg-transparent border-none text-black placeholder-zinc-400 px-4 font-['Roboto'] text-lg outline-none"
              />

              <div className="flex items-center gap-2">
                {(currentInputValue || searchImagePath) && (
                  <button onClick={clearSearch} className="rounded-full p-2 text-zinc-400 hover:text-zinc-900 transition">
                    <X className="w-4 h-4" />
                  </button>
                )}
                <button onClick={() => fileInputRef.current?.click()} className="rounded-full p-2 text-zinc-400 hover:text-primary transition" title="Tìm kiếm theo hình ảnh">
                  <Upload className="w-5 h-5" />
                </button>
                <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
                <Button
                  variant="ghost" 
                  size="icon"
                  onClick={() => setShowFilters(true)}
                  className={cn("rounded-full w-10 h-10 transition-colors relative", hasActiveFilters ? "text-primary bg-primary/10" : "text-zinc-400 hover:text-primary hover:bg-zinc-100/50")}
                >
                  <Filter className="w-5 h-5" />
                  {hasActiveFilters && <div className="absolute top-2 right-2 w-2 h-2 bg-primary rounded-full shadow-[0_0_10px_rgba(var(--primary),0.5)]" />}
                </Button>
              </div>
            </div>

            {/* Search History Dropdown */}
            {searchFocused && !currentInputValue && !searchImagePath && searchHistory.length > 0 && (
              <div className="absolute top-full mt-3 left-0 right-0 bg-white dark:bg-zinc-900 border border-zinc-100 dark:border-white/5 rounded-3xl shadow-2xl overflow-hidden py-3 px-3 z-[150] animate-in fade-in slide-in-from-top-4 duration-300">
                <div className="text-[11px] font-black text-zinc-300 dark:text-white/20 px-4 py-2 uppercase tracking-[0.2em]">Tìm kiếm gần đây</div>
                {searchHistory.map((q, i) => (
                  <button
                    key={i}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      if (searchInputRef.current) searchInputRef.current.value = q;
                      onSearchQueryChange?.(q);
                      onSearchSubmit?.();
                      setSearchFocused(false);
                    }}
                    className="w-full flex items-center gap-4 px-4 py-3 hover:bg-zinc-50 dark:hover:bg-white/5 rounded-2xl cursor-pointer text-base text-left text-zinc-700 dark:text-zinc-300 transition-colors"
                  >
                    <History className="w-4 h-4 text-zinc-300" />
                    <span className="flex-1 truncate">{q}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Grid Menu Trigger — Glassmorphism */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button 
                variant="ghost" 
                size="icon" 
                className="w-14 h-14 rounded-full bg-gray-200 border border-zinc-200 shadow-[0_4px_16px_rgba(0,0,0,0.04)] hover:bg-gray-900 hover:border-primary/30 hover:shadow-[0_8px_32px_rgba(0,0,0,0.08)] hover:-translate-y-0.5 text-zinc-600 hover:text-primary transition-all duration-300 active:scale-95"
              >
                <Grid3X3 className="w-6 h-6" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent 
              align="end" 
              className="w-72 mt-4 bg-white/70 dark:bg-black/40 backdrop-blur-3xl border-white/20 dark:border-white/10 shadow-[0_20px_50px_rgba(0,0,0,0.2)] rounded-[32px] p-3 font-['Roboto']"
            >
              <DropdownMenuLabel className="px-5 py-3 text-xs font-black uppercase tracking-[0.25em] text-white/30">Danh mục chính</DropdownMenuLabel>
              <div className="space-y-1">
                {menuItems.map(item => (
                  <DropdownMenuItem 
                    key={item.key} 
                    onClick={() => onNavClick?.(item.key)}
                    className={cn(
                      "rounded-2xl px-5 py-4 cursor-pointer transition-all duration-200 group flex items-center gap-4",
                      activeKey === item.key 
                        ? "bg-white/15 text-white shadow-inner" 
                        : "hover:bg-white/10 text-white/70 hover:text-white"
                    )}
                  >
                    <div className={cn(
                      "p-2 rounded-xl transition-colors",
                      activeKey === item.key ? "bg-primary/20" : "bg-white/5 group-hover:bg-white/10"
                    )}>
                      <item.icon className="w-5 h-5" />
                    </div>
                    <span className="text-[17px] font-medium">{item.title}</span>
                  </DropdownMenuItem>
                ))}
              </div>
              <div className="h-px bg-white/5 my-3" />
              <DropdownMenuItem 
                onClick={() => setShowSettings(true)}
                className="rounded-2xl px-5 py-4 cursor-pointer text-white/50 hover:text-white hover:bg-white/5 flex items-center gap-4 transition-colors"
              >
                <div className="p-2 rounded-xl bg-white/5 group-hover:bg-white/10">
                  <Settings className="w-5 h-5" />
                </div>
                <span className="text-[17px]">Cài đặt</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* ═══════════════ MAIN CONTENT ═══════════════ */}
      <main className="flex flex-col flex-1 w-full overflow-hidden relative z-10 bg-white">
        <div className="w-full h-full">
          {children}
        </div>
      </main>

      <FilterPanel open={showFilters} onOpenChange={setShowFilters} activeFilters={activeFilters} onFiltersChange={onFiltersChange} />
      <SettingsModal open={showSettings} onOpenChange={setShowSettings} currentSourceDir={sourceDir} onSourceDirChange={onSourceDirChange} />
    </div>
  );
}
