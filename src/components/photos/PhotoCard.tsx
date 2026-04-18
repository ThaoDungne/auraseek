import { useState, useRef, useEffect } from "react";
import type { Photo } from "@/types/photo.type";
import { cn } from "@/lib/utils";
import { useSelection } from "@/contexts/SelectionContext";
import { Heart, CheckCircle2, Circle, Play } from "lucide-react";
import { AuraSeekApi } from "@/lib/api";
import { SegmentOverlay } from "./SegmentOverlay";

type PhotoCardProps = {
  photo: Photo;
  onClick?: () => void;
  selectionMode?: boolean;
  showBbox?: boolean;
  overlayShowFaces?: boolean;
  overlayShowLabels?: boolean;
  className?: string;
};

export function PhotoCard({
  photo,
  onClick,
  selectionMode     = false,
  showBbox          = true,
  overlayShowFaces  = true,
  overlayShowLabels = true,
  className,
}: PhotoCardProps) {

  const { selectedIds, toggleSelection } = useSelection();
  const isSelected   = selectedIds.has(photo.id);
  const [isFavorite, setIsFavorite] = useState(photo.favorite ?? false);
  const [hovered,    setHovered]    = useState(false);

  const isVideo = photo.type === "video";

  // Ref for the image element (used by SegmentOverlay for dimensions)

  const imgRef = useRef<HTMLImageElement>(null);

  // Track rendered dimensions for the overlay
  const [displayW, setDisplayW] = useState(0);
  const [displayH, setDisplayH] = useState(0);

  useEffect(() => {
    const el = imgRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setDisplayW(el.clientWidth);
      setDisplayH(el.clientHeight);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const handleSelect = (e: React.MouseEvent) => {
    e.stopPropagation();
    toggleSelection(photo.id);
  };

  const handleFavorite = async (e: React.MouseEvent) => {
    e.stopPropagation();
    const next = !isFavorite;
    setIsFavorite(next);
    // Notify other components (App.tsx listener, FullScreenViewer, etc.)
    window.dispatchEvent(new CustomEvent("photo_toggle_favorite", { detail: { id: photo.id } }));
    try {
      await AuraSeekApi.toggleFavorite(photo.id);
    } catch {
      setIsFavorite(!next);
      window.dispatchEvent(new Event("refresh_photos"));
    }
  };

  const hasOverlays =
    !isVideo && (
      (photo.detectedObjects && photo.detectedObjects.length > 0) ||
      (photo.detectedFaces   && photo.detectedFaces.length   > 0)
    );

  const imgNaturalW = photo.width  || imgRef.current?.naturalWidth  || 0;
  const imgNaturalH = photo.height || imgRef.current?.naturalHeight || 0;

  return (
    <button
      type="button"
      onClick={selectionMode ? handleSelect : onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={cn("group relative block overflow-hidden bg-zinc-200 dark:bg-zinc-800 rounded-2xl shadow-sm transition-shadow hover:shadow-xl", className || "w-full h-full")}
    >
      <div className={cn(
        "w-full h-full transition-all duration-300 ease-out relative",
        isSelected && selectionMode ? "p-3" : "p-0"
      )}>

        {/* ── Video — show static thumbnail image in grid ──────── */}
        {isVideo ? (
          <img
            ref={imgRef}
            src={photo.thumbnailUrl || photo.url}
            alt="Video"
            className={cn(
              "h-full w-full select-none object-cover transition-transform duration-700 ease-out",
              !(isSelected && selectionMode) && "group-hover:scale-110",
              isSelected && selectionMode && "rounded-xl"
            )}
            draggable={false}
          />
        ) : (
          /* ── Image ──────────────────────────────────────────── */
          <img
            ref={imgRef}
            src={photo.url}
            alt="Photo"
            className={cn(
              "h-full w-full select-none object-cover transition-transform duration-700 ease-out",
              !(isSelected && selectionMode) && "group-hover:scale-110",
              isSelected && selectionMode && "rounded-xl"
            )}
            draggable={false}
          />
        )}

        {/* ── Segmentation overlay (images only) ───────────────── */}
        {showBbox && hovered && hasOverlays && displayW > 0 && imgNaturalW > 0 && (
          <div className="absolute inset-0 pointer-events-none z-5">
            <SegmentOverlay
              detectedObjects={photo.detectedObjects}
              detectedFaces={photo.detectedFaces}
              imgNaturalW={imgNaturalW}
              imgNaturalH={imgNaturalH}
              displayW={displayW}
              displayH={displayH}
              objectFit="cover"
              showFaces={overlayShowFaces}
              showLabels={overlayShowLabels}
              showMasks
              showBoxes
            />
          </div>
        )}

        {/* ── Video play badge ──────────────────────────────────── */}
        {isVideo && !hovered && (
          <div className="absolute bottom-2 right-2 z-10 flex items-center gap-1 bg-black/60 text-white text-[10px] px-1.5 py-0.5 rounded-full backdrop-blur-sm">
            <Play className="w-2.5 h-2.5 fill-white" />
            <span>VIDEO</span>
          </div>
        )}

        {isSelected && selectionMode && (
          <div className="pointer-events-none absolute inset-0 bg-black/20 rounded-lg m-3" />
        )}
      </div>

      {/* ── Selection checkbox ────────────────────────────────── */}
      {selectionMode && (
        <div
          role="button"
          onClick={handleSelect}
          className={cn(
            "absolute left-2 top-2 z-10 rounded-full transition-all duration-200",
            isSelected ? "opacity-100 scale-100" : "opacity-0 group-hover:opacity-100 scale-95 hover:scale-100"
          )}
        >
          {isSelected ? (
            <CheckCircle2 className="w-6 h-6 text-primary bg-white rounded-full border-none" />
          ) : (
            <Circle className="w-6 h-6 text-white/80 hover:text-white fill-black/20" />
          )}
        </div>
      )}

      {/* ── Favourite heart ───────────────────────────────────── */}
      <div
        role="button"
        onClick={handleFavorite}
        className={cn(
          "absolute right-2 top-2 z-10 rounded-full p-1 transition-all duration-200",
          isFavorite
            ? "opacity-100"
            : "opacity-0 group-hover:opacity-100 hover:scale-110"
        )}
      >
        <Heart className={cn(
          "w-5 h-5 transition-colors drop-shadow-md",
          isFavorite
            ? "fill-red-500 text-red-500"
            : "fill-black/30 text-white/90 hover:fill-red-500 hover:text-red-500"
        )} />
      </div>
    </button>
  );
}
