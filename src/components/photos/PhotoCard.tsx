import { useState, useRef } from "react";
import type { Photo } from "@/types/photo.type";
import { cn } from "@/lib/utils";
import { useSelection } from "@/contexts/SelectionContext";
import { Heart, CheckCircle2, Circle } from "lucide-react";
import { AuraSeekApi } from "@/lib/api";

type PhotoCardProps = {
  photo: Photo;
  onClick?: () => void;
  selectionMode?: boolean;
  showBbox?: boolean;
};

export function PhotoCard({ photo, onClick, selectionMode = false, showBbox = true }: PhotoCardProps) {
  const { selectedIds, toggleSelection } = useSelection();
  const isSelected = selectedIds.has(photo.id);
  const [isFavorite, setIsFavorite] = useState(photo.favorite ?? false);
  const [hovered, setHovered] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  const handleSelect = (e: React.MouseEvent) => {
    e.stopPropagation();
    toggleSelection(photo.id);
  };

  const handleFavorite = async (e: React.MouseEvent) => {
    e.stopPropagation();
    const newState = !isFavorite;
    setIsFavorite(newState);
    try {
      await AuraSeekApi.toggleFavorite(photo.id);
    } catch {
      setIsFavorite(!newState);
    }
  };

  const hasOverlays =
    (photo.detectedObjects && photo.detectedObjects.length > 0) ||
    (photo.detectedFaces && photo.detectedFaces.length > 0);

  return (
    <button
      type="button"
      onClick={selectionMode ? handleSelect : onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="group relative block aspect-[4/3] overflow-hidden bg-background"
    >
      <div className={cn(
        "w-full h-full transition-all duration-200 ease-out relative",
        isSelected && selectionMode ? "p-3" : "p-0"
      )}>
        <img
          ref={imgRef}
          src={photo.url}
          alt="Photo"
          className={cn(
            "h-full w-full select-none object-cover transition-transform duration-500 ease-out",
            !(isSelected && selectionMode) && "group-hover:scale-[1.03]",
            isSelected && selectionMode && "rounded-lg"
          )}
          draggable={false}
        />

        {/* Bbox overlays on hover */}
        {showBbox && hovered && hasOverlays && (
          <BboxOverlay
            photo={photo}
            imgRef={imgRef}
          />
        )}

        {isSelected && selectionMode && (
          <div className="pointer-events-none absolute inset-0 bg-black/20 opacity-100 rounded-lg m-3" />
        )}
      </div>

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
        <Heart
          className={cn(
            "w-5 h-5 transition-colors drop-shadow-md",
            isFavorite ? "fill-red-500 text-red-500" : "fill-black/30 text-white/90 hover:fill-red-500 hover:text-red-500"
          )}
        />
      </div>
    </button>
  );
}

function BboxOverlay({
  photo,
  imgRef,
}: {
  photo: Photo;
  imgRef: React.RefObject<HTMLImageElement | null>;
}) {
  const img = imgRef.current;
  if (!img) return null;

  const imgW = photo.width || img.naturalWidth || 1;
  const imgH = photo.height || img.naturalHeight || 1;

  const displayW = img.clientWidth;
  const displayH = img.clientHeight;

  const imgAspect = imgW / imgH;
  const displayAspect = displayW / displayH;

  let scaleX: number, scaleY: number, offsetX: number, offsetY: number;

  if (imgAspect > displayAspect) {
    const fitH = displayW / imgAspect;
    scaleX = displayW / imgW;
    scaleY = fitH / imgH;
    offsetX = 0;
    offsetY = (displayH - fitH) / 2;
  } else {
    const fitW = displayH * imgAspect;
    scaleX = fitW / imgW;
    scaleY = displayH / imgH;
    offsetX = (displayW - fitW) / 2;
    offsetY = 0;
  }

  // object-cover crops to fill the container
  const coverScale = Math.max(displayW / imgW, displayH / imgH);
  const renderedW = imgW * coverScale;
  const renderedH = imgH * coverScale;
  const cropX = (renderedW - displayW) / 2;
  const cropY = (renderedH - displayH) / 2;

  scaleX = coverScale;
  scaleY = coverScale;
  offsetX = -cropX;
  offsetY = -cropY;

  const boxes: React.ReactNode[] = [];

  photo.detectedObjects?.forEach((obj, i) => {
    const left = obj.bbox.x * scaleX + offsetX;
    const top = obj.bbox.y * scaleY + offsetY;
    const w = obj.bbox.w * scaleX;
    const h = obj.bbox.h * scaleY;
    boxes.push(
      <div
        key={`obj-${i}`}
        className="absolute pointer-events-none"
        style={{ left, top, width: w, height: h, border: "2px solid #22d3ee", borderRadius: 4 }}
      >
        <span className="absolute -top-5 left-0 text-[10px] bg-cyan-500/80 text-white px-1 rounded whitespace-nowrap">
          {obj.class_name} {(obj.conf * 100).toFixed(0)}%
        </span>
      </div>
    );
  });

  photo.detectedFaces?.forEach((face, i) => {
    const left = face.bbox.x * scaleX + offsetX;
    const top = face.bbox.y * scaleY + offsetY;
    const w = face.bbox.w * scaleX;
    const h = face.bbox.h * scaleY;
    boxes.push(
      <div
        key={`face-${i}`}
        className="absolute pointer-events-none"
        style={{ left, top, width: w, height: h, border: "2px solid #a78bfa", borderRadius: 4 }}
      >
        <span className="absolute -top-5 left-0 text-[10px] bg-violet-500/80 text-white px-1 rounded whitespace-nowrap">
          {face.name || "Face"} {(face.conf * 100).toFixed(0)}%
        </span>
      </div>
    );
  });

  return (
    <div className="absolute inset-0 pointer-events-none z-[5]">
      {boxes}
    </div>
  );
}
