import type { Photo } from "@/types/photo.type";
import { PhotoCard } from "./PhotoCard";

type PhotoGridProps = {
  photos: Photo[];
  onPhotoClick?: (photo: Photo) => void;
  selectionMode?: boolean;
  showBbox?: boolean;
  overlayShowFaces?: boolean;
  overlayShowLabels?: boolean;
};

export function PhotoGrid({
  photos,
  onPhotoClick,
  selectionMode      = false,
  showBbox           = true,
  overlayShowFaces   = true,
  overlayShowLabels  = true,
}: PhotoGridProps) {
  return (
    <div className="grid gap-[15px] grid-cols-[repeat(auto-fill,minmax(140px,1fr))] sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
      {photos.map((photo, i) => {
        // Figma style irregular grid: pattern of wide/narrow
        // Wide (span 2): column index 1 or 3 in the pattern (approximate)
        const isWide = (i % 5 === 1) || (i % 5 === 3);
        const gridClass = isWide ? "col-span-1 sm:col-span-2 aspect-[345/185]" : "col-span-1 aspect-[152/185]";
        
        return (
          <PhotoCard
            key={photo.id}
            photo={photo}
            className={gridClass}
            onClick={onPhotoClick ? () => onPhotoClick(photo) : undefined}
            selectionMode={selectionMode}
            showBbox={showBbox}
            overlayShowFaces={overlayShowFaces}
            overlayShowLabels={overlayShowLabels}
          />
        );
      })}
    </div>
  );
}
