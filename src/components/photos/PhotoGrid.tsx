import type { Photo } from "@/types/photo.type";
import { PhotoCard } from "./PhotoCard";

type PhotoGridProps = {
  photos: Photo[];
  onPhotoClick?: (photo: Photo) => void;
  selectionMode?: boolean;
  showBbox?: boolean;
};

export function PhotoGrid({ photos, onPhotoClick, selectionMode = false, showBbox = true }: PhotoGridProps) {
  return (
    <div className="grid gap-1 sm:gap-1.5 md:gap-2 lg:gap-2.5 xl:gap-3 [grid-template-columns:repeat(auto-fill,minmax(160px,1fr))]">
      {photos.map((photo) => (
        <PhotoCard
          key={photo.id}
          photo={photo}
          onClick={onPhotoClick ? () => onPhotoClick(photo) : undefined}
          selectionMode={selectionMode}
          showBbox={showBbox}
        />
      ))}
    </div>
  );
}
