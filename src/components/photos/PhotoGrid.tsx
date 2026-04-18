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
    <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 mb-8">
      {photos.map((photo) => (
        <PhotoCard
          key={photo.id}
          photo={photo}
          className="aspect-[4/3] w-full"
          onClick={onPhotoClick ? () => onPhotoClick(photo) : undefined}
          selectionMode={selectionMode}
          showBbox={showBbox}
          overlayShowFaces={overlayShowFaces}
          overlayShowLabels={overlayShowLabels}
        />
      ))}
    </div>
  );
}
