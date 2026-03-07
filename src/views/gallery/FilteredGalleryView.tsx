import { useState, useMemo } from "react";
import type { Photo } from "@/types/photo.type";
import { PhotoGrid } from "@/components/photos/PhotoGrid";
import { FullScreenViewer } from "@/components/photo-detail/FullScreenViewer";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";

type FilteredGalleryProps = {
    title: string;
    subtitle?: string;
    filterType?: "album" | "person" | "favorites" | "recent";
    filterPayload?: string;
    photos?: Photo[];
    onBack: () => void;
};

export function FilteredGalleryView({ title, subtitle, filterType, filterPayload, photos = [], onBack }: FilteredGalleryProps) {
    const [selectedPhoto, setSelectedPhoto] = useState<Photo | null>(null);

    const filteredPhotos = useMemo(() => {
        return photos.filter(photo => {
            switch (filterType) {
                case "favorites": return photo.favorite;
                case "recent":
                    // Last 7 days
                    const sevenDaysAgo = new Date();
                    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
                    return new Date(photo.takenAt) >= sevenDaysAgo;
                case "person": return photo.faceIds?.includes(filterPayload || "") || photo.faces?.includes(filterPayload || "");
                case "album":
                    if (filterPayload?.startsWith("tag_")) {
                        const tag = filterPayload.replace("tag_", "");
                        return photo.labels?.map(l => l.toLowerCase()).includes(tag);
                    }
                    // For dummy standard collections (fav, cam, scr, vid) handled separately in some apps, 
                    // but if it arrives here, pass through or fake it:
                    if (filterPayload === "vid") return photo.type === "video";
                    return true;
                default:
                    return true;
            }
        });
    }, [filterType, filterPayload, photos]);

    return (
        <div className="flex flex-col h-full w-full">

            {/* Context Header */}
            <div className="h-16 flex items-center px-4 shrink-0 bg-background/95 backdrop-blur z-20 border-b border-border/10 sticky top-0">
                <Button variant="ghost" size="icon" onClick={onBack} className="rounded-full mr-3 text-muted-foreground hover:text-foreground">
                    <ArrowLeft className="w-5 h-5" />
                </Button>
                <div className="flex flex-col">
                    <span className="font-medium text-lg tracking-tight">{title}</span>
                    {subtitle && <span className="text-xs text-muted-foreground">{subtitle}</span>}
                </div>
            </div>

            {/* Gallery Content */}
            <div className="flex-1 overflow-y-auto px-4 py-6 will-change-scroll relative">

                {filteredPhotos.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-muted-foreground opacity-70">
                        <div className="text-lg">Không có nội dung nào</div>
                    </div>
                ) : (
                    <div className="mb-8">
                        <div className="text-sm font-medium mb-4 text-muted-foreground">{filteredPhotos.length} ảnh và video</div>
                        <PhotoGrid photos={filteredPhotos} onPhotoClick={setSelectedPhoto} showBbox={filterType !== "person"} />
                    </div>
                )}
            </div>

            {selectedPhoto && (
                <FullScreenViewer
                    photo={selectedPhoto}
                    onClose={() => setSelectedPhoto(null)}
                />
            )}
        </div>
    );
}
