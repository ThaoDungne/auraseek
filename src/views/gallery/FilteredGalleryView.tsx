import { useState, useMemo, useEffect } from "react";
import type { Photo } from "@/types/photo.type";
import { PhotoGrid } from "@/components/photos/PhotoGrid";
import { FullScreenViewer } from "@/components/photo-detail/FullScreenViewer";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Loader2 } from "lucide-react";
import { AuraSeekApi, localFileUrl } from "@/lib/api";

type FilteredGalleryProps = {
    title: string;
    subtitle?: string;
    filterType?: "album" | "person" | "favorites" | "videos";
    filterPayload?: string;
    photos?: Photo[];
    onBack: () => void;
};

function timelineGroupsToPhotos(groups: any[]): Photo[] {
    const photos: Photo[] = [];
    for (const group of groups) {
        for (const item of group.items || []) {
            const path = item.file_path || item.filePath || "";
            const thumb = item.thumbnail_path || item.thumbnailPath || "";
            photos.push({
                id: item.media_id || item.id,
                url: path ? localFileUrl(path) : "",
                thumbnailUrl: thumb ? localFileUrl(thumb) : (path ? localFileUrl(path) : ""),
                filePath: path,
                type: (item.media_type === "video" ? "video" : "photo") as "video" | "photo",
                favorite: item.favorite || false,
                metadata: item.metadata,
                faceIds: item.detected_faces?.map((f: any) => f.face_id) || [],
                faces: item.detected_faces?.map((f: any) => f.face_id) || [],
                labels: item.detected_objects?.map((o: any) => o.class_name) || [],
            } as any as Photo);
        }
    }
    return photos;
}

export function FilteredGalleryView({ title, subtitle, filterType, filterPayload, photos = [], onBack }: FilteredGalleryProps) {
    const [selectedPhoto, setSelectedPhoto] = useState<Photo | null>(null);
    const [albumPhotos, setAlbumPhotos] = useState<Photo[] | null>(null);
    const [isLoadingAlbum, setIsLoadingAlbum] = useState(false);

    // For manual/AI albums with a real DB id (e.g. custom_album:xxx), fetch from backend
    const isRealAlbum = filterType === "album" &&
        filterPayload &&
        filterPayload.includes(":") &&
        !["fav", "scr"].includes(filterPayload) &&
        !filterPayload.startsWith("tag_");

    useEffect(() => {
        if (!isRealAlbum || !filterPayload) return;
        let cancelled = false;
        setIsLoadingAlbum(true);
        AuraSeekApi.getAlbumPhotos(filterPayload)
            .then(groups => {
                if (!cancelled) {
                    setAlbumPhotos(timelineGroupsToPhotos(groups));
                }
            })
            .catch(err => {
                console.error("[FilteredGallery] Failed to load album photos:", err);
                if (!cancelled) setAlbumPhotos([]);
            })
            .finally(() => {
                if (!cancelled) setIsLoadingAlbum(false);
            });
        return () => { cancelled = true; };
    }, [filterPayload, isRealAlbum]);

    const filteredPhotos = useMemo(() => {
        if (isRealAlbum) return albumPhotos ?? [];

        return photos.filter(photo => {
            switch (filterType) {
                case "favorites":
                    if (!photo.favorite) return false;
                    if (filterPayload === "photos") return photo.type !== "video";
                    if (filterPayload === "videos") return photo.type === "video";
                    return true;
                case "videos":
                    return photo.type === "video";
                case "person":
                    return photo.faceIds?.includes(filterPayload || "") || photo.faces?.includes(filterPayload || "");
                case "album":
                    if (photo.type === "video") return false;
                    if (filterPayload === "scr") {
                        const path = (photo.filePath || "").toLowerCase();
                        const name = path.split(/[/\\]/).pop() || "";
                        return path.includes("screenshot") ||
                               path.includes("screen-capture") ||
                               path.includes("screencast") ||
                               path.includes("ảnh chụp màn hình") ||
                               path.includes("screenshots") ||
                               name.startsWith("scr_") ||
                               name.includes("screen_shot");
                    }
                    if (filterPayload?.startsWith("tag_")) {
                        const tag = filterPayload.replace("tag_", "");
                        return photo.labels?.map(l => l.toLowerCase()).includes(tag);
                    }
                    return true;
                default:
                    return true;
            }
        });
    }, [filterType, filterPayload, photos, albumPhotos, isRealAlbum]);

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
                {isLoadingAlbum ? (
                    <div className="flex flex-col items-center justify-center h-full text-muted-foreground gap-3">
                        <Loader2 className="w-8 h-8 animate-spin opacity-50" />
                        <span className="text-sm">Đang tải album...</span>
                    </div>
                ) : filteredPhotos.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-muted-foreground opacity-70">
                        <div className="text-lg">Album này chưa có ảnh nào</div>
                    </div>
                ) : (
                    <div className="mb-8">
                        <div className="text-sm font-medium mb-4 text-muted-foreground">{filteredPhotos.length} mục</div>
                        <PhotoGrid
                            photos={filteredPhotos}
                            onPhotoClick={setSelectedPhoto}
                            showBbox={false}
                        />
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
