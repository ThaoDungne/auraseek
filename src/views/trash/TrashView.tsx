import { useState, useEffect, useCallback, useMemo } from "react";
import { AuraSeekApi, localFileUrl, type TimelineGroup } from "@/lib/api";
import type { Photo } from "@/types/photo.type";
import { PhotoGrid } from "@/components/photos/PhotoGrid";
import { FullScreenViewer } from "@/components/photo-detail/FullScreenViewer";
import { Button } from "@/components/ui/button";
import { Trash2 } from "lucide-react";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";

export function TrashView() {
  const [timelineGroups, setTimelineGroups] = useState<TimelineGroup[]>([]);
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedPhoto, setSelectedPhoto] = useState<Photo | null>(null);

  const loadTrash = useCallback(async () => {
    setIsLoading(true);
    try {
      const groups = await AuraSeekApi.getTrash();
      setTimelineGroups(groups);

      const allPhotos: Photo[] = groups.flatMap(g =>
        g.items.map(item => ({
          id: item.media_id,
          url: localFileUrl(item.file_path),
          takenAt: item.created_at || new Date().toISOString(),
          createdAt: item.created_at || new Date().toISOString(),
          sizeBytes: 0,
          width: item.width || 0,
          height: item.height || 0,
          objects: item.objects,
          faces: item.faces,
          faceIds: item.face_ids,
          type: item.media_type as "photo" | "video",
          labels: item.objects,
          favorite: item.favorite,
          detectedObjects: item.detected_objects,
          detectedFaces: item.detected_faces,
        }))
      );
      setPhotos(allPhotos);
    } catch (err) {
      console.warn("[AuraSeek] ⚠️ Failed to load trash:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadTrash();
  }, [loadTrash]);

  const [confirmOpen, setConfirmOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const handleEmptyTrash = async () => {
    try {
      setIsDeleting(true);
      await AuraSeekApi.emptyTrash();
      loadTrash();
      window.dispatchEvent(new Event("refresh_photos"));
    } catch (err) {
      console.error("Lỗi khi làm sạch thùng rác");
    } finally {
      setIsDeleting(false);
      setConfirmOpen(false);
    }
  };

  const sections = useMemo(() => {
    return timelineGroups.map(g => ({
      id: `${g.year}-${g.month}`,
      label: g.label,
      photos: photos.filter(p => {
        const item = g.items.find(i => i.media_id === p.id);
        return !!item;
      }),
    })).filter(s => s.photos.length > 0);
  }, [timelineGroups, photos]);

  return (
    <div className="flex h-full flex-1 flex-col overflow-hidden">
      <div className="flex items-center justify-between px-6 py-4 border-b border-white/5 bg-slate-900">
        <div>
          <h1 className="text-xl font-semibold text-white">Thùng rác</h1>
          <p className="text-sm text-muted-foreground mt-1">Các mục đã xoá sẽ nằm ở đây 30 ngày trước khi bị xoá vĩnh viễn.</p>
        </div>
        <Button 
          variant="destructive" 
          onClick={() => setConfirmOpen(true)}
          disabled={photos.length === 0}
          className="flex items-center gap-2"
        >
          <Trash2 className="w-4 h-4" />
          <span>Làm sạch thùng rác</span>
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 pb-6 pt-3 sm:px-6 lg:px-8">
        {isLoading && (
          <div className="space-y-6">
            <div className="space-y-3 animate-pulse">
              <div className="h-6 w-32 bg-muted rounded-full" />
              <div className="grid grid-cols-4 gap-2">
                {[1, 2, 3, 4].map(j => <div key={j} className="aspect-square bg-muted rounded-xl" />)}
              </div>
            </div>
          </div>
        )}

        {!isLoading && sections.length === 0 && (
          <div className="flex flex-col items-center justify-center h-64 gap-4 text-muted-foreground opacity-60">
            <Trash2 className="w-16 h-16 opacity-50" />
            <div className="text-center">
              <p className="font-medium text-lg">Không có mục nào</p>
              <p className="text-sm mt-1">Các ảnh đã xoá sẽ hiển thị tại đây.</p>
            </div>
          </div>
        )}

        <div className="space-y-6 sm:space-y-8 mt-2">
          {sections.map((section) => (
            <section key={section.id} className="space-y-2 sm:space-y-3">
              <div className="sticky top-3 z-10 mb-1 inline-flex rounded-full border border-white/10 bg-slate-900/70 px-3 py-1 text-xs font-medium text-slate-100 shadow-lg backdrop-blur dark:bg-slate-900/70">
                {section.label}
              </div>
              <PhotoGrid
                photos={section.photos}
                onPhotoClick={(photo) => setSelectedPhoto(photo)}
              />
            </section>
          ))}
        </div>
      </div>

      {selectedPhoto && (
        <FullScreenViewer 
            photo={selectedPhoto} 
            onClose={() => {
                setSelectedPhoto(null);
                loadTrash(); // Reload in case it was restored or deleted inside viewer
            }} 
            isTrashMode={true}
        />
      )}

      <ConfirmDialog
        isOpen={confirmOpen}
        title="Làm sạch thùng rác"
        description="Bạn có chắc muốn làm sạch thùng rác? Hành động này sẽ xoá vĩnh viễn tất cả các tệp khỏi ổ đĩa và không thể hoàn tác."
        confirmText="Xóa tất cả"
        isDestructive
        isLoading={isDeleting}
        onConfirm={handleEmptyTrash}
        onCancel={() => setConfirmOpen(false)}
      />
    </div>
  );
}
