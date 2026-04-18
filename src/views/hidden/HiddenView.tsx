import { useState, useEffect, useCallback, useMemo } from "react";
import { AuraSeekApi, localFileUrl, type TimelineGroup } from "@/lib/api";
import type { Photo } from "@/types/photo.type";
import { PhotoGrid } from "@/components/photos/PhotoGrid";
import { FullScreenViewer } from "@/components/photo-detail/FullScreenViewer";
import { Button } from "@/components/ui/button";
import { Lock, EyeOff, ShieldCheck } from "lucide-react";

export function HiddenView() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [timelineGroups, setTimelineGroups] = useState<TimelineGroup[]>([]);
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [selectedPhoto, setSelectedPhoto] = useState<Photo | null>(null);

  const loadHiddenPhotos = useCallback(async () => {
    setIsLoading(true);
    try {
      const groups = await AuraSeekApi.getHiddenPhotos();
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
      console.warn("[AuraSeek] ⚠️ Failed to load hidden photos:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleAuthenticate = async () => {
    setIsAuthenticating(true);
    try {
      const success = await AuraSeekApi.authenticateOs();
      if (success) {
        setIsAuthenticated(true);
      } else {
        console.warn("Xác thực thất bại.");
      }
    } catch (err) {
      console.error("[AuraSeek] ❌ Auth error:", err);
    } finally {
      setIsAuthenticating(false);
    }
  };

  useEffect(() => {
    if (isAuthenticated) {
      loadHiddenPhotos();
    }
  }, [isAuthenticated, loadHiddenPhotos]);

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

  if (!isAuthenticated) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-6 bg-slate-950/20">
        <div className="p-6 rounded-full bg-primary/10 text-primary animate-pulse">
            <Lock className="w-16 h-16" />
        </div>
        <div className="text-center space-y-2">
            <h1 className="text-2xl font-bold text-foreground">Thư mục ẩn</h1>
            <p className="text-muted-foreground text-sm max-w-sm">
                Thư mục này được bảo vệ bởi hệ thống. Bạn cần xác thực với hệ điều hành để tiếp tục.
            </p>
        </div>
        <Button 
            size="lg" 
            className="px-8 font-semibold flex items-center gap-2"
            onClick={handleAuthenticate}
            disabled={isAuthenticating}
        >
            {isAuthenticating ? (
                <ShieldCheck className="w-5 h-5 animate-bounce" />
            ) : (
                <Lock className="w-5 h-5" />
            )}
            {isAuthenticating ? "Đang xác thực..." : "Mở thư mục ẩn"}
        </Button>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-1 flex-col overflow-hidden">
      <div className="flex items-center justify-between px-6 py-4 border-b border-white/5 bg-slate-900">
        <div>
          <h1 className="text-xl font-semibold text-white flex items-center gap-2">
            <EyeOff className="w-5 h-5 text-primary" />
            Thư mục ẩn
          </h1>
          <p className="text-sm text-muted-foreground mt-1">Các ảnh trong này sẽ không hiện ở dòng thời gian chính.</p>
        </div>
        <div className="flex items-center gap-3">
            <span className="text-xs text-green-500 font-medium flex items-center gap-1">
                <ShieldCheck className="w-3 h-3" />
                Đã xác thực
            </span>
            <Button variant="outline" size="sm" onClick={() => setIsAuthenticated(false)}>
                Khóa lại
            </Button>
        </div>
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
            <EyeOff className="w-16 h-16 opacity-50" />
            <div className="text-center">
              <p className="font-medium text-lg">Trống rỗng</p>
              <p className="text-sm mt-1">Chưa có ảnh nào được ẩn.</p>
            </div>
          </div>
        )}

        <div className="space-y-6 sm:space-y-8 mt-2">
          {sections.map((section) => (
            <section key={section.id} className="space-y-2 sm:space-y-3">
              <div className="sticky top-3 z-10 mb-1 inline-flex rounded-full border border-white/10 bg-slate-900/70 px-3 py-1 text-xs font-medium text-slate-100 shadow-lg backdrop-blur">
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
                loadHiddenPhotos();
            }} 
            isHiddenMode={true}
        />
      )}
    </div>
  );
}
