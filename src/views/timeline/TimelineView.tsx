import { useMemo, useState } from "react";
import type { Photo } from "@/types/photo.type";
import type { TimelineGroup } from "@/lib/api";
import { PhotoGrid } from "@/components/photos/PhotoGrid";
import { FullScreenViewer } from "@/components/photo-detail/FullScreenViewer";
import { localFileUrl } from "@/lib/api";

interface TimelineViewProps {
  timelineGroups?: TimelineGroup[];
  photos?: Photo[];
  searchQuery?: string;
  isLoading?: boolean;
  selectionMode?: boolean;
}


export function TimelineView({
  timelineGroups = [],
  photos = [],
  searchQuery = "",
  isLoading = false,
  selectionMode = false,
}: TimelineViewProps) {
  const [selectedPhoto, setSelectedPhoto] = useState<Photo | null>(null);

  // Process into sections
  const sections = useMemo(() => {
    // Prefer structured timeline groups
    if (timelineGroups.length > 0) {
      return timelineGroups.map(g => ({
        id: `${g.year}-${g.month}`,
        label: g.label,
        photos: g.items
          .filter(item => {
            if (!searchQuery.trim()) return true;
            const q = searchQuery.toLowerCase();
            return (
              item.objects.some(o => o.toLowerCase().includes(q)) ||
              item.faces.some(f => f.toLowerCase().includes(q)) ||
              item.file_path.toLowerCase().includes(q)
            );
          })
          .map(item => ({
            id: item.media_id,
            url: localFileUrl(item.file_path),
            takenAt: item.created_at || new Date().toISOString(),
            createdAt: item.created_at || new Date().toISOString(),
            sizeBytes: 0,
            width: item.width || 0,
            height: item.height || 0,
            objects: item.objects,
            faces: item.faces,
            type: item.media_type === "video" ? "video" as const : "photo" as const,
            labels: item.objects,
          })),
      })).filter(s => s.photos.length > 0);
    }

    // Fallback: group flat photos by month
    const map = new Map<string, { id: string; label: string; photos: Photo[] }>();
    const filteredPhotos = searchQuery.trim() === ""
      ? photos
      : photos.filter(p => {
        const q = searchQuery.toLowerCase();
        return (
          p.labels?.some(l => l.toLowerCase().includes(q)) ||
          p.objects?.some(o => o.toLowerCase().includes(q)) ||
          p.id.toLowerCase().includes(q)
        );
      });

    for (const photo of filteredPhotos) {
      const date = new Date(photo.takenAt);
      const id = `${date.getFullYear()}-${date.getMonth() + 1}`;
      const label = new Intl.DateTimeFormat("vi-VN", { month: "long", year: "numeric" })
        .format(date);
      const existing = map.get(id);
      if (!existing) {
        map.set(id, { id, label, photos: [photo] });
      } else {
        existing.photos.push(photo);
      }
    }
    return Array.from(map.values()).sort((a, b) => a.id < b.id ? 1 : -1);
  }, [timelineGroups, photos, searchQuery]);

  return (
    <div className="flex h-full flex-1 flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto px-4 pb-6 pt-3 sm:px-6 lg:px-8">

        {/* Loading skeleton */}
        {isLoading && (
          <div className="space-y-6">
            {[1, 2, 3].map(i => (
              <div key={i} className="space-y-3 animate-pulse">
                <div className="h-6 w-32 bg-muted rounded-full" />
                <div className="grid grid-cols-4 gap-2">
                  {[1, 2, 3, 4].map(j => (
                    <div key={j} className="aspect-square bg-muted rounded-xl" />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Empty state */}
        {!isLoading && sections.length === 0 && (
          <div className="flex flex-col items-center justify-center h-64 gap-4 text-muted-foreground opacity-60">
            <div className="text-5xl">📷</div>
            <div className="text-center">
              <p className="font-medium text-lg">Chưa có ảnh nào</p>
              <p className="text-sm mt-1">Vào Cài đặt → Khởi chạy bộ quét AI để import ảnh</p>
            </div>
          </div>
        )}

        <div className="space-y-6 sm:space-y-8">
          {sections.map((section) => (
            <section key={section.id} className="space-y-2 sm:space-y-3">
              <div className="sticky top-3 z-10 mb-1 inline-flex rounded-full border border-white/10 bg-slate-900/70 px-3 py-1 text-xs font-medium text-slate-100 shadow-lg backdrop-blur dark:bg-slate-900/70 bg-slate-700/70">
                {section.label}
                <span className="ml-2 opacity-60">· {section.photos.length} ảnh</span>
              </div>

              <PhotoGrid
                photos={section.photos}
                onPhotoClick={(photo) => setSelectedPhoto(photo)}
                selectionMode={selectionMode}
              />
            </section>
          ))}
        </div>
      </div>

      {selectedPhoto && (
        <FullScreenViewer photo={selectedPhoto} onClose={() => setSelectedPhoto(null)} />
      )}
    </div>
  );
}
