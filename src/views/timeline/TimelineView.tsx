import { useMemo, useState } from "react";
import type { Photo } from "@/types/photo.type";
import type { TimelineGroup } from "@/lib/api";
import { PhotoGrid } from "@/components/photos/PhotoGrid";
import { FullScreenViewer } from "@/components/photo-detail/FullScreenViewer";
import { localFileUrl, streamFileUrlSync } from "@/lib/api";

interface TimelineViewProps {
  timelineGroups?: TimelineGroup[];
  photos?: Photo[];
  searchQuery?: string;
  isLoading?: boolean;
  selectionMode?: boolean;
  mediaType?: "video" | "photo";
}


export function TimelineView({
  timelineGroups = [],
  photos = [],
  searchQuery = "",
  isLoading = false,
  selectionMode = false,
  mediaType,
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
            if (mediaType === "video") {
              return item.media_type === "video";
            }
            if (mediaType === "photo") {
              // Treat anything that is not an explicit video as "photo-like"
              return item.media_type !== "video";
            }

            if (!searchQuery.trim()) return true;
            const q = searchQuery.toLowerCase();
            return (
              item.objects.some(o => o.toLowerCase().includes(q)) ||
              item.faces.some(f => f.toLowerCase().includes(q)) ||
              item.file_path.toLowerCase().includes(q)
            );
          })
          .map(item => {
            const isVideo = item.media_type === "video";
            const url = localFileUrl(item.file_path);
            // Use streamFileUrlSync for absolute paths (cached thumbs in data dir), else fallback cleanly.
            let thumbnailUrl = undefined;
            if (item.thumbnail_path) {
              if (item.thumbnail_path.startsWith("/") || item.thumbnail_path.match(/^[A-Za-z]:\\/)) {
                thumbnailUrl = streamFileUrlSync(item.thumbnail_path);
              } else {
                thumbnailUrl = localFileUrl(item.thumbnail_path);
              }
            }

            return {
              id: item.media_id,
              url,
              takenAt: item.created_at || new Date().toISOString(),
              createdAt: item.created_at || new Date().toISOString(),
              sizeBytes: 0,
              width: item.width || 0,
              height: item.height || 0,
              objects: item.objects,
              faces: item.faces,
              type: isVideo ? "video" as const : "photo" as const,
              labels: item.objects,
              favorite: item.favorite,
              detectedObjects: item.detected_objects,
              detectedFaces: item.detected_faces,
              thumbnailUrl,
              filePath: item.file_path,
            } as Photo;
          }),
      })).filter(s => s.photos.length > 0);
    }

    // Fallback: group flat photos by month
    const map = new Map<string, { id: string; label: string; photos: Photo[] }>();
    const filteredPhotos = photos.filter(p => {
      if (mediaType === "video") {
        return p.type === "video";
      }
      if (mediaType === "photo") {
        // Default everything that is not marked as video into the photo bucket
        return p.type !== "video";
      }
      if (!searchQuery.trim()) return true;

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
  }, [timelineGroups, photos, searchQuery, mediaType]);

  return (
    <div className="flex relative h-full flex-1 flex-col overflow-hidden">
      <div
        id="timeline-scroll-container"
        className="flex-1 overflow-y-auto px-4 pb-6 pt-3 sm:px-6 lg:px-8 relative"
      >
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
            <div className="text-5xl">{mediaType === "video" ? "🎬" : "📷"}</div>
            <div className="text-center">
              <p className="font-medium text-lg">
                {mediaType === "video" ? "Chưa có video nào" : "Chưa có ảnh nào"}
              </p>
              <p className="text-sm mt-1">
                Vào Cài đặt → Khởi chạy bộ quét AI để import {mediaType === "video" ? "video" : "ảnh"}
              </p>
            </div>
          </div>
        )}

        <div className="space-y-6 sm:space-y-8 pr-6">
          {sections.map((section) => (
            <section key={section.id} id={`section-${section.id}`} className="space-y-3 pt-2">
              <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-black/60 px-4 py-1.5 text-[13px] font-bold text-slate-100 shadow-2xl backdrop-blur-md">
                {section.label}
                <span className="text-[11px] font-medium text-slate-400">· {section.photos.length} mục</span>
              </div>

              <PhotoGrid
                photos={section.photos}
                onPhotoClick={(photo) => setSelectedPhoto(photo)}
                selectionMode={selectionMode}
                showBbox={false}
              />
            </section>
          ))}
        </div>
      </div>

      {/* Right side timeline scrubber */}
      {!isLoading && sections.length > 0 && (
        <div className="absolute right-0 top-32 bottom-32 w-8 sm:w-16 flex flex-col justify-between items-end pr-2 py-4 z-20 opacity-0 hover:opacity-100 transition-opacity duration-300">
          {(() => {
            const seenYears = new Set();
            return sections.map((sec) => {
              const [year, month] = sec.id.split('-');
              const isFirstOfYear = !seenYears.has(year);
              if (isFirstOfYear) seenYears.add(year);

              return (
                <div
                  key={sec.id}
                  className="relative cursor-pointer flex items-center justify-end w-full group/item py-0.5"
                  onClick={() => {
                    const el = document.getElementById(`section-${sec.id}`);
                    const container = document.getElementById('timeline-scroll-container');
                    if (el && container) {
                      const topPos = el.offsetTop - container.offsetTop;
                      container.scrollTo({ top: topPos, behavior: 'smooth' });
                    }
                  }}
                >
                  <div className={`transition-all w-full flex justify-end items-center`}>
                    <div className={`text-[9px] sm:text-[10px] font-bold text-muted-foreground/70 group-hover/item:hidden ${isFirstOfYear ? 'block' : 'hidden'}`}>
                      {year}
                    </div>
                    <div className={`h-1 w-1 sm:h-1.5 sm:w-1.5 rounded-full bg-muted-foreground/30 group-hover/item:hidden ${isFirstOfYear ? 'hidden' : 'block mr-1'}`}></div>

                    <div className="hidden group-hover/item:block text-[10px] sm:text-[11px] font-bold text-primary whitespace-nowrap">
                      thg {month} {year}
                    </div>
                  </div>
                </div>
              );
            });
          })()}
        </div>
      )}

      {selectedPhoto && (
        <FullScreenViewer photo={selectedPhoto} onClose={() => setSelectedPhoto(null)} />
      )}
    </div>
  );
}
