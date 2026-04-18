import { fakePhotos } from "@/data/fake/photos";
import { GlassCard } from "@/components/common/GlassCard";
import { PhotoGrid } from "@/components/photos/PhotoGrid";

export function PhotosView() {
  return (
    <div className="flex h-full flex-1 flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto px-4 pb-6 pt-3 sm:px-6 lg:px-8">
        <GlassCard className="bg-slate-900/40 p-3 sm:p-4">
          <PhotoGrid photos={fakePhotos} />
        </GlassCard>
      </div>
    </div>
  );
}

