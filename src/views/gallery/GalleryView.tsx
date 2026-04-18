import { useEffect, useRef, useState } from "react";

interface GalleryViewProps {
  images: string[];
}

export function GalleryView({ images }: GalleryViewProps) {
  const [columns, setColumns] = useState(6);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;

    const handleWheel = (event: WheelEvent) => {
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();

        if (event.deltaY > 0) {
          setColumns((previous) => Math.min(previous + 1, 10));
        } else {
          setColumns((previous) => Math.max(previous - 1, 2));
        }
      }
    };

    container.addEventListener("wheel", handleWheel, { passive: false });

    return () => {
      container.removeEventListener("wheel", handleWheel);
    };
  }, []);

  return (
    <div
      ref={scrollRef}
      className="flex-1 overflow-y-auto px-4 pb-6 pt-2 will-change-scroll"
    >
      <div
        className="grid gap-0.5 transition-all duration-300 ease-out"
        style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
      >
        {images.map((source, index) => (
          <div
            key={index}
            className="group relative aspect-square bg-muted overflow-hidden cursor-pointer"
          >
            <img
              src={source}
              alt={`Mock ${index}`}
              className="w-full h-full object-cover select-none"
              draggable={false}
            />
            <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
        ))}
      </div>
      <div className="h-[500px]" />
    </div>
  );
}

