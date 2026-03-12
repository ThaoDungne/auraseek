import { useState, useRef, useEffect } from "react";
import { ArrowLeft, Sparkles, SortAsc, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { SearchResult } from "@/lib/api";
import { localFileUrl, streamFileUrlSync } from "@/lib/api";
import { SegmentOverlay } from "@/components/photos/SegmentOverlay";
import { FullScreenViewer } from "@/components/photo-detail/FullScreenViewer";
import type { Photo } from "@/types/photo.type";

interface SearchResultsViewProps {
    results: SearchResult[];
    query?: string;
    isLoading?: boolean;
    onBack: () => void;
}

export function SearchResultsView({ results, query, isLoading, onBack }: SearchResultsViewProps) {
    const [selectedPhoto, setSelectedPhoto] = useState<Photo | null>(null);

    if (isLoading) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center gap-4 text-muted-foreground">
                <div className="relative w-16 h-16">
                    <div className="absolute inset-0 rounded-full border-2 border-primary/20 animate-ping" />
                    <div className="absolute inset-2 rounded-full border-2 border-primary/40 animate-ping [animation-delay:150ms]" />
                    <Sparkles className="absolute inset-0 m-auto w-7 h-7 text-primary animate-pulse" />
                </div>
                <p className="font-medium">Đang tìm kiếm bằng AI...</p>
                <p className="text-sm opacity-60">Đang so sánh embedding vectors</p>
            </div>
        );
    }

    return (
        <div className="flex flex-col h-full w-full">
            {/* Header */}
            <div className="h-14 flex items-center px-4 shrink-0 bg-background/95 backdrop-blur z-20 border-b border-border/10 sticky top-0">
                <Button
                    variant="ghost"
                    size="icon"
                    onClick={onBack}
                    className="rounded-full mr-3 text-muted-foreground hover:text-foreground"
                >
                    <ArrowLeft className="w-5 h-5" />
                </Button>
                <div className="flex flex-col">
                    <span className="font-medium tracking-tight">
                        {query ? `Kết quả: "${query}"` : "Kết quả tìm kiếm"}
                    </span>
                    <span className="text-xs text-muted-foreground">
                        {results.length} kết quả · sắp xếp theo độ tương đồng
                    </span>
                </div>
                <div className="flex-1" />
                <div className="flex items-center gap-1 text-xs text-muted-foreground mr-2">
                    <SortAsc className="w-3.5 h-3.5" />
                    <span>Similarity ↓</span>
                </div>
            </div>

            {/* Results Grid */}
            <div className="flex-1 overflow-y-auto px-4 py-4">
                {results.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full gap-4 text-muted-foreground">
                        <Sparkles className="w-12 h-12 opacity-20" />
                        <p className="text-lg font-medium">Không tìm thấy kết quả</p>
                        <p className="text-sm opacity-60">Thử thay đổi từ khóa hoặc sử dụng ảnh tìm kiếm</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-2">
                        {results.map((result) => (
                            <SearchResultCard
                                key={result.media_id}
                                result={result}
                                isSelected={selectedPhoto?.id === result.media_id}
                                onClick={() => {
                                    const isVideo = result.media_type === "video";
                                    let thumbnailUrl = undefined;
                                    if (isVideo && result.thumbnail_path) {
                                        if (result.thumbnail_path.startsWith("/") || result.thumbnail_path.match(/^[A-Za-z]:\\/)) {
                                            thumbnailUrl = streamFileUrlSync(result.thumbnail_path);
                                        } else {
                                            thumbnailUrl = localFileUrl(result.thumbnail_path);
                                        }
                                    }

                                    setSelectedPhoto({
                                        id: result.media_id,
                                        url: localFileUrl(result.file_path),
                                        takenAt: result.metadata.created_at || new Date().toISOString(),
                                        createdAt: result.metadata.created_at || new Date().toISOString(),
                                        sizeBytes: 0,
                                        width: result.width || 0,
                                        height: result.height || 0,
                                        objects: result.metadata.objects || [],
                                        faces: result.metadata.faces || [],
                                        type: isVideo ? "video" : "photo",
                                        labels: result.metadata.objects || [],
                                        favorite: false, // Default since it's not currently tracked in SearchResult
                                        detectedObjects: result.detected_objects || [],
                                        detectedFaces: result.detected_faces || [],
                                        thumbnailUrl,
                                        filePath: result.file_path,
                                    });
                                }}
                            />
                        ))}
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

function SearchResultCard({
    result,
    isSelected,
    onClick,
}: {
    result: SearchResult;
    isSelected: boolean;
    onClick: () => void;
}) {
    const isVideo = result.media_type === "video";
    const [imgError, setImgError] = useState(false);
    const [hovered, setHovered] = useState(false);
    const imgRef = useRef<HTMLImageElement>(null);
    const [displayW, setDisplayW] = useState(0);
    const [displayH, setDisplayH] = useState(0);
    const [activeObjectIndex, setActiveObjectIndex] = useState<number | null>(null);
    const hoverTimerRef = useRef<number | null>(null);

    useEffect(() => {
        const el = imgRef.current;
        if (!el) return;
        const ro = new ResizeObserver(() => {
            setDisplayW(el.clientWidth);
            setDisplayH(el.clientHeight);
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    const hasOverlays =
        !isVideo && (
            (result.detected_objects && result.detected_objects.length > 0) ||
            (result.detected_faces && result.detected_faces.length > 0)
        );

    const imgNaturalW = result.width || imgRef.current?.naturalWidth || 0;
    const imgNaturalH = result.height || imgRef.current?.naturalHeight || 0;
    // For videos use the provided thumbnail_path
    let src = localFileUrl(result.file_path);
    if (isVideo && result.thumbnail_path) {
        if (result.thumbnail_path.startsWith("/") || result.thumbnail_path.match(/^[A-Za-z]:\\/)) {
            src = streamFileUrlSync(result.thumbnail_path);
        } else {
            src = localFileUrl(result.thumbnail_path);
        }
    }

    // Xác định object đang được hover dựa trên toạ độ chuột và bbox
    const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
        if (!result.detected_objects || result.detected_objects.length === 0) return;
        if (!imgRef.current || displayW === 0 || displayH === 0 || imgNaturalW === 0 || imgNaturalH === 0) return;

        const rect = e.currentTarget.getBoundingClientRect();
        const px = e.clientX - rect.left;
        const py = e.clientY - rect.top;

        // mapping giống SegmentOverlay với objectFit="cover"
        const s = Math.max(displayW / imgNaturalW, displayH / imgNaturalH);
        const cropX = (imgNaturalW * s - displayW) / 2;
        const cropY = (imgNaturalH * s - displayH) / 2;

        let hoveredIndex: number | null = null;
        result.detected_objects.forEach((obj, idx) => {
            const x = obj.bbox.x * s - cropX;
            const y = obj.bbox.y * s - cropY;
            const w = obj.bbox.w * s;
            const h = obj.bbox.h * s;
            const pad = 2;
            if (px >= x - pad && px <= x + w + pad && py >= y - pad && py <= y + h + pad) {
                hoveredIndex = idx;
            }
        });

        if (hoveredIndex === null) {
            if (hoverTimerRef.current != null) {
                window.clearTimeout(hoverTimerRef.current);
                hoverTimerRef.current = null;
            }
            if (activeObjectIndex !== null) {
                setActiveObjectIndex(null);
            }
            return;
        }

        if (hoverTimerRef.current != null) {
            window.clearTimeout(hoverTimerRef.current);
        }
        hoverTimerRef.current = window.setTimeout(() => {
            setActiveObjectIndex(hoveredIndex);
        }, 500);
    };

    const handleMouseLeave = () => {
        setHovered(false);
        if (hoverTimerRef.current != null) {
            window.clearTimeout(hoverTimerRef.current);
            hoverTimerRef.current = null;
        }
        setActiveObjectIndex(null);
    };

    return (
        <div
            onClick={onClick}
            onMouseEnter={() => setHovered(true)}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            className={`relative group cursor-pointer rounded-xl overflow-hidden aspect-square transition-all duration-200 ${isSelected ? "ring-2 ring-primary ring-offset-2 ring-offset-background" : "hover:scale-[1.02]"
                }`}
        >
            {imgError ? (
                <div className="w-full h-full bg-muted flex items-center justify-center">
                    <span className="text-xs text-muted-foreground text-center px-2 truncate">
                        {result.file_path.split(/[/\\]/).pop()}
                    </span>
                </div>
            ) : (
                <img
                    ref={imgRef}
                    src={src}
                    alt={result.file_path}
                    className="w-full h-full object-cover"
                    onError={() => setImgError(true)}
                />
            )}


            {/* Segmentation overlay: chỉ vẽ object đang hover, không vẽ face/label */}
            {hovered && hasOverlays && displayW > 0 && imgNaturalW > 0 && (
                <SegmentOverlay
                    detectedObjects={result.detected_objects}
                    detectedFaces={result.detected_faces}
                    imgNaturalW={imgNaturalW}
                    imgNaturalH={imgNaturalH}
                    displayW={displayW}
                    displayH={displayH}
                    objectFit="cover"
                    showFaces={false}
                    showLabels={false}
                    showBoxes={false}
                    activeObjectIndex={activeObjectIndex}
                    onlyActive={true}
                />
            )}

            {/* Dim overlay */}
            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-all duration-200 pointer-events-none" />

            {/* Video badge */}
            {isVideo && !hovered && (
                <div className="absolute bottom-2 right-2 flex items-center gap-1 bg-black/60 text-white text-[10px] px-1.5 py-0.5 rounded-full backdrop-blur-sm">
                    <Play className="w-2.5 h-2.5 fill-white" />
                    <span>VIDEO</span>
                </div>
            )}
        </div>
    );
}
