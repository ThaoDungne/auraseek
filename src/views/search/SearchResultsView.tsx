import { useState } from "react";
import { ArrowLeft, Sparkles, SortAsc } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { SearchResult } from "@/lib/api";
import { localFileUrl } from "@/lib/api";

interface SearchResultsViewProps {
    results: SearchResult[];
    query?: string;
    isLoading?: boolean;
    onBack: () => void;
}

export function SearchResultsView({ results, query, isLoading, onBack }: SearchResultsViewProps) {
    const [selectedId, setSelectedId] = useState<string | null>(null);

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
                                isSelected={selectedId === result.media_id}
                                onClick={() => setSelectedId(result.media_id === selectedId ? null : result.media_id)}
                            />
                        ))}
                    </div>
                )}
            </div>
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
    const [imgError, setImgError] = useState(false);

    return (
        <div
            onClick={onClick}
            className={`relative group cursor-pointer rounded-xl overflow-hidden aspect-square transition-all duration-200 ${isSelected ? "ring-2 ring-primary ring-offset-2 ring-offset-background" : "hover:scale-[1.02]"
                }`}
        >
            {/* Image */}
            {imgError ? (
                <div className="w-full h-full bg-muted flex items-center justify-center">
                    <span className="text-xs text-muted-foreground text-center px-2 truncate">{result.file_path.split(/[/\\]/).pop()}</span>
                </div>
            ) : (
                <img
                    src={localFileUrl(result.file_path)}
                    alt={result.file_path}
                    className="w-full h-full object-cover"
                    onError={() => setImgError(true)}
                />
            )}

            {/* Overlay on hover */}
            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-all duration-200" />

            {/* Video indicator */}
            {result.media_type === "video" && (
                <div className="absolute bottom-2 right-2 bg-black/70 text-white text-[10px] px-1.5 py-0.5 rounded-full">
                    VIDEO
                </div>
            )}


        </div>
    );
}
