import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import {
    CopySlash, Trash2, CheckCircle2, Film, Image as ImageIcon,
    ChevronDown, AlertTriangle, RefreshCw, Loader2
} from "lucide-react";
import { AuraSeekApi, type DuplicateGroup, localFileUrl, streamFileUrlSync } from "@/lib/api";

function formatSize(bytes: number): string {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

interface DuplicatesViewProps {
    mediaType: "image" | "video";
}

export function DuplicatesView({ mediaType }: DuplicatesViewProps) {
    const [groups, setGroups] = useState<DuplicateGroup[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    // per-group: set of media_ids marked to delete
    const [markedForDelete, setMarkedForDelete] = useState<Record<string, Set<string>>>({});
    const [deletingGroup, setDeletingGroup] = useState<string | null>(null);
    const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());

    const fetchDuplicates = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const data = await AuraSeekApi.getDuplicates(mediaType);
            setGroups(data);
            // Auto-mark all non-first items as delete candidates
            const initial: Record<string, Set<string>> = {};
            data.forEach(g => {
                initial[g.group_id] = new Set(g.items.slice(1).map(i => i.media_id));
            });
            setMarkedForDelete(initial);
            // Expand all groups by default
            setExpandedGroups(new Set(data.map(g => g.group_id)));
        } catch (err) {
            setError(String(err));
        } finally {
            setIsLoading(false);
        }
    }, [mediaType]);

    useEffect(() => { fetchDuplicates(); }, [fetchDuplicates]);

    const toggleMark = (groupId: string, mediaId: string) => {
        setMarkedForDelete(prev => {
            const group = new Set(prev[groupId] ?? []);
            const groupItems = groups.find(g => g.group_id === groupId)?.items ?? [];
            const marked = new Set(group);
            if (marked.has(mediaId)) {
                // Prevent unmarking if it would leave nothing to keep
                const keptCount = groupItems.length - (marked.size - 1);
                if (keptCount < 1) return prev;
                marked.delete(mediaId);
            } else {
                // Prevent marking all items — at least 1 must be kept
                const keptCount = groupItems.length - (marked.size + 1);
                if (keptCount < 1) return prev;
                marked.add(mediaId);
            }
            return { ...prev, [groupId]: marked };
        });
    };

    const deleteMarked = async (groupId: string) => {
        const toDelete = [...(markedForDelete[groupId] ?? [])];
        if (toDelete.length === 0) return;
        setDeletingGroup(groupId);
        try {
            for (const mediaId of toDelete) {
                await AuraSeekApi.moveToTrash(mediaId);
            }
            // Refresh
            await fetchDuplicates();
        } catch (err) {
            setError(String(err));
        } finally {
            setDeletingGroup(null);
        }
    };

    const deleteAllMarked = async () => {
        setIsLoading(true);
        try {
            for (const [, set] of Object.entries(markedForDelete)) {
                for (const mediaId of set) {
                    await AuraSeekApi.moveToTrash(mediaId);
                }
            }
            await fetchDuplicates();
        } catch (err) {
            setError(String(err));
        } finally {
            setIsLoading(false);
        }
    };

    const toggleExpand = (groupId: string) => {
        setExpandedGroups(prev => {
            const s = new Set(prev);
            if (s.has(groupId)) s.delete(groupId); else s.add(groupId);
            return s;
        });
    };

    // Stats
    const totalDelete = Object.values(markedForDelete).reduce((a, s) => a + s.size, 0);
    const totalSaveMB = groups
        .flatMap(g => g.items.filter(i => (markedForDelete[g.group_id] ?? new Set()).has(i.media_id)))
        .reduce((a, i) => a + i.size, 0);

    const isVideo = mediaType === "video";

    return (
        <div className="flex-1 overflow-y-auto px-6 py-8 bg-background/50">
            <div className="max-w-5xl mx-auto space-y-8">

                {/* Header */}
                <div className="flex items-start justify-between gap-4 flex-wrap">
                    <div className="flex flex-col gap-2">
                        <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center mb-1">
                            {isVideo
                                ? <Film className="w-6 h-6 text-primary" />
                                : <ImageIcon className="w-6 h-6 text-primary" />}
                        </div>
                        <h1 className="text-2xl font-semibold tracking-tight">
                            {isVideo ? "Video trùng lặp" : "Ảnh trùng lặp"}
                        </h1>
                        <p className="text-muted-foreground text-sm max-w-lg">
                            {isVideo
                                ? "Phát hiện video trùng lặp bằng Hash + AI phân tích nội dung đoạn đầu (ngưỡng ≥ 97%). Gom nhóm để bạn dễ dàng dọn dẹp."
                                : "Phát hiện ảnh trùng lặp bằng Hash chính xác, pHash nhận diện ảnh tương tự, và AI embedding (ngưỡng ≥ 97%)."}
                        </p>
                    </div>
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={fetchDuplicates}
                        disabled={isLoading}
                        className="rounded-full gap-2 shrink-0"
                    >
                        <RefreshCw className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
                        Quét lại
                    </Button>
                </div>

                {/* Summary bar */}
                {!isLoading && !error && groups.length > 0 && (
                    <div className="flex items-center justify-between p-4 bg-muted/40 rounded-2xl border border-border/20 gap-4 flex-wrap">
                        <div className="flex gap-6">
                            <div>
                                <div className="text-xs text-muted-foreground mb-0.5">Nhóm trùng lặp</div>
                                <div className="text-lg font-semibold">{groups.length}</div>
                            </div>
                            <div>
                                <div className="text-xs text-muted-foreground mb-0.5">Đề xuất xóa</div>
                                <div className="text-lg font-semibold text-destructive">{totalDelete} mục</div>
                            </div>
                            <div>
                                <div className="text-xs text-muted-foreground mb-0.5">Có thể tiết kiệm</div>
                                <div className="text-lg font-semibold text-emerald-500">{formatSize(totalSaveMB)}</div>
                            </div>
                        </div>
                        {totalDelete > 0 && (
                            <Button
                                variant="destructive"
                                size="sm"
                                className="rounded-full gap-2"
                                onClick={deleteAllMarked}
                                disabled={isLoading}
                            >
                                <Trash2 className="w-4 h-4" />
                                Xóa tất cả {totalDelete} mục thừa
                            </Button>
                        )}
                    </div>
                )}

                {/* Error */}
                {error && (
                    <div className="flex items-center gap-3 p-4 bg-destructive/10 border border-destructive/20 rounded-xl text-destructive text-sm">
                        <AlertTriangle className="w-5 h-5 shrink-0" />
                        {error}
                    </div>
                )}

                {/* Loading */}
                {isLoading && (
                    <div className="flex flex-col items-center justify-center py-20 gap-4 text-muted-foreground">
                        <Loader2 className="w-10 h-10 animate-spin text-primary/40" />
                        <p className="text-sm">Đang phân tích {isVideo ? "video" : "ảnh"} trùng lặp...</p>
                        <p className="text-xs text-muted-foreground/60">Quá trình này có thể mất vài giây tuỳ số lượng</p>
                    </div>
                )}

                {/* Empty state */}
                {!isLoading && !error && groups.length === 0 && (
                    <div className="flex flex-col items-center justify-center py-24 gap-3 text-center">
                        <div className="text-5xl mb-2">✨</div>
                        <div className="font-semibold text-xl text-foreground">
                            Không tìm thấy {isVideo ? "video" : "ảnh"} trùng lặp
                        </div>
                        <div className="text-sm text-muted-foreground">
                            {isVideo ? "Thư viện video" : "Thư viện ảnh"} của bạn đang rất gọn gàng.
                        </div>
                    </div>
                )}

                {/* Groups */}
                {!isLoading && !error && groups.length > 0 && (
                    <div className="space-y-4">
                        {groups.map((group) => {
                            const deleted = markedForDelete[group.group_id] ?? new Set();
                            const isExpanded = expandedGroups.has(group.group_id);
                            const isDeleting = deletingGroup === group.group_id;
                            const deleteCount = deleted.size;

                            return (
                                <div key={group.group_id}
                                    className="bg-background rounded-2xl border border-border/30 overflow-hidden shadow-sm"
                                >
                                    {/* Group header */}
                                    <div
                                        className="flex items-center justify-between px-5 py-4 bg-muted/10 cursor-pointer select-none"
                                        onClick={() => toggleExpand(group.group_id)}
                                    >
                                        <div className="flex items-center gap-3 min-w-0">
                                            <CopySlash className="w-4 h-4 text-muted-foreground shrink-0" />
                                            <div className="min-w-0">
                                                <div className="text-sm font-medium text-foreground truncate">
                                                    Nhóm trùng lặp ({group.items.length} {isVideo ? "video" : "ảnh"})
                                                </div>
                                                <div className="text-xs text-muted-foreground mt-0.5">
                                                    Đang chọn xóa {deleteCount} mục
                                                </div>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-2 shrink-0">
                                            {deleteCount > 0 && (
                                                <Button
                                                    variant="outline"
                                                    size="sm"
                                                    className="h-8 rounded-full text-xs hover:bg-destructive/10 hover:text-destructive hover:border-destructive/30"
                                                    disabled={isDeleting}
                                                    onClick={(e) => { e.stopPropagation(); deleteMarked(group.group_id); }}
                                                >
                                                    {isDeleting
                                                        ? <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                                                        : <Trash2 className="w-3 h-3 mr-1" />}
                                                    Xóa {deleteCount} mục
                                                </Button>
                                            )}
                                            <ChevronDown className={`w-4 h-4 text-muted-foreground transition-transform ${isExpanded ? "rotate-180" : ""}`} />
                                        </div>
                                    </div>

                                    {/* Items */}
                                    {isExpanded && (
                                        <div className="p-5 flex gap-4 overflow-x-auto snap-x pb-6">
                                            {group.items.map((item) => {
                                                const isMarked = deleted.has(item.media_id);
                                                const isKept = !isMarked;

                                                return (
                                                    <div
                                                        key={item.media_id}
                                                        className="snap-start shrink-0 flex flex-col gap-2 cursor-pointer"
                                                        style={{ width: isVideo ? 240 : 176 }}
                                                        onClick={() => toggleMark(group.group_id, item.media_id)}
                                                    >
                                                        {/* Thumbnail */}
                                                        <div className={`relative rounded-xl overflow-hidden transition-all
                                                            ${isKept
                                                                ? "ring-2 ring-emerald-500 ring-offset-2 ring-offset-background"
                                                                : "ring-2 ring-destructive ring-offset-2 ring-offset-background opacity-70 hover:opacity-100"
                                                            }`}
                                                            style={{ aspectRatio: isVideo ? "16/9" : "4/3" }}
                                                        >
                                                            {(() => {
                                                                let src: string | undefined;
                                                                if (isVideo) {
                                                                    if (item.thumbnail_path) {
                                                                        if (item.thumbnail_path.startsWith("/") || item.thumbnail_path.match(/^[A-Za-z]:\\/)) {
                                                                            src = streamFileUrlSync(item.thumbnail_path);
                                                                        } else {
                                                                            src = localFileUrl(item.thumbnail_path);
                                                                        }
                                                                    }
                                                                } else {
                                                                    src = localFileUrl(item.file_path);
                                                                }

                                                                if (src) {
                                                                    return (
                                                                        <img
                                                                            src={src}
                                                                            className="w-full h-full object-cover"
                                                                            loading="lazy"
                                                                        />
                                                                    );
                                                                }

                                                                // Fallback placeholder for videos without thumbnail
                                                                return (
                                                                    <div className="w-full h-full flex items-center justify-center bg-muted/40 text-muted-foreground">
                                                                        <Film className="w-8 h-8" />
                                                                    </div>
                                                                );
                                                            })()}

                                                            {/* Badge */}
                                                            <div className="absolute top-2 left-2 z-10">
                                                                {isKept ? (
                                                                    <span className="bg-emerald-500/90 text-white backdrop-blur px-2 py-0.5 rounded-full text-[10px] font-semibold flex items-center gap-1 shadow">
                                                                        <CheckCircle2 className="w-3 h-3" />
                                                                        GIỮ LẠI
                                                                    </span>
                                                                ) : (
                                                                    <span className="bg-destructive/90 text-white backdrop-blur px-2 py-0.5 rounded-full text-[10px] font-semibold flex items-center gap-1 shadow">
                                                                        <Trash2 className="w-3 h-3" />
                                                                        XÓA
                                                                    </span>
                                                                )}
                                                            </div>

                                                            {/* Size badge */}
                                                            <div className="absolute bottom-2 right-2 bg-black/60 text-white backdrop-blur px-1.5 py-0.5 rounded text-[10px]">
                                                                {formatSize(item.size)}
                                                            </div>
                                                        </div>

                                                        {/* Filename */}
                                                        <div className="text-xs text-muted-foreground truncate px-0.5">
                                                            {item.file_path.split("/").pop()}
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                )}

                <div className="h-20" />
            </div>
        </div>
    );
}
