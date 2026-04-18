import {
    Sheet,
    SheetContent,
    SheetDescription,
    SheetHeader,
    SheetTitle,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Filter, Calendar, Users, FileType, Tag, X, Loader2, SearchX } from "lucide-react";
import { useState, useEffect } from "react";
import type { ActiveFilters } from "@/App";
import { AuraSeekApi, type PersonGroup } from "@/lib/api";

const MONTHS = [
    { label: "Tháng 1", value: 1 }, { label: "Tháng 2", value: 2 },
    { label: "Tháng 3", value: 3 }, { label: "Tháng 4", value: 4 },
    { label: "Tháng 5", value: 5 }, { label: "Tháng 6", value: 6 },
    { label: "Tháng 7", value: 7 }, { label: "Tháng 8", value: 8 },
    { label: "Tháng 9", value: 9 }, { label: "Tháng 10", value: 10 },
    { label: "Tháng 11", value: 11 }, { label: "Tháng 12", value: 12 },
];

interface FilterPanelProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    activeFilters?: ActiveFilters;
    onFiltersChange?: (filters: ActiveFilters) => void;
}

export function FilterPanel({ open, onOpenChange, activeFilters, onFiltersChange }: FilterPanelProps) {
    const [localFilters, setLocalFilters] = useState<ActiveFilters>(activeFilters || {});
    const [objectSearch, setObjectSearch] = useState("");

    const [dbObjects, setDbObjects] = useState<string[]>([]);
    const [dbPeople, setDbPeople] = useState<PersonGroup[]>([]);
    const [loadingObjects, setLoadingObjects] = useState(false);
    const [loadingPeople, setLoadingPeople] = useState(false);

    useEffect(() => {
        if (open) {
            setLocalFilters(activeFilters || {});
            loadDbData();
        }
    }, [open, activeFilters]);

    const loadDbData = async () => {
        setLoadingObjects(true);
        setLoadingPeople(true);
        try {
            const objs = await AuraSeekApi.getDistinctObjects();
            setDbObjects(objs);
        } catch {
            setDbObjects([]);
        } finally {
            setLoadingObjects(false);
        }
        try {
            const ppl = await AuraSeekApi.getPeople();
            setDbPeople(ppl);
        } catch {
            setDbPeople([]);
        } finally {
            setLoadingPeople(false);
        }
    };

    const update = (patch: Partial<ActiveFilters>) => {
        setLocalFilters(prev => ({ ...prev, ...patch }));
    };

    const handleApply = () => {
        onFiltersChange?.(localFilters);
        onOpenChange(false);
    };

    const handleReset = () => {
        const empty: ActiveFilters = {};
        setLocalFilters(empty);
        onFiltersChange?.(empty);
        onOpenChange(false);
    };

    const activeCount = Object.values(localFilters).filter(v => v !== undefined).length;

    const filteredObjects = objectSearch
        ? dbObjects.filter(o => o.toLowerCase().includes(objectSearch.toLowerCase()))
        : dbObjects;

    return (
        <Sheet open={open} onOpenChange={onOpenChange}>
            <SheetContent side="right" className="w-full sm:max-w-md p-0 flex flex-col bg-background border-l-border/30 shadow-2xl">
                <SheetHeader className="px-6 py-5 border-b border-border/10 bg-muted/20">
                    <SheetTitle className="flex items-center gap-2 text-base font-bold">
                        <Filter className="w-5 h-5 text-primary" />
                        Bộ lọc nâng cao
                        {activeCount > 0 && (
                            <span className="ml-auto flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider bg-primary/10 text-primary px-2 py-0.5 rounded-full">
                                {activeCount} đang bật
                            </span>
                        )}
                    </SheetTitle>
                    <SheetDescription className="text-[13px] text-muted-foreground/80">
                        Kết hợp nhiều điều kiện để tìm chính xác nội dung bạn cần.
                    </SheetDescription>
                </SheetHeader>

                <div className="flex-1 overflow-y-auto px-6 py-6 space-y-8">

                    {/* Media Type */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 text-sm font-bold text-foreground">
                            <FileType className="w-4 h-4 text-primary/70" />
                            Loại tệp
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {[
                                { label: "Tất cả", value: undefined },
                                { label: "Chỉ Ảnh", value: "image" },
                                { label: "Chỉ Video", value: "video" },
                            ].map(opt => (
                                <Button
                                    key={opt.label}
                                    variant="outline"
                                    className={`rounded-full h-9 px-4 text-[13px] transition-all ${localFilters.mediaType === opt.value
                                            ? "bg-primary/10 border-primary/30 text-primary font-medium"
                                            : "text-muted-foreground hover:text-foreground"
                                        }`}
                                    onClick={() => update({ mediaType: opt.value })}
                                >
                                    {opt.label}
                                </Button>
                            ))}
                        </div>
                    </div>

                    {/* Time Filter */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 text-sm font-bold text-foreground">
                            <Calendar className="w-4 h-4 text-primary/70" />
                            Tháng / Năm
                        </div>
                        <div className="space-y-1.5">
                            <label className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground/70">Năm</label>
                            <input
                                type="number"
                                min="2000"
                                max="2030"
                                value={localFilters.year || ""}
                                onChange={(e) => update({ year: e.target.value ? parseInt(e.target.value) : undefined })}
                                placeholder="e.g. 2024"
                                className="flex h-9 w-32 rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm font-medium"
                            />
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {MONTHS.map(m => (
                                <Button
                                    key={m.value}
                                    variant="outline"
                                    className={`rounded-full text-[12px] h-8 px-3 transition-all ${localFilters.month === m.value
                                            ? "bg-primary/10 border-primary/30 text-primary font-medium"
                                            : "text-muted-foreground hover:text-foreground"
                                        }`}
                                    onClick={() => update({ month: localFilters.month === m.value ? undefined : m.value })}
                                >
                                    {m.label}
                                </Button>
                            ))}
                        </div>
                    </div>

                    {/* Object Filter — loaded from DB */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 text-sm font-bold text-foreground">
                            <Tag className="w-4 h-4 text-primary/70" />
                            Đối tượng (từ dữ liệu đã quét)
                        </div>

                        {localFilters.object && (
                            <div className="flex items-center gap-2">
                                <span className="bg-primary/10 text-primary text-xs px-2.5 py-1 rounded-full border border-primary/20">
                                    {localFilters.object}
                                </span>
                                <button onClick={() => update({ object: undefined })} className="text-muted-foreground hover:text-foreground">
                                    <X className="w-3.5 h-3.5" />
                                </button>
                            </div>
                        )}

                        {loadingObjects ? (
                            <div className="flex items-center gap-2 text-xs text-muted-foreground py-2">
                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                Đang tải danh sách đối tượng...
                            </div>
                        ) : dbObjects.length === 0 ? (
                            <div className="flex items-center gap-2 text-xs text-muted-foreground py-2">
                                <SearchX className="w-3.5 h-3.5" />
                                Chưa có đối tượng nào được phát hiện. Hãy quét ảnh trước.
                            </div>
                        ) : (
                            <>
                                <input
                                    type="text"
                                    placeholder="Tìm đối tượng..."
                                    value={objectSearch}
                                    onChange={e => setObjectSearch(e.target.value)}
                                    className="flex h-8 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm"
                                />

                                <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto">
                                    {filteredObjects.length === 0 ? (
                                        <span className="text-xs text-muted-foreground py-1">Không tìm thấy đối tượng phù hợp</span>
                                    ) : (
                                        filteredObjects.map(obj => (
                                            <button
                                                key={obj}
                                                onClick={() => update({ object: localFilters.object === obj ? undefined : obj })}
                                                className={`text-xs px-2 py-1 rounded-md border transition-all ${localFilters.object === obj
                                                        ? "bg-primary/10 border-primary/30 text-primary font-medium"
                                                        : "border-border/40 hover:bg-muted"
                                                    }`}
                                            >
                                                {obj}
                                            </button>
                                        ))
                                    )}
                                </div>
                            </>
                        )}
                    </div>

                    {/* Person Filter — loaded from DB */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 text-sm font-bold text-foreground">
                            <Users className="w-4 h-4 text-primary/70" />
                            Người trong ảnh (từ dữ liệu đã quét)
                        </div>

                        {localFilters.face && (
                            <div className="flex items-center gap-2">
                                <span className="bg-violet-500/10 text-violet-600 dark:text-violet-400 text-xs px-2.5 py-1 rounded-full border border-violet-500/20">
                                    {localFilters.face}
                                </span>
                                <button onClick={() => update({ face: undefined })} className="text-muted-foreground hover:text-foreground">
                                    <X className="w-3.5 h-3.5" />
                                </button>
                            </div>
                        )}

                        {loadingPeople ? (
                            <div className="flex items-center gap-2 text-xs text-muted-foreground py-2">
                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                Đang tải danh sách người...
                            </div>
                        ) : dbPeople.length === 0 ? (
                            <div className="flex items-center gap-2 text-xs text-muted-foreground py-2">
                                <SearchX className="w-3.5 h-3.5" />
                                Chưa phát hiện người nào. Hãy quét ảnh trước.
                            </div>
                        ) : (
                            <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto">
                                {dbPeople.map((person, idx) => {
                                    const displayName = person.name || `Người ${idx + 1}`;
                                    const isActive = localFilters.face === person.face_id;
                                    return (
                                        <button
                                            key={person.face_id}
                                            onClick={() => update({ face: isActive ? undefined : person.face_id })}
                                            className={`text-xs px-2.5 py-1.5 rounded-lg border transition-all flex items-center gap-1.5 ${isActive
                                                    ? "bg-violet-500/10 border-violet-500/30 text-violet-600 dark:text-violet-400 font-medium"
                                                    : "border-border/40 hover:bg-muted"
                                                }`}
                                        >
                                            {person.thumbnail && (
                                                <img
                                                    src={`asset://localhost/${encodeURIComponent(person.thumbnail).replace(/%2F/g, "/")}`}
                                                    className="w-5 h-5 rounded-full object-cover"
                                                    alt=""
                                                />
                                            )}
                                            <span>{displayName}</span>
                                            <span className="text-[10px] opacity-60">({person.photo_count})</span>
                                        </button>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                </div>

                <div className="p-5 border-t border-border/10 bg-background flex gap-3">
                    <Button variant="outline" className="flex-1 h-11 rounded-full font-bold text-[13px]" onClick={handleReset}>
                        Xóa tất cả
                    </Button>
                    <Button className="flex-2 h-11 rounded-full font-bold text-[13px]" onClick={handleApply}>
                        Áp dụng bộ lọc
                        {activeCount > 0 && ` (${activeCount})`}
                    </Button>
                </div>
            </SheetContent>
        </Sheet>
    );
}
