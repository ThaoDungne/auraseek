import { useState, useEffect, useCallback } from "react";
import type { PersonGroup } from "@/lib/api";
import { localFileUrl, streamFileUrl, AuraSeekApi } from "@/lib/api";
import { Pencil, Check, X, User, Trash2, CheckCircle2, Circle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";

interface PeopleViewProps {
    people?: PersonGroup[];
    onNavigate?: (payload: any) => void;
}

const AVATAR_PX = 112;

function FaceCropAvatar({
    imageUrl,
    bbox,
    alt,
}: {
    imageUrl: string;
    bbox: { x: number; y: number; w: number; h: number } | null;
    alt: string;
}) {
    const [bgStyle, setBgStyle] = useState<React.CSSProperties | null>(null);
    const [loaded, setLoaded] = useState(false);
    const [failed, setFailed] = useState(false);

    const computeCrop = useCallback((naturalW: number, naturalH: number) => {
        if (!bbox || !naturalW || !naturalH) {
            setBgStyle({
                backgroundImage: `url("${imageUrl}")`,
                backgroundSize: "cover",
                backgroundPosition: "center",
            });
            setLoaded(true);
            return;
        }

        const faceCx = bbox.x + bbox.w / 2;
        const faceCy = bbox.y + bbox.h / 2;
        const cropSize = Math.max(bbox.w, bbox.h) * 2;

        let cropX = faceCx - cropSize / 2;
        let cropY = faceCy - cropSize / 2;
        const clampedSize = Math.min(cropSize, naturalW, naturalH);
        cropX = Math.max(0, Math.min(cropX, naturalW - clampedSize));
        cropY = Math.max(0, Math.min(cropY, naturalH - clampedSize));

        const scale = AVATAR_PX / clampedSize;
        const bgW = naturalW * scale;
        const bgH = naturalH * scale;

        setBgStyle({
            backgroundImage: `url("${imageUrl}")`,
            backgroundSize: `${bgW}px ${bgH}px`,
            backgroundPosition: `${-cropX * scale}px ${-cropY * scale}px`,
            backgroundRepeat: "no-repeat",
        });
        setLoaded(true);
    }, [bbox, imageUrl]);

    useEffect(() => {
        setFailed(false);
        setLoaded(false);
        setBgStyle(null);

        const img = new Image();
        img.onload = () => computeCrop(img.naturalWidth, img.naturalHeight);
        img.onerror = () => setFailed(true);
        img.src = imageUrl;

        return () => { img.onload = null; img.onerror = null; };
    }, [imageUrl, computeCrop]);

    if (failed) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-muted text-muted-foreground">
                <User className="w-10 h-10 opacity-40" />
            </div>
        );
    }

    if (!loaded || !bgStyle) {
        return <div className="w-full h-full bg-muted animate-pulse rounded-full" />;
    }

    return (
        <div
            className="w-full h-full rounded-full transition-transform duration-500 ease-out group-hover:scale-105"
            style={bgStyle}
            role="img"
            aria-label={alt}
        />
    );
}

function PersonCard({
    person,
    index,
    onNavigate,
    onRename,
    isDeleteMode,
    isSelected,
    onToggleSelect,
}: {
    person: PersonGroup;
    index: number;
    onNavigate?: (payload: any) => void;
    onRename?: (faceId: string, name: string) => void;
    isDeleteMode: boolean;
    isSelected: boolean;
    onToggleSelect: (id: string) => void;
}) {
    const [editing, setEditing] = useState(false);
    const [name, setName] = useState(person.name || "");

    const handleSave = async () => {
        if (!name.trim()) return;
        try {
            await AuraSeekApi.namePerson(person.face_id, name.trim());
            onRename?.(person.face_id, name.trim());
        } catch (e) {
            console.error("Failed to name person:", e);
        }
        setEditing(false);
    };

    const displayName = person.name || `Người ${index + 1}`;
    const rawPath = person.thumbnail || person.cover_path;
    const [imageUrl, setImageUrl] = useState<string | null>(null);

    useEffect(() => {
        if (!rawPath) { setImageUrl(null); return; }
        if (rawPath.startsWith("/") || rawPath.match(/^[A-Za-z]:\\/)) {
            streamFileUrl(rawPath).then(url => setImageUrl(url));
        } else {
            setImageUrl(localFileUrl(rawPath));
        }
    }, [rawPath]);

    const handleClick = () => {
        if (isDeleteMode) {
            onToggleSelect(person.face_id);
        } else {
            onNavigate?.({ id: person.face_id, title: displayName });
        }
    };

    return (
        <div className={`group flex flex-col items-center gap-3 transition-all duration-300 ${isDeleteMode ? "scale-95 hover:scale-100" : ""}`}>
            <div
                className={`relative rounded-full overflow-hidden bg-muted transition-all duration-300 ring-4 cursor-pointer shadow-sm hover:shadow-md ${
                    isSelected 
                        ? "ring-red-500/50" 
                        : isDeleteMode 
                            ? "ring-transparent group-hover:ring-red-500/20" 
                            : "ring-transparent group-hover:ring-primary"
                }`}
                style={{ width: AVATAR_PX, height: AVATAR_PX }}
                onClick={handleClick}
            >
                {imageUrl ? (
                    <FaceCropAvatar
                        imageUrl={imageUrl}
                        bbox={person.face_bbox}
                        alt={displayName}
                    />
                ) : (
                    <div className="w-full h-full flex items-center justify-center bg-muted text-muted-foreground">
                        <User className="w-10 h-10 opacity-40" />
                    </div>
                )}

                {/* Selection Overlay for Delete Mode */}
                {isDeleteMode && (
                    <div className={`absolute inset-0 flex items-center justify-center transition-colors ${isSelected ? "bg-red-500/20" : "bg-black/0 group-hover:bg-black/10"}`}>
                        <div className={`rounded-full p-1.5 transition-all transform ${isSelected ? "bg-red-500 text-white scale-110" : "bg-black/20 text-white/80 opacity-0 group-hover:opacity-100 animate-in fade-in"}`}>
                            {isSelected ? <CheckCircle2 className="w-6 h-6" /> : <Circle className="w-6 h-6" />}
                        </div>
                    </div>
                )}
            </div>

            <div className={`flex flex-col items-center gap-0.5 w-full px-1 transition-opacity ${isDeleteMode && !isSelected ? "opacity-60" : "opacity-100"}`}>
                {editing ? (
                    <div className="flex items-center gap-1 w-full justify-center">
                        <input
                            type="text"
                            value={name}
                            onChange={e => setName(e.target.value)}
                            onKeyDown={e => { if (e.key === "Enter") handleSave(); if (e.key === "Escape") setEditing(false); }}
                            className="w-24 text-xs border border-primary/30 rounded px-1.5 py-0.5 text-center bg-background outline-none focus:border-primary"
                            autoFocus
                        />
                        <button onClick={handleSave} className="text-emerald-500 hover:text-emerald-400"><Check className="w-3.5 h-3.5" /></button>
                        <button onClick={() => setEditing(false)} className="text-muted-foreground hover:text-foreground"><X className="w-3.5 h-3.5" /></button>
                    </div>
                ) : (
                    <div className="flex items-center gap-1 group/name">
                        <span className="text-sm font-medium truncate max-w-[100px] text-center">{displayName}</span>
                        {!isDeleteMode && (
                            <button
                                onClick={(e) => { e.stopPropagation(); setName(person.name || ""); setEditing(true); }}
                                className="opacity-0 group-hover/name:opacity-100 text-muted-foreground hover:text-primary transition-opacity"
                            >
                                <Pencil className="w-3 h-3" />
                            </button>
                        )}
                    </div>
                )}
                <span className="text-xs text-muted-foreground">{person.photo_count} ảnh</span>
            </div>
        </div>
    );
}

export function PeopleView({ people = [], onNavigate }: PeopleViewProps) {
    const [localPeople, setLocalPeople] = useState<PersonGroup[]>(people);
    const [isDeleteMode, setIsDeleteMode] = useState(false);
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

    useEffect(() => {
        setLocalPeople(people);
    }, [people]);

    const handleRename = (faceId: string, name: string) => {
        setLocalPeople(prev =>
            prev.map(p => p.face_id === faceId ? { ...p, name } : p)
        );
    };

    const toggleSelection = (id: string) => {
        setSelectedIds(prev => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });
    };

    const handleDelete = async () => {
        const idsToDelete = Array.from(selectedIds);
        try {
            await Promise.all(idsToDelete.map(id => AuraSeekApi.deletePerson(id)));
            // Refresh local state
            setLocalPeople(prev => prev.filter(p => !selectedIds.has(p.face_id)));
            setIsDeleteMode(false);
            setSelectedIds(new Set());
            setShowDeleteConfirm(false);
        } catch (e) {
            console.error("Failed to delete people groups:", e);
        }
    };

    return (
        <div className="flex-1 overflow-y-auto px-6 py-8 will-change-scroll">
            <div className="max-w-7xl mx-auto space-y-8">
                <div className="flex items-end justify-between">
                    <div>
                        <h1 className="text-2xl font-medium tracking-tight">Người</h1>
                        <p className="text-muted-foreground mt-1 text-sm">
                            Tự động nhóm khuôn mặt bằng AI. Click vào tên để đặt tên dễ nhớ hơn.
                        </p>
                    </div>
                    {localPeople.length > 0 && (
                        <div className="flex items-center gap-2">
                            {isDeleteMode ? (
                                <>
                                    <Button 
                                        variant="ghost" 
                                        size="sm"
                                        onClick={() => { setIsDeleteMode(false); setSelectedIds(new Set()); }}
                                    >
                                        Hủy
                                    </Button>
                                    <Button 
                                        variant="destructive" 
                                        size="sm"
                                        disabled={selectedIds.size === 0}
                                        onClick={() => setShowDeleteConfirm(true)}
                                        className="gap-2"
                                    >
                                        <Trash2 className="w-4 h-4" />
                                        Xóa ({selectedIds.size})
                                    </Button>
                                </>
                            ) : (
                                <Button 
                                    variant="outline" 
                                    size="sm"
                                    onClick={() => setIsDeleteMode(true)}
                                    className="gap-2 text-muted-foreground hover:text-destructive"
                                >
                                    <Trash2 className="w-4 h-4" />
                                    Quản lý nhóm
                                </Button>
                            )}
                        </div>
                    )}
                </div>

                {localPeople.length === 0 ? (
                    <div className="text-center py-20 text-muted-foreground opacity-60">
                        <div className="text-5xl mb-4">🤖</div>
                        <p className="font-medium">Chưa có khuôn mặt nào được nhận diện</p>
                        <p className="text-sm mt-1">Hãy quét thư viện ảnh để AI nhận diện khuôn mặt tự động</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-2 min-[500px]:grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 xl:grid-cols-8 gap-y-10 gap-x-4">
                        {localPeople.map((person, i) => (
                            <PersonCard
                                key={person.face_id}
                                person={person}
                                index={i}
                                onNavigate={onNavigate}
                                onRename={handleRename}
                                isDeleteMode={isDeleteMode}
                                isSelected={selectedIds.has(person.face_id)}
                                onToggleSelect={toggleSelection}
                            />
                        ))}
                    </div>
                )}
            </div>

            <ConfirmDialog 
                isOpen={showDeleteConfirm}
                onCancel={() => setShowDeleteConfirm(false)}
                title={`Xóa ${selectedIds.size} nhóm người?`}
                description="Hành động này sẽ xóa dữ liệu nhóm khuôn mặt này. Ảnh gốc vẫn sẽ được giữ lại, nhưng AI sẽ coi như chưa nhận diện được các khuôn mặt này trong nhóm."
                onConfirm={handleDelete}
                confirmText="Xóa dữ liệu"
                isDestructive={true}
            />
        </div>
    );
}
