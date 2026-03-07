import { useState, useEffect, useCallback } from "react";
import type { PersonGroup } from "@/lib/api";
import { localFileUrl, AuraSeekApi } from "@/lib/api";
import { Pencil, Check, X, User } from "lucide-react";

interface PeopleViewProps {
    people?: PersonGroup[];
    onNavigate?: (payload: any) => void;
}

const AVATAR_PX = 112;

/**
 * Renders a face-cropped avatar.
 * Takes the face bbox, doubles its size (centered on face), and crops.
 * Uses CSS background-image for reliable crop rendering.
 */
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

        // Face center in original image pixels
        const faceCx = bbox.x + bbox.w / 2;
        const faceCy = bbox.y + bbox.h / 2;

        // Crop region = 2x the face bbox, made square
        const cropSize = Math.max(bbox.w, bbox.h) * 2;

        // Center the square crop on the face, clamped to image bounds
        let cropX = faceCx - cropSize / 2;
        let cropY = faceCy - cropSize / 2;
        const clampedSize = Math.min(cropSize, naturalW, naturalH);
        cropX = Math.max(0, Math.min(cropX, naturalW - clampedSize));
        cropY = Math.max(0, Math.min(cropY, naturalH - clampedSize));

        // background-size: scale full image so cropSize maps to AVATAR_PX
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
}: {
    person: PersonGroup;
    index: number;
    onNavigate?: (payload: any) => void;
    onRename?: (faceId: string, name: string) => void;
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

    // Use thumbnail (matches face_bbox) when available, fall back to cover_path
    const imgPath = person.thumbnail || person.cover_path;
    const imageUrl = imgPath ? localFileUrl(imgPath) : null;

    return (
        <div className="group flex flex-col items-center gap-3">
            <div
                className="relative rounded-full overflow-hidden bg-muted transition-all duration-300 ring-2 ring-transparent group-hover:ring-primary shadow-sm hover:shadow-md cursor-pointer"
                style={{ width: AVATAR_PX, height: AVATAR_PX }}
                onClick={() => onNavigate?.({ id: person.face_id, title: displayName })}
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
            </div>

            <div className="flex flex-col items-center gap-0.5 w-full px-1">
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
                        <button
                            onClick={() => { setName(person.name || ""); setEditing(true); }}
                            className="opacity-0 group-hover/name:opacity-100 text-muted-foreground hover:text-primary transition-opacity"
                        >
                            <Pencil className="w-3 h-3" />
                        </button>
                    </div>
                )}
                <span className="text-xs text-muted-foreground">{person.photo_count} ảnh</span>
            </div>
        </div>
    );
}

export function PeopleView({ people = [], onNavigate }: PeopleViewProps) {
    const [localPeople, setLocalPeople] = useState<PersonGroup[]>(people);

    useEffect(() => {
        setLocalPeople(people);
    }, [people]);

    const handleRename = (faceId: string, name: string) => {
        setLocalPeople(prev =>
            prev.map(p => p.face_id === faceId ? { ...p, name } : p)
        );
    };

    return (
        <div className="flex-1 overflow-y-auto px-6 py-8 will-change-scroll">
            <div className="max-w-7xl mx-auto space-y-8">
                <div>
                    <h1 className="text-2xl font-medium tracking-tight">Người</h1>
                    <p className="text-muted-foreground mt-1 text-sm">
                        Tự động nhóm khuôn mặt bằng AI. Click vào tên để đặt tên dễ nhớ hơn.
                    </p>
                </div>

                {localPeople.length === 0 ? (
                    <div className="text-center py-20 text-muted-foreground opacity-60">
                        <div className="text-5xl mb-4">🤖</div>
                        <p className="font-medium">Chưa có khuôn mặt nào được nhận diện</p>
                        <p className="text-sm mt-1">Hãy quét thư viện ảnh để AI nhận diện khuôn mặt tự động</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-2 min-[500px]:grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 xl:grid-cols-8 gap-y-8 gap-x-4">
                        {localPeople.map((person, i) => (
                            <PersonCard
                                key={person.face_id}
                                person={person}
                                index={i}
                                onNavigate={onNavigate}
                                onRename={handleRename}
                            />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
