/**
 * FullScreenVideoViewer – xem video fullscreen, tách riêng khỏi FullScreenPhotoViewer.
 *
 * WebKitGTK (dùng bởi Tauri trên Linux) phát video qua GStreamer:
 * - Hỗ trợ tốt: MP4 container + H.264 codec (AVC) – dùng định dạng này để test.
 * - WebM/VP9: phụ thuộc GStreamer plugins, thường thiếu mặc định.
 */
import { ExternalLink } from "lucide-react";
import type { Photo } from "@/types/photo.type";
import { PhotoInfoPanel } from "./PhotoInfoPanel";
import { useState, useEffect } from "react";
import { createPortal } from "react-dom";
import { AuraSeekApi } from "@/lib/api";
import { FullScreenTopBar } from "./FullScreenTopBar";
import { openPath } from "@tauri-apps/plugin-opener";

/** Ứng dụng mở video ngoài (Linux/macOS/Windows). */
const EXTERNAL_VIDEO_APP = "cine";

export function FullScreenVideoViewer({
    photo,
    onClose,
    isTrashMode = false,
    isHiddenMode = false,
}: {
    photo: Photo;
    onClose: () => void;
    isTrashMode?: boolean;
    isHiddenMode?: boolean;
}) {
    const [showInfo, setShowInfo] = useState(true);
    const [isFavorite, setIsFavorite] = useState(photo.favorite || false);
    const [isSharing, setIsSharing] = useState(false);
    const [videoError, setVideoError] = useState(false);
    const [streamUrl, setStreamUrl] = useState<string | null>(null);

    useEffect(() => {
        setIsFavorite(photo.favorite || false);
        setVideoError(false);
        setStreamUrl(null);

        let active = true;

        if (photo.filePath) {
            // Bypass WebKitGTK/GStreamer asset streaming limitations by using a local HTTP stream
            AuraSeekApi.getStreamPort().then(port => {
                if (active && port) {
                    setStreamUrl(`http://127.0.0.1:${port}/stream?path=${encodeURIComponent(photo.filePath!)}`);
                } else if (active) {
                    setVideoError(true);
                }
            }).catch(err => {
                console.error("Failed to get stream port:", err);
                if (active) setVideoError(true);
            });
        } else {
            setVideoError(true);
        }

        return () => {
            active = false;
        };
    }, [photo.id, photo.filePath, photo.favorite]);

    const handleFavorite = async () => {
        try {
            const nextState = !isFavorite;
            setIsFavorite(nextState);
            window.dispatchEvent(new CustomEvent("photo_toggle_favorite", { detail: { id: photo.id } }));
            await AuraSeekApi.toggleFavorite(photo.id);
        } catch (e) {
            console.error("Toggle favorite failed", e);
            setIsFavorite(!isFavorite);
            window.dispatchEvent(new Event("refresh_photos"));
        }
    };

    const handleShare = async () => {
        try {
            setIsSharing(true);
            const response = await fetch(photo.url);
            const blob = await response.blob();
            await navigator.clipboard.write([
                new ClipboardItem({ [blob.type]: blob }),
            ]);
        } catch (e) {
            console.error("Share failed", e);
        } finally {
            setIsSharing(false);
        }
    };

    const handleMoveToTrash = async () => {
        try {
            await AuraSeekApi.moveToTrash(photo.id);
            window.dispatchEvent(new Event("refresh_photos"));
            onClose();
        } catch (e) {
            console.error("Move to trash failed", e);
        }
    };

    const handleRestoreFromTrash = async () => {
        try {
            await AuraSeekApi.restoreFromTrash(photo.id);
            window.dispatchEvent(new Event("refresh_photos"));
            onClose();
        } catch (e) {
            console.error("Restore failed", e);
        }
    };

    const handleHide = async () => {
        try {
            await AuraSeekApi.hidePhoto(photo.id);
            window.dispatchEvent(new Event("refresh_photos"));
            onClose();
        } catch (e) {
            console.error("Hide failed", e);
        }
    };

    const handleUnhide = async () => {
        try {
            await AuraSeekApi.unhidePhoto(photo.id);
            window.dispatchEvent(new Event("refresh_photos"));
            onClose();
        } catch (e) {
            console.error("Unhide failed", e);
        }
    };

    const handleOpenWithCine = async () => {
        if (!photo.filePath) return;
        try {
            await openPath(photo.filePath, EXTERNAL_VIDEO_APP);
        } catch (e) {
            console.error("Open with Cine failed", e);
        }
    };

    return createPortal(
        <div className="fixed inset-0 z-[9999] flex bg-background w-full h-full text-foreground">
            <div className="relative flex-1 flex flex-col overflow-hidden bg-black transition-all">
                <FullScreenTopBar
                    hasOverlays={false}
                    showBbox={false}
                    onToggleBbox={() => { }}
                    showMask={false}
                    onToggleMask={() => { }}
                    scale={1}
                    onZoomClick={() => { }}
                    isTrashMode={isTrashMode}
                    isHiddenMode={isHiddenMode}
                    isFavorite={isFavorite}
                    onToggleFavorite={handleFavorite}
                    onHide={handleHide}
                    onShare={handleShare}
                    onRestoreFromTrash={handleRestoreFromTrash}
                    onUnhide={handleUnhide}
                    onMoveToTrash={handleMoveToTrash}
                    isSharing={isSharing}
                    showInfo={showInfo}
                    onToggleInfo={() => setShowInfo((p) => !p)}
                    onClose={onClose}
                    isVideo
                />

                <div className="flex-1 flex items-center justify-center p-4 bg-black">
                    {videoError ? (
                        <div className="flex flex-col items-center justify-center gap-4 text-center max-w-md">
                            <img
                                src={photo.thumbnailUrl}
                                alt="Video thumbnail"
                                className="max-h-64 rounded-lg object-contain opacity-80"
                            />
                            <p className="text-muted-foreground text-sm">
                                Video không phát được trong app. WebKitGTK hỗ trợ tốt nhất MP4 (H.264).
                                Mở bằng Cine để xem.
                            </p>
                            <button
                                type="button"
                                onClick={handleOpenWithCine}
                                disabled={!photo.filePath}
                                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                <ExternalLink className="w-4 h-4" />
                                Mở bằng Cine
                            </button>
                        </div>
                    ) : !streamUrl ? (
                        <div className="flex items-center justify-center">
                            <span className="text-slate-400 animate-pulse text-sm font-medium">Đang kết nối luồng phát...</span>
                        </div>
                    ) : (
                        <video
                            src={streamUrl}
                            poster={photo.thumbnailUrl}
                            controls
                            autoPlay
                            className="max-w-full max-h-full rounded-xl bg-black shadow-2xl"
                            onError={(e) => {
                                const err = (e.target as HTMLVideoElement).error;
                                console.error("Video playback error:", err);
                                setVideoError(true);
                            }}
                        >
                            Trình duyệt không hỗ trợ định dạng này.
                        </video>
                    )}
                </div>
            </div>

            {showInfo && (
                <div className="w-[360px] md:w-[400px] shrink-0 border-l border-border/20 bg-background flex flex-col h-full overflow-hidden transition-all shadow-xl">
                    <div className="h-14 flex items-center px-4 border-b border-border/10">
                        <h2 className="text-lg font-medium tracking-tight">Thông tin</h2>
                    </div>
                    <div className="flex-1 overflow-y-auto">
                        <PhotoInfoPanel photo={photo} />
                    </div>
                </div>
            )}
        </div>,
        document.body
    );
}
