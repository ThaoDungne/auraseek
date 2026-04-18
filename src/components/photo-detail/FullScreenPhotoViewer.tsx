import { Plus, Minus } from "lucide-react";
import type { Photo } from "@/types/photo.type";
import { PhotoInfoPanel } from "./PhotoInfoPanel";
import { useState, useRef, useEffect } from "react";
import { createPortal } from "react-dom";
import { AuraSeekApi } from "@/lib/api";
import { SegmentOverlay } from "../photos/SegmentOverlay";
import { FullScreenTopBar } from "./FullScreenTopBar";

export function FullScreenPhotoViewer({
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
    const [showBbox, setShowBbox] = useState(() => {
        const saved = localStorage.getItem(`auraseek_bbox_${photo.id}`);
        return saved === "true";
    });
  // Mask mặc định tắt để không gây bất ngờ; người dùng bật bằng icon cọ vẽ.
  const [showMask, setShowMask] = useState(false);
  const [activeObjectIndex, setActiveObjectIndex] = useState<number | null>(null);
    const [isFavorite, setIsFavorite] = useState(photo.favorite || false);
    const [isSharing, setIsSharing] = useState(false);

    useEffect(() => {
        const saved = localStorage.getItem(`auraseek_bbox_${photo.id}`);
        setShowBbox(saved === "true");
        setIsFavorite(photo.favorite || false);
    }, [photo.id, photo.favorite]);

    const imgRef = useRef<HTMLImageElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

    const [scale, setScale] = useState(1);
    const panRef = useRef({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const dragStart = useRef({ x: 0, y: 0 });
    const MIN_SCALE = 1;
    const MAX_SCALE = 5;

    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current) {
                setDimensions({
                    width: containerRef.current.clientWidth,
                    height: containerRef.current.clientHeight,
                });
            }
        };
        window.addEventListener("resize", updateDimensions);
        updateDimensions();
        return () => window.removeEventListener("resize", updateDimensions);
    }, []);

    let renderedW = dimensions.width - 32;
    let renderedH = dimensions.height - 32;
    if (imgRef.current && dimensions.width && dimensions.height) {
        const natW = photo.width || imgRef.current.naturalWidth || 1;
        const natH = photo.height || imgRef.current.naturalHeight || 1;
        const imgAspect = natW / natH;
        const containerAspect = (dimensions.width - 32) / (dimensions.height - 32);
        if (imgAspect > containerAspect) {
            renderedW = dimensions.width - 32;
            renderedH = (dimensions.width - 32) / imgAspect;
        } else {
            renderedH = dimensions.height - 32;
            renderedW = (dimensions.height - 32) * imgAspect;
        }
    }

    const clampPan = (p: { x: number; y: number }, curScale: number) => {
        const maxX = Math.max(0, (renderedW * curScale - (dimensions.width - 32)) / 2);
        const maxY = Math.max(0, (renderedH * curScale - (dimensions.height - 32)) / 2);
        return {
            x: Math.min(Math.max(p.x, -maxX), maxX),
            y: Math.min(Math.max(p.y, -maxY), maxY),
        };
    };

    const applyPanToDOM = (x: number, y: number, currentScale: number) => {
        panRef.current = { x, y };
        const contentDiv = document.getElementById(`fst-content-div-${photo.id}`);
        if (contentDiv) {
            contentDiv.style.transform = `translate(${x}px, ${y}px) scale(${currentScale})`;
        }
        const minimapBox = document.getElementById(`fst-minimap-box-${photo.id}`);
        if (minimapBox && currentScale > 1) {
            const left = Math.max(0, Math.min(100, (((renderedW * currentScale - (dimensions.width - 32)) / 2 - x) / (renderedW * currentScale)) * 100));
            const top = Math.max(0, Math.min(100, (((renderedH * currentScale - (dimensions.height - 32)) / 2 - y) / (renderedH * currentScale)) * 100));
            minimapBox.style.left = `${left}%`;
            minimapBox.style.top = `${top}%`;
        }
    };

    useEffect(() => {
        applyPanToDOM(panRef.current.x, panRef.current.y, scale);
    }, [scale, dimensions.width, dimensions.height]);

    const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
        if (!containerRef.current) return;
        const zoomDelta = -e.deltaY * 0.005;
        let newScale = scale * Math.exp(zoomDelta);
        newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, newScale));
        if (newScale === MIN_SCALE) {
            setScale(MIN_SCALE);
            applyPanToDOM(0, 0, MIN_SCALE);
            return;
        }
        const rect = containerRef.current.getBoundingClientRect();
        const clientX = e.clientX - rect.left - rect.width / 2;
        const clientY = e.clientY - rect.top - rect.height / 2;
        const scaleRatio = newScale / scale;
        const newPanX = clientX - (clientX - panRef.current.x) * scaleRatio;
        const newPanY = clientY - (clientY - panRef.current.y) * scaleRatio;
        setScale(newScale);
        const clamped = clampPan({ x: newPanX, y: newPanY }, newScale);
        applyPanToDOM(clamped.x, clamped.y, newScale);
    };

    const handleDoubleClick = (e: React.MouseEvent) => {
        if (scale > 1) {
            setScale(1);
            applyPanToDOM(0, 0, 1);
        } else {
            const newScale = 2.5;
            const rect = containerRef.current!.getBoundingClientRect();
            const clientX = e.clientX - rect.left - rect.width / 2;
            const clientY = e.clientY - rect.top - rect.height / 2;
            setScale(newScale);
            const clamped = clampPan({ x: -clientX * 1.5, y: -clientY * 1.5 }, newScale);
            applyPanToDOM(clamped.x, clamped.y, newScale);
        }
    };

    const handlePointerDown = (e: React.PointerEvent) => {
        if (scale === 1) return;
        setIsDragging(true);
        dragStart.current = { x: e.clientX - panRef.current.x, y: e.clientY - panRef.current.y };
        e.currentTarget.setPointerCapture(e.pointerId);
    };

    const handlePointerMove = (e: React.PointerEvent) => {
        if (!isDragging) return;
        const newPanX = e.clientX - dragStart.current.x;
        const newPanY = e.clientY - dragStart.current.y;
        const clamped = clampPan({ x: newPanX, y: newPanY }, scale);
        applyPanToDOM(clamped.x, clamped.y, scale);
    };

    const handlePointerUp = (e: React.PointerEvent) => {
        setIsDragging(false);
        e.currentTarget.releasePointerCapture(e.pointerId);
    };

    const [isMinimapDragging, setIsMinimapDragging] = useState(false);
    const minimapDragStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

    const handleMinimapPointerDown = (e: React.PointerEvent) => {
        e.stopPropagation();
        if (scale === 1) return;
        setIsMinimapDragging(true);
        minimapDragStart.current = {
            x: e.clientX,
            y: e.clientY,
            panX: panRef.current.x,
            panY: panRef.current.y,
        };
        e.currentTarget.setPointerCapture(e.pointerId);
    };

    const handleMinimapPointerMove = (e: React.PointerEvent) => {
        if (!isMinimapDragging) return;
        e.stopPropagation();
        const dx = e.clientX - minimapDragStart.current.x;
        const dy = e.clientY - minimapDragStart.current.y;
        const minimapW = 144;
        const ratio = (renderedW * scale) / minimapW;
        const newPanX = minimapDragStart.current.panX - dx * ratio;
        const newPanY = minimapDragStart.current.panY - dy * ratio;
        const clamped = clampPan({ x: newPanX, y: newPanY }, scale);
        applyPanToDOM(clamped.x, clamped.y, scale);
    };

    const handleMinimapPointerUp = (e: React.PointerEvent) => {
        setIsMinimapDragging(false);
        e.currentTarget.releasePointerCapture(e.pointerId);
    };

  // ─── Object/face hit-testing for "press to mask" ─────────────────────
  const handleObjectPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!imgRef.current || !photo.detectedObjects || photo.detectedObjects.length === 0) {
      return;
    }
    const imgBox = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - imgBox.left;
    const py = e.clientY - imgBox.top;
    if (px < 0 || py < 0 || px > imgBox.width || py > imgBox.height) return;

    const natW = photo.width || imgRef.current.naturalWidth || 1;
    const natH = photo.height || imgRef.current.naturalHeight || 1;
    const scaleX = imgBox.width / natW;
    const scaleY = imgBox.height / natH;
    const ox = px / scaleX;
    const oy = py / scaleY;

    const idx = photo.detectedObjects.findIndex((obj) => {
      const { x, y, w, h } = obj.bbox;
      return ox >= x && ox <= x + w && oy >= y && oy <= y + h;
    });

    if (idx >= 0) {
      // Tạm thời chỉ highlight object này; không đụng tới state nút cọ.
      setActiveObjectIndex(idx);
    }
  };

  const handleObjectPointerUp = () => {
    setActiveObjectIndex(null);
  };

    const hasOverlays = Boolean(
        (photo.detectedObjects && photo.detectedObjects.length > 0) ||
            (photo.detectedFaces && photo.detectedFaces.length > 0)
    );

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
            await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
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

    const handleToggleBbox = () => {
        const newVal = !showBbox;
        setShowBbox(newVal);
        localStorage.setItem(`auraseek_bbox_${photo.id}`, newVal.toString());
    };

  const handleToggleMask = () => {
    setShowMask((prev) => !prev);
  };

    return createPortal(
        <div className="fixed inset-0 z-[9999] flex bg-background w-full h-full text-foreground">
            <div className="relative flex-1 flex flex-col overflow-hidden bg-black transition-all">
                <FullScreenTopBar
                    hasOverlays={hasOverlays}
                    showBbox={showBbox}
                    onToggleBbox={handleToggleBbox}
                    showMask={showMask}
                    onToggleMask={handleToggleMask}
                    scale={scale}
                    onZoomClick={handleDoubleClick}
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
                    isVideo={false}
                />

                <div
                    ref={containerRef}
                    className="flex-1 flex items-center justify-center p-4 relative overflow-hidden select-none outline-none"
                    onWheel={handleWheel}
                    onDoubleClick={handleDoubleClick}
                    onPointerDown={handlePointerDown}
                    onPointerMove={handlePointerMove}
                    onPointerUp={handlePointerUp}
                    onPointerCancel={handlePointerUp}
                    style={{ cursor: scale > 1 ? (isDragging ? "grabbing" : "grab") : "default" }}
                >
                    <div
                        id={`fst-content-div-${photo.id}`}
                        className="relative w-full h-full flex justify-center items-center pointer-events-none"
                        style={{
                            transform: `translate(${panRef.current.x}px, ${panRef.current.y}px) scale(${scale})`,
                            transformOrigin: "center center",
                            transition: isDragging ? "none" : "transform 0.1s ease-out, scale 0.1s ease-out",
                        }}
                    >
                        <div className="relative pointer-events-auto shadow-2xl" style={{ width: renderedW, height: renderedH }}>
                            <div
                                className="w-full h-full relative"
                                onPointerDown={handleObjectPointerDown}
                                onPointerUp={handleObjectPointerUp}
                                onPointerLeave={handleObjectPointerUp}
                            >
                                <img
                                    ref={imgRef}
                                    src={photo.url}
                                    alt="View"
                                    className="w-full h-full object-contain"
                                    draggable={false}
                                    onLoad={() => {
                                        setDimensions({
                                            width: containerRef.current?.clientWidth || 0,
                                            height: containerRef.current?.clientHeight || 0,
                                        });
                                    }}
                                />
                                {(showBbox || showMask || activeObjectIndex != null) && hasOverlays && renderedW > 0 && (
                                    <SegmentOverlay
                                        detectedObjects={photo.detectedObjects}
                                        detectedFaces={photo.detectedFaces}
                                        imgNaturalW={photo.width || imgRef.current?.naturalWidth || 0}
                                        imgNaturalH={photo.height || imgRef.current?.naturalHeight || 0}
                                        displayW={renderedW}
                                        displayH={renderedH}
                                        objectFit="contain"
                                        showFaces={showBbox}
                                        showLabels
                                        showMasks={showMask || activeObjectIndex != null}
                                        showBoxes={showBbox}
                                        viewScale={scale}
                                        activeObjectIndex={activeObjectIndex}
                                    />
                                )}
                            </div>
                        </div>
                    </div>

                    {scale > 1 && (
                        <div className="absolute top-4 right-4 z-20 flex flex-col items-end gap-2 text-white pointer-events-auto w-[160px]">
                            <div className="bg-black/60 backdrop-blur-md border border-white/20 rounded-xl p-2 shadow-2xl relative w-full flex items-center justify-center">
                                <div className="relative" style={{ width: 144, height: 144 * (renderedH / renderedW) }}>
                                    <img src={photo.url} className="w-full h-full object-contain opacity-60 rounded-sm" draggable={false} />
                                    <div
                                        id={`fst-minimap-box-${photo.id}`}
                                        className="absolute border border-white/80 bg-white/20 rounded-[2px] shadow-[0_0_0_9999px_rgba(0,0,0,0.4)] pointer-events-auto cursor-move"
                                        onPointerDown={handleMinimapPointerDown}
                                        onPointerMove={handleMinimapPointerMove}
                                        onPointerUp={handleMinimapPointerUp}
                                        onPointerCancel={handleMinimapPointerUp}
                                        style={{
                                            width: `${Math.min(100, ((dimensions.width - 32) / (renderedW * scale)) * 100)}%`,
                                            height: `${Math.min(100, ((dimensions.height - 32) / (renderedH * scale)) * 100)}%`,
                                            left: `${Math.max(0, Math.min(100, (((renderedW * scale - (dimensions.width - 32)) / 2 - panRef.current.x) / (renderedW * scale)) * 100))}%`,
                                            top: `${Math.max(0, Math.min(100, (((renderedH * scale - (dimensions.height - 32)) / 2 - panRef.current.y) / (renderedH * scale)) * 100))}%`,
                                            transition: isDragging || isMinimapDragging ? "none" : "all 0.1s ease-out",
                                        }}
                                    />
                                </div>
                            </div>
                            <div
                                className="bg-black/60 backdrop-blur-md border border-white/20 rounded-full flex items-center px-3 py-2 text-white gap-2 shadow-2xl w-full cursor-default"
                                onPointerDown={(e) => e.stopPropagation()}
                                onPointerMove={(e) => e.stopPropagation()}
                                onPointerUp={(e) => e.stopPropagation()}
                                onPointerCancel={(e) => e.stopPropagation()}
                            >
                                <Minus className="w-4 h-4 shrink-0 text-white/70" />
                                <input
                                    type="range"
                                    min={MIN_SCALE}
                                    max={MAX_SCALE}
                                    step={0.01}
                                    value={scale}
                                    onChange={(e) => {
                                        const ns = parseFloat(e.target.value);
                                        setScale(ns);
                                        const clamped = clampPan(panRef.current, ns);
                                        applyPanToDOM(clamped.x, clamped.y, ns);
                                    }}
                                    className="flex-1 min-w-0 h-1.5 bg-white/30 rounded-full custom-zoom-slider"
                                />
                                <Plus className="w-4 h-4 shrink-0 text-white/70" />
                            </div>
                        </div>
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
