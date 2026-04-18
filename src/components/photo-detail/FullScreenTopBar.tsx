import type React from "react";
import { ArrowLeft, Info, Share2, Heart, Trash2, Eye, EyeOff, ZoomIn, Undo, Scan, Paintbrush } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";

interface FullScreenTopBarProps {
    hasOverlays: boolean;
    showBbox: boolean;
    onToggleBbox: () => void;
    showMask: boolean;
    onToggleMask: () => void;
    scale: number;
    onZoomClick: (e: React.MouseEvent) => void;
    isTrashMode: boolean;
    isHiddenMode: boolean;
    isFavorite: boolean;
    onToggleFavorite: () => void;
    onHide: () => void;
    onShare: () => void;
    onRestoreFromTrash: () => void;
    onUnhide: () => void;
    onMoveToTrash: () => void;
    isSharing: boolean;
    showInfo: boolean;
    onToggleInfo: () => void;
    onClose: () => void;
    isVideo?: boolean;
}

export function FullScreenTopBar({
    hasOverlays,
    showBbox,
    onToggleBbox,
    showMask,
    onToggleMask,
    scale,
    onZoomClick,
    isTrashMode,
    isHiddenMode,
    isFavorite,
    onToggleFavorite,
    onHide,
    onShare,
    onRestoreFromTrash,
    onUnhide,
    onMoveToTrash,
    isSharing,
    showInfo,
    onToggleInfo,
    onClose,
    isVideo = false,
}: FullScreenTopBarProps) {
    return (
        <div className="h-14 shrink-0 flex items-center justify-between px-2 bg-black text-white z-10">
            <div className="flex items-center gap-2">
                <Tooltip>
                    <TooltipTrigger asChild>
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={onClose}
                            className="rounded-full text-white/80 hover:text-white hover:bg-white/20"
                        >
                            <ArrowLeft className="w-5 h-5" />
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" className="text-xs">
                        <p>Quay lại</p>
                    </TooltipContent>
                </Tooltip>
            </div>
            <div className="flex items-center gap-1">
                {hasOverlays && (
                    <>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    onClick={onToggleBbox}
                                    className={`rounded-full transition-colors ${
                                        showBbox
                                            ? "bg-white/20 text-white hover:bg-white/30"
                                            : "text-white/80 hover:text-white hover:bg-white/20"
                                    }`}
                                >
                                    <Scan className="w-5 h-5" />
                                </Button>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="text-xs">
                                <p>{showBbox ? "Ẩn khung AI (bbox)" : "Hiện khung AI (bbox)"}</p>
                            </TooltipContent>
                        </Tooltip>

                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    onClick={onToggleMask}
                                    className={`rounded-full transition-colors ${
                                        showMask
                                            ? "bg-white/20 text-white hover:bg-white/30"
                                            : "text-white/80 hover:text-white hover:bg-white/20"
                                    }`}
                                >
                                    <Paintbrush className="w-5 h-5" />
                                </Button>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="text-xs">
                                <p>{showMask ? "Ẩn mask AI" : "Hiện mask AI"}</p>
                            </TooltipContent>
                        </Tooltip>
                    </>
                )}

                {!isVideo && (
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Button
                                variant="ghost"
                                size="icon"
                                className={`rounded-full transition-colors ${
                                    scale > 1
                                        ? "bg-white/20 text-white hover:bg-white/30"
                                        : "text-white/80 hover:text-white hover:bg-white/20"
                                }`}
                                onClick={onZoomClick}
                            >
                                <ZoomIn className="w-5 h-5" />
                            </Button>
                        </TooltipTrigger>
                        <TooltipContent side="bottom" className="text-xs">
                            <p>Thu phóng ({Math.round(scale * 100)}%)</p>
                        </TooltipContent>
                    </Tooltip>
                )}

                {/* Common actions based on mode */}
                {isTrashMode ? (
                    <>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="rounded-full hover:bg-white/10 text-white"
                                    onClick={onRestoreFromTrash}
                                >
                                    <Undo className="w-5 h-5" />
                                </Button>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="text-xs">
                                Khôi phục
                            </TooltipContent>
                        </Tooltip>
                    </>
                ) : isHiddenMode ? (
                    <>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="rounded-full hover:bg-white/10 text-white"
                                    onClick={onUnhide}
                                >
                                    <Eye className="w-5 h-5" />
                                </Button>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="text-xs">
                                Bỏ ẩn
                            </TooltipContent>
                        </Tooltip>
                    </>
                ) : (
                    <>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className={`rounded-full transition-colors ${
                                        isFavorite
                                            ? "text-red-500 bg-red-500/10 hover:bg-red-500/20"
                                            : "text-white hover:bg-white/10"
                                    }`}
                                    onClick={onToggleFavorite}
                                >
                                    <Heart className={`w-5 h-5 ${isFavorite ? "fill-current" : ""}`} />
                                </Button>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="text-xs">
                                Yêu thích
                            </TooltipContent>
                        </Tooltip>

                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="rounded-full hover:bg-white/10 text-white"
                                    onClick={onHide}
                                >
                                    <EyeOff className="w-5 h-5" />
                                </Button>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="text-xs">
                                Ẩn ảnh
                            </TooltipContent>
                        </Tooltip>

                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="rounded-full hover:bg-white/10 text-white"
                                    onClick={onShare}
                                    disabled={isSharing}
                                >
                                    <Share2 className="w-5 h-5" />
                                </Button>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="text-xs">
                                Chia sẻ
                            </TooltipContent>
                        </Tooltip>

                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="rounded-full text-white hover:text-destructive-foreground hover:bg-destructive"
                                    onClick={onMoveToTrash}
                                >
                                    <Trash2 className="w-5 h-5" />
                                </Button>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="text-xs">
                                Xóa vào thùng rác
                            </TooltipContent>
                        </Tooltip>
                    </>
                )}

                <div className="h-6 w-px bg-white/20 mx-1" />

                <Tooltip>
                    <TooltipTrigger asChild>
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={onToggleInfo}
                            className={`rounded-full transition-colors ${
                                showInfo
                                    ? "bg-white/20 text-white hover:bg-white/30"
                                    : "text-white/80 hover:text-white hover:bg-white/20"
                            }`}
                        >
                            <Info className="w-5 h-5" />
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" className="text-xs">
                        <p>Thông tin</p>
                    </TooltipContent>
                </Tooltip>
            </div>
        </div>
    );
}

