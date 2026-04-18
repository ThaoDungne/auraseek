import type { Photo } from "@/types/photo.type";
import { FullScreenPhotoViewer } from "./FullScreenPhotoViewer";
import { FullScreenVideoViewer } from "./FullScreenVideoViewer";

/**
 * Router: chọn FullScreenPhotoViewer hoặc FullScreenVideoViewer theo loại media.
 * Callers giữ nguyên interface, phân biệt ảnh/video bằng file riêng.
 */
export function FullScreenViewer({
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
    if (photo.type === "video") {
        return (
            <FullScreenVideoViewer
                photo={photo}
                onClose={onClose}
                isTrashMode={isTrashMode}
                isHiddenMode={isHiddenMode}
            />
        );
    }
    return (
        <FullScreenPhotoViewer
            photo={photo}
            onClose={onClose}
            isTrashMode={isTrashMode}
            isHiddenMode={isHiddenMode}
        />
    );
}
