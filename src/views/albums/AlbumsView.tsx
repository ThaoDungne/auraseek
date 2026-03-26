import { Plus, FolderHeart, LibraryBig, Trash2, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Photo } from "@/types/photo.type";
import { useState, useEffect } from "react";
import { AuraSeekApi, localFileUrl } from "@/lib/api";
import { PromptDialog } from "@/components/ui/PromptDialog";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";

export function AlbumsView({ photos = [], onNavigate }: { photos?: Photo[], onNavigate?: (payload: any) => void }) {
    const [manualAlbums, setManualAlbums] = useState<{ id: string, title: string, count: number, cover_url: string | null }[]>([]);
    const [isCreateOpen, setIsCreateOpen] = useState(false);
    const [isCreating, setIsCreating] = useState(false);

    // Create album state
    const [selectionMode, setSelectionMode] = useState(false);
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [newAlbumTitle, setNewAlbumTitle] = useState("");

    // Delete album state
    const [deleteMode, setDeleteMode] = useState(false);
    const [deleteSelectedIds, setDeleteSelectedIds] = useState<Set<string>>(new Set());
    const [isDeleting, setIsDeleting] = useState(false);
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

    const [actionError, setActionError] = useState<string | null>(null);

    const loadManualAlbums = async () => {
        try {
            const result = await AuraSeekApi.getAlbums();
            setManualAlbums(result);
        } catch (e) { console.error(e); }
    };

    useEffect(() => {
        loadManualAlbums();
    }, []);

    const handleStartCreate = () => setIsCreateOpen(true);

    const handleNameConfirm = (title: string) => {
        if (!title.trim()) return;
        setNewAlbumTitle(title);
        setIsCreateOpen(false);
        setSelectionMode(true);
        setSelectedIds(new Set());
    };

    const toggleMediaSelection = (id: string) => {
        const next = new Set(selectedIds);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        setSelectedIds(next);
    };

    const handleSaveAlbum = async () => {
        if (!newAlbumTitle || selectedIds.size === 0) {
            setSelectionMode(false);
            return;
        }
        try {
            setIsCreating(true);
            const albumId = await AuraSeekApi.createAlbum(newAlbumTitle);
            await AuraSeekApi.addToAlbum(albumId, Array.from(selectedIds));
            await new Promise(r => setTimeout(r, 150));
            await loadManualAlbums();
            setSelectionMode(false);
            setNewAlbumTitle("");
            setSelectedIds(new Set());
        } catch (e: any) {
            console.error(e);
            setActionError("Chi tiết lỗi: " + (e?.message || e?.toString() || "Lỗi không xác định"));
        } finally {
            setIsCreating(false);
        }
    };

    // --- Delete album logic ---
    const toggleDeleteSelection = (id: string) => {
        const next = new Set(deleteSelectedIds);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        setDeleteSelectedIds(next);
    };

    const handleConfirmDelete = async () => {
        if (deleteSelectedIds.size === 0) return;
        try {
            setIsDeleting(true);
            for (const id of deleteSelectedIds) {
                await AuraSeekApi.deleteAlbum(id);
            }
            await loadManualAlbums();
            setDeleteMode(false);
            setDeleteSelectedIds(new Set());
        } catch (e: any) {
            setActionError("Không thể xóa album: " + (e?.message || "Lỗi không xác định"));
        } finally {
            setIsDeleting(false);
            setShowDeleteConfirm(false);
        }
    };

    // Only build albums from images (exclude videos completely)
    const imagePhotos = photos.filter(p => p.type !== "video");
    const favPhotos = imagePhotos.filter(p => p.favorite);

    const collections = [
        {
            id: "fav",
            title: "Yêu thích",
            count: favPhotos.length,
            icon: FolderHeart,
            coverUrl: favPhotos[0]?.thumbnailUrl || favPhotos[0]?.url || null,
            emptyMsg: "Chưa có ảnh yêu thích nào"
        },
    ];

    // AI Albums from YOLO tags
    const albumsMap = new Map<string, { id: string; title: string; count: number; coverUrl: string; }>();
    const titleMap: Record<string, string> = {
        person: "Con người", dog: "Chó", cat: "Mèo", car: "Ô tô",
        keyboard: "Bàn phím", laptop: "Máy tính", cell_phone: "Điện thoại",
        mouse: "Chuột", cup: "Cốc cà phê", bottle: "Chai nước", book: "Sách",
        motorcycle: "Xe máy", airplane: "Máy bay", bus: "Xe buýt", truck: "Xe tải",
        bird: "Chim", horse: "Ngựa", sheep: "Cừu", cow: "Bò", elephant: "Voi",
        bear: "Gấu", zebra: "Ngựa vằn", giraffe: "Hươu cao cổ", backpack: "Balo",
        umbrella: "Cái ô", handbag: "Túi xách", tie: "Cà vạt", suitcase: "Vali",
        skateboard: "Trượt ván", surfboard: "Ván lướt sóng", tennis_racket: "Vợt tennis",
        wine_glass: "Ly rượu", fork: "Cái nĩa", knife: "Cái dao", spoon: "Cái thìa",
        bowl: "Cái bát", banana: "Quả chuối", apple: "Quả táo", pizza: "Bánh pizza",
        cake: "Bánh ngọt", chair: "Cái ghế", couch: "Ghế sofa", bed: "Cái giường",
        tv: "Tivi", refrigerator: "Tủ lạnh", clock: "Cái đồng hồ", vase: "Cái bình",
    };

    for (const p of imagePhotos) {
        if (!p.labels) continue;
        const seenTags = new Set<string>();
        for (const label of p.labels) {
            const normalizedTag = label.toLowerCase();
            if (seenTags.has(normalizedTag)) continue;
            seenTags.add(normalizedTag);
            const title = titleMap[normalizedTag] || label;
            if (!albumsMap.has(normalizedTag)) {
                albumsMap.set(normalizedTag, { id: "tag_" + normalizedTag, title, count: 0, coverUrl: p.thumbnailUrl || p.url });
            }
            albumsMap.get(normalizedTag)!.count++;
        }
    }
    const aiAlbums = Array.from(albumsMap.values()).sort((a, b) => b.count - a.count);

    const isInAnyMode = selectionMode || deleteMode;

    return (
        <div className="flex-1 overflow-y-auto px-6 py-8 will-change-scroll">
            <div className="max-w-7xl mx-auto space-y-12">

                {/* Header */}
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-medium tracking-tight">Bộ sưu tập</h1>
                        {selectionMode && (
                            <p className="text-sm text-primary animate-pulse mt-1 font-medium">
                                Đang tạo album "{newAlbumTitle}" — đã chọn {selectedIds.size} mục...
                            </p>
                        )}
                        {deleteMode && (
                            <p className="text-sm text-destructive mt-1 font-medium">
                                Chọn album để xóa — đã chọn {deleteSelectedIds.size} album
                            </p>
                        )}
                    </div>

                    {!isInAnyMode ? (
                        <div className="flex items-center gap-2">
                            {manualAlbums.length > 0 && (
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => { setDeleteMode(true); setDeleteSelectedIds(new Set()); }}
                                    className="rounded-full h-9 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                                >
                                    <Trash2 className="w-4 h-4 mr-2" />
                                    Xóa album
                                </Button>
                            )}
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={handleStartCreate}
                                className="rounded-full shadow-sm text-xs font-medium h-9 border-primary/20 hover:bg-primary/5 hover:text-primary transition-all pr-5"
                            >
                                <Plus className="w-4 h-4 mr-2" />
                                Album mới
                            </Button>
                        </div>
                    ) : selectionMode ? (
                        <div className="flex items-center gap-2">
                            <Button variant="ghost" size="sm" onClick={() => setSelectionMode(false)} className="rounded-full h-9">
                                Hủy
                            </Button>
                            <Button size="sm" onClick={handleSaveAlbum} disabled={isCreating || selectedIds.size === 0} className="rounded-full h-9 px-6 font-bold shadow-lg shadow-primary/20">
                                {isCreating ? "Đang lưu..." : `Lưu album (${selectedIds.size})`}
                            </Button>
                        </div>
                    ) : (
                        <div className="flex items-center gap-2">
                            <Button variant="ghost" size="sm" onClick={() => { setDeleteMode(false); setDeleteSelectedIds(new Set()); }} className="rounded-full h-9">
                                Hủy
                            </Button>
                            <Button
                                size="sm"
                                variant="destructive"
                                disabled={deleteSelectedIds.size === 0 || isDeleting}
                                onClick={() => setShowDeleteConfirm(true)}
                                className="rounded-full h-9 px-6 font-bold"
                            >
                                <Trash2 className="w-4 h-4 mr-2" />
                                {isDeleting ? "Đang xóa..." : `Xóa (${deleteSelectedIds.size})`}
                            </Button>
                        </div>
                    )}
                </div>

                {selectionMode ? (
                    /* Media selection grid for creating album */
                    <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 xl:grid-cols-8 gap-2">
                        {photos.map(p => (
                            <div
                                key={p.id}
                                onClick={() => toggleMediaSelection(p.id)}
                                className={`relative aspect-square rounded-lg overflow-hidden cursor-pointer group transition-all duration-300 ${selectedIds.has(p.id) ? 'ring-4 ring-primary ring-offset-2 ring-offset-background scale-95' : 'hover:opacity-90'}`}
                            >
                                <img src={p.thumbnailUrl || p.url} className={`w-full h-full object-cover transition-transform duration-500 ${selectedIds.has(p.id) ? 'opacity-50' : 'group-hover:scale-110'}`} />
                                {selectedIds.has(p.id) && (
                                    <div className="absolute inset-0 flex items-center justify-center">
                                        <CheckCircle2 className="w-8 h-8 text-primary drop-shadow-lg" />
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                ) : (
                    /* Normal Album Grid */
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-x-6 gap-y-10">
                        {/* Fixed Collections (Favorites) */}
                        {collections.map(col => (
                            <div key={col.id} className="group cursor-pointer" onClick={() => !deleteMode && onNavigate?.({ id: col.id, title: col.title })}>
                                <div className="aspect-square rounded-2xl overflow-hidden bg-muted/40 mb-3 transition-all duration-300 ring-4 ring-transparent group-hover:ring-primary/20 shadow-sm group-hover:shadow-xl relative border border-border/10">
                                    {col.coverUrl ? (
                                        <>
                                            <img src={col.coverUrl} className="w-full h-full object-cover transition-transform duration-700 ease-out group-hover:scale-110" />
                                            <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/20 to-transparent opacity-60 group-hover:opacity-80 transition-opacity" />
                                        </>
                                    ) : (
                                        <div className="w-full h-full flex flex-col items-center justify-center gap-3 text-muted-foreground/40 bg-muted/20">
                                            <col.icon className="w-10 h-10 stroke-[1.5]" />
                                            <span className="text-[10px] font-bold uppercase tracking-widest">{col.emptyMsg}</span>
                                        </div>
                                    )}
                                    <div className="absolute bottom-4 left-4">
                                        <div className="p-2 rounded-lg bg-white/10 backdrop-blur-md border border-white/10">
                                            <col.icon className="w-4 h-4 text-white" />
                                        </div>
                                    </div>
                                </div>
                                <div className="font-bold text-[15px] tracking-tight truncate px-1">{col.title}</div>
                                <div className="text-[12px] font-medium text-muted-foreground/70 px-1">{col.count} mục</div>
                            </div>
                        ))}

                        {/* Manual Albums with optional delete selection */}
                        {manualAlbums.map(album => {
                            const isDeleteSelected = deleteSelectedIds.has(album.id);
                            return (
                                <div
                                    key={album.id}
                                    className={`group cursor-pointer relative`}
                                    onClick={() => {
                                        if (deleteMode) {
                                            toggleDeleteSelection(album.id);
                                        } else {
                                            onNavigate?.({ id: album.id, title: album.title });
                                        }
                                    }}
                                >
                                    <div className={`aspect-square rounded-2xl overflow-hidden bg-muted/40 mb-3 transition-all duration-300 shadow-sm relative border ${deleteMode
                                        ? isDeleteSelected
                                            ? 'ring-4 ring-destructive ring-offset-2 ring-offset-background scale-95 border-destructive/30'
                                            : 'ring-4 ring-transparent border-border/10 group-hover:ring-destructive/30'
                                        : 'ring-4 ring-transparent border-border/10 group-hover:ring-primary/20 group-hover:shadow-xl'
                                        }`}>
                                        {album.cover_url ? (
                                            <img
                                                src={localFileUrl(album.cover_url)}
                                                className="w-full h-full object-cover transition-transform duration-700 ease-out group-hover:scale-110"
                                                onError={(e) => { e.currentTarget.style.display = 'none'; }}
                                            />
                                        ) : (
                                            <div className="w-full h-full flex flex-col items-center justify-center gap-3 text-muted-foreground/40 bg-muted/20">
                                                <LibraryBig className="w-10 h-10 stroke-[1.5]" />
                                            </div>
                                        )}

                                        {/* Delete overlay */}
                                        {deleteMode && (
                                            <div className={`absolute inset-0 flex items-center justify-center transition-all duration-200 ${isDeleteSelected ? 'bg-destructive/30' : 'bg-transparent group-hover:bg-destructive/10'}`}>
                                                {isDeleteSelected && (
                                                    <CheckCircle2 className="w-10 h-10 text-destructive drop-shadow-lg" />
                                                )}
                                            </div>
                                        )}
                                    </div>
                                    <div className="font-bold text-[15px] tracking-tight truncate px-1">{album.title}</div>
                                    <div className="text-[12px] font-medium text-muted-foreground/70 px-1">{album.count} mục</div>
                                </div>
                            );
                        })}

                        {/* AI / Smart Albums */}
                        {aiAlbums.map(album => (
                            <div key={album.id} className="group cursor-pointer" onClick={() => !deleteMode && onNavigate?.({ id: album.id, title: album.title })}>
                                <div className="aspect-square rounded-2xl overflow-hidden bg-muted/40 mb-3 transition-all duration-300 ring-4 ring-transparent group-hover:ring-primary/20 shadow-sm group-hover:shadow-xl relative border border-border/10">
                                    <img src={album.coverUrl} className="w-full h-full object-cover transition-transform duration-700 ease-out group-hover:scale-110" />
                                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/20 to-transparent opacity-60 group-hover:opacity-80 transition-opacity" />
                                </div>
                                <div className="font-bold text-[15px] tracking-tight truncate px-1">{album.title}</div>
                                <div className="text-[12px] font-medium text-muted-foreground/70 px-1">{album.count} mục</div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Dialogs */}
            <PromptDialog
                isOpen={isCreateOpen}
                title="Tạo album mới"
                description="Nhập tên cho album mới của bạn:"
                placeholder="VD: Chuyến đi Đà Lạt..."
                confirmText="Tiếp theo"
                cancelText="Hủy"
                isLoading={isCreating}
                onConfirm={handleNameConfirm}
                onCancel={() => setIsCreateOpen(false)}
            />

            <ConfirmDialog
                isOpen={showDeleteConfirm}
                title="Xóa album đã chọn?"
                description={`Bạn sắp xóa ${deleteSelectedIds.size} album. Thao tác này không thể hoàn tác. Ảnh bên trong sẽ không bị xóa.`}
                confirmText="Xóa"
                cancelText="Hủy"
                isDestructive={true}
                isLoading={isDeleting}
                onConfirm={handleConfirmDelete}
                onCancel={() => setShowDeleteConfirm(false)}
            />

            <ConfirmDialog
                isOpen={!!actionError}
                title="Lỗi thao tác"
                description={actionError || ""}
                confirmText="Đã hiểu"
                isDestructive={true}
                type="alert"
                onConfirm={() => setActionError(null)}
                onCancel={() => setActionError(null)}
            />
        </div>
    );
}
