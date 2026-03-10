import { Plus, FolderHeart, Smartphone } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Photo } from "@/types/photo.type";

export function AlbumsView({ photos = [], onNavigate }: { photos?: Photo[], onNavigate?: (payload: any) => void }) {

    // Filter photos for collections
    const favPhotos = photos.filter(p => p.favorite);

    // Robust screenshot detection logic
    const isScreenshot = (p: Photo) => {
        if (!p.filePath) return false;
        const path = p.filePath.toLowerCase();
        const name = path.split(/[/\\]/).pop() || "";

        return path.includes("screenshot") ||
            path.includes("screen-capture") ||
            path.includes("screencast") ||
            path.includes("ảnh chụp màn hình") ||
            path.includes("screenshots") ||
            name.startsWith("scr_") ||
            name.includes("screen_shot") ||
            // Some systems use specific labels if detected
            p.labels?.includes("cell phone") ||
            p.labels?.includes("laptop") ||
            p.labels?.includes("monitor");
    };

    const scrPhotos = photos.filter(isScreenshot);

    const collections = [
        {
            id: "fav",
            title: "Yêu thích",
            count: favPhotos.length,
            icon: FolderHeart,
            coverUrl: favPhotos[0]?.thumbnailUrl || favPhotos[0]?.url || null,
            emptyMsg: "Chưa có ảnh yêu thích nào"
        },
        {
            id: "scr",
            title: "Ảnh chụp màn hình",
            count: scrPhotos.length,
            icon: Smartphone,
            coverUrl: scrPhotos[0]?.thumbnailUrl || scrPhotos[0]?.url || null,
            emptyMsg: "Chưa có ảnh chụp màn hình nào"
        },
    ];

    // Compute Custom Albums / Smart Albums based on YOLO AI Tags
    const albumsMap = new Map<string, { id: string; title: string; count: number; coverUrl: string; }>();

    // Map common english labels to Vietnamese for nice display
    const titleMap: Record<string, string> = {
        person: "Con người", dog: "Chó", cat: "Mèo", car: "Ô tô",
        keyboard: "Bàn phím", laptop: "Máy tính", cell_phone: "Điện thoại",
        mouse: "Chuột", cup: "Cốc cà phê", bottle: "Chai nước", book: "Sách",
        motorcycle: "Xe máy", airplane: "Máy bay", bus: "Xe buýt", truck: "Xe tải",
        bird: "Chim", horse: "Ngựa", sheep: "Cừu", cow: "Bò", elephant: "Voi",
        bear: "Gấu", zebra: "Ngựa vằn", giraffe: "Hươu cao cổ", backpack: "Balo",
        umbrella: "Cái ô", handbag: "Túi xách", tie: "Cà vạt", suitcase: "Vali",
        frisbee: "Đĩa bay", skis: "Ván trượt", snowboard: "Ván trượt tuyết", sports_ball: "Quả bóng",
        kite: "Cái diều", baseball_bat: "Gậy bóng chày", baseball_glove: "Găng tay bóng chày",
        skateboard: "Trượt ván", surfboard: "Ván lướt sóng", tennis_racket: "Vợt tennis",
        bottle_water: "Chai nước", wine_glass: "Ly rượu", cup_coffee: "Tách cà phê", fork: "Cái nĩa",
        knife: "Cái dao", spoon: "Cái thìa", bowl: "Cái bát", banana: "Quả chuối",
        apple: "Quả táo", sandwich: "Bánh mì kẹp", orange: "Quả cam", broccoli: "Súp lơ",
        carrot: "Củ cà rốt", hot_dog: "Xúc xích", pizza: "Bánh pizza", donut: "Bánh donut",
        cake: "Bánh ngọt", chair: "Cái ghế", couch: "Ghế sofa", potted_plant: "Chậu cây",
        bed: "Cái giường", dining_table: "Bàn ăn", toilet: "Bồn cầu", tv: "Tivi",
        remote: "Điều khiển", microwave: "Lò vi sóng", oven: "Lò nướng", toaster: "Máy nướng bánh",
        sink: "Bồn rửa", refrigerator: "Tủ lạnh", clock: "Cái đồng hồ", vase: "Cái bình",
        scissors: "Cái kéo", teddy_bear: "Gấu bông", hair_drier: "Máy sấy tóc", toothbrush: "Bàn chải",
    };

    for (const p of photos) {
        if (!p.labels) continue;
        for (const label of p.labels) {
            const normalizedTag = label.toLowerCase();
            const title = titleMap[normalizedTag] || label;

            if (!albumsMap.has(normalizedTag)) {
                albumsMap.set(normalizedTag, {
                    id: "tag_" + normalizedTag,
                    title,
                    count: 0,
                    coverUrl: p.thumbnailUrl || p.url
                });
            }
            albumsMap.get(normalizedTag)!.count++;
        }
    }

    const customAlbums = Array.from(albumsMap.values())
        .sort((a, b) => b.count - a.count);

    return (
        <div className="flex-1 overflow-y-auto px-6 py-8 will-change-scroll">
            <div className="max-w-7xl mx-auto space-y-12">

                {/* Header */}
                <div className="flex items-center justify-between">
                    <h1 className="text-2xl font-medium tracking-tight">Bộ sưu tập</h1>
                </div>

                {/* Default Collections */}
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
                    {collections.map(col => (
                        <div key={col.id} className="group cursor-pointer" onClick={() => onNavigate?.({ id: col.id, title: col.title })}>
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
                                <div className="absolute bottom-4 left-4 flex items-center gap-2">
                                    <div className="p-2 rounded-lg bg-white/10 backdrop-blur-md border border-white/10">
                                        <col.icon className="w-4 h-4 text-white" />
                                    </div>
                                </div>
                            </div>
                            <div className="font-bold text-[15px] tracking-tight truncate px-1">{col.title}</div>
                            <div className="text-[12px] font-medium text-muted-foreground/70 px-1">{col.count} mục</div>
                        </div>
                    ))}
                </div>

                {/* Custom Albums */}
                <div>
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-lg font-medium">Album của bạn</h2>
                        <Button variant="outline" size="sm" className="rounded-full shadow-sm text-xs font-medium h-9">
                            <Plus className="w-4 h-4 mr-1" />
                            Tạo album mới
                        </Button>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-x-4 gap-y-8">
                        {customAlbums.map(album => (
                            <div key={album.id} className="group cursor-pointer" onClick={() => onNavigate?.({ id: album.id, title: album.title })}>
                                <div className="aspect-[4/3] rounded-xl overflow-hidden bg-muted mb-3 transition-all duration-300 ring-2 ring-transparent group-hover:ring-primary shadow-sm group-hover:shadow-md">
                                    <img src={album.coverUrl} className="w-full h-full object-cover transition-transform duration-500 ease-out group-hover:scale-105" />
                                </div>
                                <div className="font-medium text-sm truncate px-1">{album.title}</div>
                                <div className="text-xs text-muted-foreground px-1">{album.count} mục</div>
                            </div>
                        ))}
                    </div>
                </div>

            </div>
        </div>
    );
}
