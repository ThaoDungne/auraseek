import type { Photo } from "@/types/photo.type";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Calendar, Smartphone, HardDrive, Tag, Plus, UserPlus, Image as ImageIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { AuraSeekApi } from "@/lib/api";

export function PhotoInfoPanel({ photo }: { photo: Photo }) {
  const [description, setDescription] = useState("");
  const [deviceName, setDeviceName] = useState<string | null>(null);
  const [effectiveSizeBytes, setEffectiveSizeBytes] = useState<number>(photo.sizeBytes || 0);

  const takenDate = new Date(photo.takenAt);
  const formattedDate = new Intl.DateTimeFormat("vi-VN", { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' }).format(takenDate);
  const formattedTime = new Intl.DateTimeFormat("vi-VN", { hour: '2-digit', minute: '2-digit' }).format(takenDate);

  useEffect(() => {
    let cancelled = false;

    // Cập nhật size từ prop trước
    setEffectiveSizeBytes(photo.sizeBytes || 0);

    // Lấy tên thiết bị hiện tại (hostname / tên máy)
    AuraSeekApi.getDeviceName()
      .then((name) => {
        if (!cancelled) setDeviceName(name);
      })
      .catch(() => {
        if (!cancelled) setDeviceName(null);
      });

    // Nếu sizeBytes đang là 0 nhưng có filePath, hỏi backend để lấy dung lượng thật
    if (!photo.sizeBytes && photo.filePath) {
      AuraSeekApi.getFileSize(photo.filePath)
        .then((size) => {
          if (!cancelled) setEffectiveSizeBytes(size);
        })
        .catch(() => {
          // ignore – giữ 0 nếu không đọc được
        });
    }

    return () => {
      cancelled = true;
    };
  }, [photo.id, photo.sizeBytes, photo.filePath]);

  const sizeInMb = effectiveSizeBytes > 0 ? (effectiveSizeBytes / 1048576).toFixed(2) : "0.00";

  return (
    <ScrollArea className="h-full w-full">
      <div className="flex flex-col gap-6 p-4 pb-20">

        {/* Description Input */}
        <div className="group">
          <input
            type="text"
            placeholder="Thêm mô tả..."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="w-full bg-transparent hover:bg-muted/50 focus:bg-muted/50 focus:outline-none rounded-md px-3 py-2 text-sm transition-colors placeholder:text-muted-foreground/70"
          />
        </div>

        {/* Date & Time Section */}
        <div className="flex gap-4 items-start px-3 group">
          <Calendar className="w-5 h-5 text-muted-foreground shrink-0 mt-0.5" />
          <div className="flex-1 cursor-pointer rounded-md hover:bg-muted/40 -m-1.5 p-1.5 transition-colors">
            <div className="text-sm">{formattedDate}</div>
            <div className="text-xs text-muted-foreground mt-0.5">{formattedTime} • GMT+07:00</div>
          </div>
        </div>

        {/* Device Info */}
        <div className="flex gap-4 items-start px-3">
          <Smartphone className="w-5 h-5 text-muted-foreground shrink-0 mt-0.5" />
          <div className="flex-1">
            <div className="text-sm font-medium">
              {deviceName || photo.cameraModel || "Thiết bị này"}
            </div>
            {photo.iso && <div className="text-xs text-muted-foreground mt-0.5">ƒ/1.8 • 1/120 • {photo.focalLength}mm • ISO {photo.iso}</div>}
            <div className="text-xs text-muted-foreground mt-0.5">
              {photo.width} × {photo.height} • {sizeInMb} MB
            </div>
          </div>
        </div>

        {/* People / Faces */}
        <div className="flex gap-4 items-start px-3">
          <UserPlus className="w-5 h-5 text-muted-foreground shrink-0 mt-0.5" />
          <div className="flex-1 flex flex-col gap-2">
            <div className="text-sm font-medium">Người trong ảnh</div>
            <div className="flex flex-wrap gap-2 mt-1">
              {photo.people?.map(p => (
                <div key={p.id} className="flex items-center gap-2 bg-muted/40 hover:bg-muted/80 cursor-pointer rounded-full pr-3 transition-colors border border-transparent hover:border-border/30">
                  <div className="w-7 h-7 rounded-full bg-primary/20 flex items-center justify-center text-xs font-medium text-primary shrink-0">
                    {p.name.charAt(0)}
                  </div>
                  <span className="text-xs">{p.name}</span>
                </div>
              ))}
              <Button variant="ghost" size="icon" className="w-7 h-7 rounded-full bg-muted/40 hover:bg-muted/80">
                <Plus className="w-4 h-4 text-muted-foreground" />
              </Button>
            </div>
          </div>
        </div>

        {/* Labels / Tags */}
        <div className="flex gap-4 items-start px-3">
          <Tag className="w-5 h-5 text-muted-foreground shrink-0 mt-0.5" />
          <div className="flex-1 flex flex-col gap-2">
            <div className="text-sm font-medium">Nhãn đối tượng (AI)</div>
            <div className="flex flex-wrap gap-1.5 mt-1">
              {photo.labels?.map(l => (
                <span key={l} className="bg-muted px-2.5 py-1 rounded-md text-xs hover:bg-muted/80 cursor-pointer border border-transparent hover:border-border/30 transition-colors">
                  {l}
                </span>
              ))}
              <span className="text-xs text-primary cursor-pointer hover:underline py-1 ml-1">Thêm nhãn</span>
            </div>
          </div>
        </div>

        {/* Albums */}
        <div className="flex gap-4 items-start px-3">
          <ImageIcon className="w-5 h-5 text-muted-foreground shrink-0 mt-0.5" />
          <div className="flex-1">
            <div className="text-sm text-primary cursor-pointer hover:underline">Thêm vào album</div>
          </div>
        </div>

        {/* Storage Details */}
        <div className="flex gap-4 items-start px-3 pt-4 border-t border-border/10">
          <HardDrive className="w-5 h-5 text-muted-foreground shrink-0 mt-0.5" />
          <div className="flex-1">
            <div className="text-sm">Đã sao lưu ở chất lượng gốc</div>
            <div className="text-xs text-muted-foreground mt-0.5">
              {photo.type === "video" ? "Video" : "Ảnh"} này chiếm {sizeInMb} MB dung lượng bộ nhớ
            </div>
            {photo.filePath && (
              <div className="text-xs text-muted-foreground mt-1 break-all font-mono">Path: {photo.filePath}</div>
            )}
          </div>
        </div>

      </div>
    </ScrollArea>
  );
}
