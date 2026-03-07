import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { Settings, HardDrive, Cpu, FolderPlus, CheckCircle2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { AuraSeekApi, type IngestSummary } from "@/lib/api";

interface SettingsModalProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function SettingsModal({ open, onOpenChange }: SettingsModalProps) {
    const [isScanning, setIsScanning] = useState(false);
    const [sourceFolder, setSourceFolder] = useState<string>(() =>
        localStorage.getItem("auraseek_source") || ""
    );
    const [scanResult, setScanResult] = useState<IngestSummary | null>(null);
    const [scanError, setScanError] = useState<string | null>(null);

    const handleScan = async () => {
        if (!sourceFolder.trim()) {
            setScanError("Vui lòng nhập đường dẫn thư mục nguồn");
            return;
        }
        setIsScanning(true);
        setScanResult(null);
        setScanError(null);
        localStorage.setItem("auraseek_source", sourceFolder);

        try {
            await AuraSeekApi.init();
            const result = await AuraSeekApi.scanFolder(sourceFolder);
            setScanResult(result);
            window.dispatchEvent(new Event("refresh_photos"));
        } catch (e: any) {
            setScanError(String(e));
        } finally {
            setIsScanning(false);
        }
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[520px] p-0 overflow-hidden bg-background/95 backdrop-blur-xl shadow-2xl border-border/40 sm:rounded-2xl">
                <DialogHeader className="px-6 py-5 border-b border-border/10 bg-muted/30">
                    <DialogTitle className="flex items-center gap-2 text-xl font-medium tracking-tight text-foreground">
                        <Settings className="w-5 h-5 text-primary" />
                        Cài đặt
                    </DialogTitle>
                    <DialogDescription className="mt-1.5 leading-relaxed">
                        Quản lý thư viện ảnh AuraSeek
                    </DialogDescription>
                </DialogHeader>

                <div className="px-6 py-6 flex flex-col gap-6 overflow-y-auto max-h-[70vh]">

                    {/* Source Folder */}
                    <div className="space-y-3">
                        <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
                            <HardDrive className="w-4 h-4 text-primary" />
                            Thư mục nguồn ảnh
                        </h3>
                        <p className="text-sm text-muted-foreground leading-relaxed">
                            Nhập đường dẫn thư mục chứa ảnh/video. AI sẽ tự động quét, phát hiện đối tượng, nhận diện khuôn mặt và tạo embedding.
                        </p>
                        <input
                            type="text"
                            value={sourceFolder}
                            onChange={e => setSourceFolder(e.target.value)}
                            placeholder="/home/user/Pictures"
                            className="flex h-10 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm font-mono"
                        />

                        <Button
                            onClick={handleScan}
                            disabled={isScanning}
                            className="w-full h-12 mt-1 rounded-xl font-medium transition-all shadow-sm"
                        >
                            {isScanning ? (
                                <>
                                    <div className="w-4 h-4 mr-2 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    Đang quét và xử lý AI...
                                </>
                            ) : (
                                <>
                                    <FolderPlus className="w-4 h-4 mr-2" />
                                    Khởi chạy bộ quét AI
                                </>
                            )}
                        </Button>
                    </div>

                    {/* Scan Result */}
                    {scanResult && (
                        <div className="rounded-xl border border-emerald-500/20 bg-emerald-50 dark:bg-emerald-950/20 p-4 space-y-2">
                            <div className="flex items-center gap-2 text-emerald-700 dark:text-emerald-400 font-medium text-sm">
                                <CheckCircle2 className="w-4 h-4" />
                                Quét hoàn thành
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                                <span>Tổng tìm thấy: <strong className="text-foreground">{scanResult.total_found}</strong></span>
                                <span>Mới thêm: <strong className="text-emerald-600 dark:text-emerald-400">{scanResult.newly_added}</strong></span>
                                <span>Bỏ qua trùng: <strong className="text-foreground">{scanResult.skipped_dup}</strong></span>
                                <span>Lỗi: <strong className="text-destructive">{scanResult.errors}</strong></span>
                            </div>
                        </div>
                    )}

                    {/* Error */}
                    {scanError && (
                        <div className="rounded-xl border border-destructive/20 bg-destructive/5 p-4 flex gap-2 text-sm text-destructive">
                            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                            {scanError}
                        </div>
                    )}

                    {/* About */}
                    <div className="pt-4 border-t border-border/10 flex flex-col items-center justify-center text-center space-y-3">
                        <div className="w-12 h-12 bg-gradient-to-tr from-primary to-indigo-500 rounded-2xl flex items-center justify-center shadow-lg shadow-primary/20">
                            <Cpu className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <h3 className="text-base font-medium tracking-tight text-foreground">AuraSeek</h3>
                            <p className="text-xs text-muted-foreground mt-0.5">Phiên bản 1.0.0 · SurrealDB + Local AI · Offline</p>
                        </div>
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
}
