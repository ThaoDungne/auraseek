import { HardDrive, AlertTriangle, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";

interface SettingsSourceSectionProps {
    sourceFolder: string;
    setSourceFolder: (value: string) => void;
    saving: boolean;
    saved: boolean;
    showWarning: boolean;
    onConfirmWarning: () => void;
    onCancelWarning: () => void;
    onSave: () => void;
    hasExistingSource: boolean;
    hasChanged: boolean;
    setShowWarning: (value: boolean) => void;
    setError: (value: string | null) => void;
}

export function SettingsSourceSection({
    sourceFolder,
    setSourceFolder,
    saving,
    saved,
    showWarning,
    onConfirmWarning,
    onCancelWarning,
    onSave,
    hasExistingSource,
    hasChanged,
    setShowWarning,
    setError,
}: SettingsSourceSectionProps) {
    const isWindows = typeof navigator !== 'undefined' && navigator.userAgent.toLowerCase().includes("win");
    const placeholderTxt = isWindows ? "C:\\Users\\Admin\\Pictures" : "/home/user/Pictures";

    const handleInputChange = (value: string) => {
        setSourceFolder(value);
        setShowWarning(false);
        setError(null);
    };

    const handlePrimarySave = () => {
        const trimmed = sourceFolder.trim();
        if (!trimmed) {
            setError("Vui lòng nhập đường dẫn");
            return;
        }
        if (hasExistingSource && hasChanged && !showWarning) {
            setShowWarning(true);
            return;
        }
        onSave();
    };

    return (
        <div className="space-y-3">
            <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
                <HardDrive className="w-4 h-4 text-primary" />
                Thư mục nguồn ảnh
            </h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
                Thư mục chứa ảnh sẽ được quét, phân tích AI và đồng bộ tự động khi mở ứng dụng.
            </p>
            <input
                type="text"
                value={sourceFolder}
                onChange={(e) => handleInputChange(e.target.value)}
                placeholder={placeholderTxt}
                className="flex h-10 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm font-mono"
            />

            {showWarning && (
                <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 flex gap-2 text-sm text-amber-700 dark:text-amber-400">
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                    <div>
                        <strong>Lưu ý:</strong> Bạn phải di chuyển các ảnh cũ vào thư mục mới nếu muốn thay đổi thư mục nguồn ảnh để đảm bảo trải nghiệm tốt hơn.
                        <div className="mt-2 flex gap-2">
                            <button
                                onClick={onConfirmWarning}
                                className="rounded-md bg-amber-600 hover:bg-amber-500 px-3 py-1 text-xs text-white font-medium"
                            >
                                Tôi hiểu, xác nhận
                            </button>
                            <button
                                onClick={onCancelWarning}
                                className="rounded-md bg-white/10 hover:bg-white/20 px-3 py-1 text-xs font-medium"
                            >
                                Hủy
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {!showWarning && (
                <Button
                    onClick={handlePrimarySave}
                    disabled={saving || !sourceFolder.trim()}
                    className="w-full h-10 mt-1 rounded-xl font-medium transition-all shadow-sm"
                >
                    {saving ? (
                        <>
                            <div className="w-4 h-4 mr-2 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            Đang lưu...
                        </>
                    ) : saved ? (
                        <>
                            <CheckCircle2 className="w-4 h-4 mr-2" />
                            Đã lưu
                        </>
                    ) : (
                        "Lưu thư mục nguồn"
                    )}
                </Button>
            )}
        </div>
    );
}

