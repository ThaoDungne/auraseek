import { Database, RotateCcw, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";

interface SettingsDatabaseSectionProps {
    cleaning: boolean;
    resetting: boolean;
    onCleanup: () => void;
    onReset: () => void;
}

export function SettingsDatabaseSection({
    cleaning,
    resetting,
    onCleanup,
    onReset,
}: SettingsDatabaseSectionProps) {
    return (
        <div className="space-y-3 pt-6 border-t border-border/10">
            <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
                <Database className="w-4 h-4 text-primary" />
                Quản lý dữ liệu
            </h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
                Xử lý các vấn đề về dữ liệu nếu bạn di chuyển thư mục hoặc muốn quét lại từ đầu.
            </p>

            <div className="grid grid-cols-2 gap-3">
                <Button
                    variant="outline"
                    onClick={onCleanup}
                    disabled={cleaning || resetting}
                    className="h-10 rounded-xl font-medium border-border/40 hover:bg-muted"
                >
                    {cleaning ? (
                        "Đang dọn..."
                    ) : (
                        <>
                            <RotateCcw className="w-4 h-4 mr-2" />
                            Dọn dẹp link hỏng
                        </>
                    )}
                </Button>
                <Button
                    variant="outline"
                    onClick={onReset}
                    disabled={cleaning || resetting}
                    className="h-10 rounded-xl font-medium border-destructive/20 text-destructive hover:bg-destructive/10 hover:text-destructive"
                >
                    {resetting ? (
                        "Đang xóa..."
                    ) : (
                        <>
                            <Trash2 className="w-4 h-4 mr-2" />
                            Đặt lại database
                        </>
                    )}
                </Button>
            </div>
        </div>
    );
}

