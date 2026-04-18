import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { Settings, Cpu } from "lucide-react";
import { useState, useEffect } from "react";
import { AuraSeekApi } from "@/lib/api";
import { SettingsSourceSection } from "./SettingsSourceSection.tsx";
import { SettingsDatabaseSection } from "./SettingsDatabaseSection.tsx";
import { SettingsErrorAlert } from "./SettingsErrorAlert.tsx";

interface SettingsModalProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    currentSourceDir?: string;
    onSourceDirChange?: (dir: string) => void;
}

export function SettingsModal({ open, onOpenChange, currentSourceDir = "", onSourceDirChange }: SettingsModalProps) {
    const [sourceFolder, setSourceFolder] = useState(currentSourceDir);
    const [saving, setSaving] = useState(false);
    const [saved, setSaved] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [showWarning, setShowWarning] = useState(false);
    
    const [cleaning, setCleaning] = useState(false);
    const [resetting, setResetting] = useState(false);

    useEffect(() => {
        if (open) {
            setSourceFolder(currentSourceDir);
            setSaved(false);
            setError(null);
            setShowWarning(false);
        }
    }, [open, currentSourceDir]);

    const hasChanged = sourceFolder.trim() !== currentSourceDir.trim();

    const handleSave = async () => {
        const trimmed = sourceFolder.trim();
        if (!trimmed) { setError("Vui lòng nhập đường dẫn"); return; }

        if (currentSourceDir && hasChanged && !showWarning) {
            setShowWarning(true);
            return;
        }

        setSaving(true);
        setError(null);
        setShowWarning(false);
        try {
            await AuraSeekApi.setSourceDir(trimmed);
            setSaved(true);
            onSourceDirChange?.(trimmed);
            // Trigger auto-scan with the new folder
            try { await AuraSeekApi.autoScan(); } catch {}
            window.dispatchEvent(new Event("refresh_photos"));
            setTimeout(() => onOpenChange(false), 1200);
        } catch (e) {
            setError(String(e));
        } finally {
            setSaving(false);
        }
    };

    const handleCleanup = async () => {
        setCleaning(true);
        try {
            const removed = await AuraSeekApi.cleanupDatabase();
            setError(null);
            alert(`Đã dọn dẹp ${removed} ảnh không còn tồn tại trên đĩa.`);
            window.dispatchEvent(new Event("refresh_photos"));
        } catch (e) {
            setError(String(e));
        } finally {
            setCleaning(false);
        }
    };

    const handleReset = async () => {
        if (!confirm("Bạn có chắc chắn muốn xóa toàn bộ dữ liệu database? Thao tác này KHÔNG XÓA ảnh trên đĩa của bạn, nhưng sẽ làm mất toàn bộ thông tin nhận diện AI, người, và lịch sử tìm kiếm.")) {
            return;
        }
        setResetting(true);
        try {
            await AuraSeekApi.resetDatabase();
            setSourceFolder("");
            onSourceDirChange?.("");
            setError(null);
            alert("Đã đặt lại database thành công.");
            window.location.reload(); // Hard reload to clear all state
        } catch (e) {
            setError(String(e));
        } finally {
            setResetting(false);
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
                    <SettingsSourceSection
                        sourceFolder={sourceFolder}
                        setSourceFolder={(value: string) => {
                            setSourceFolder(value);
                            setShowWarning(false);
                            setSaved(false);
                        }}
                        saving={saving}
                        saved={saved}
                        showWarning={showWarning}
                        onConfirmWarning={handleSave}
                        onCancelWarning={() => setShowWarning(false)}
                        onSave={handleSave}
                        hasExistingSource={!!currentSourceDir}
                        hasChanged={hasChanged}
                        setShowWarning={setShowWarning}
                        setError={setError}
                    />

                    <SettingsDatabaseSection
                        cleaning={cleaning}
                        resetting={resetting}
                        onCleanup={handleCleanup}
                        onReset={handleReset}
                    />

                    <SettingsErrorAlert error={error} />

                    <div className="pt-4 border-t border-border/10 flex flex-col items-center justify-center text-center space-y-3">
                        <div className="w-12 h-12 bg-linear-to-tr from-primary to-indigo-500 rounded-2xl flex items-center justify-center shadow-lg shadow-primary/20">
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
