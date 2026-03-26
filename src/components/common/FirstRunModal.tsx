import { useState } from "react";
import { Folder, Check } from "lucide-react";
import { AuraSeekApi } from "@/lib/api";

interface FirstRunModalProps {
  onComplete: (dir: string) => void;
}

export function FirstRunModal({ onComplete }: FirstRunModalProps) {
  const [dir, setDir] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const isWindows = typeof navigator !== 'undefined' && navigator.userAgent.toLowerCase().includes("win");
  const placeholderTxt = isWindows ? "C:\\Users\\Admin\\Pictures" : "/home/user/Pictures";

  const handleConfirm = async () => {
    const trimmed = dir.trim();
    if (!trimmed) {
      setError("Vui lòng nhập đường dẫn thư mục");
      return;
    }
    setSaving(true);
    setError("");
    try {
      await AuraSeekApi.setSourceDir(trimmed);
      onComplete(trimmed);
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="w-full max-w-md rounded-2xl bg-[#1a1a2e] border border-white/10 shadow-2xl p-8 flex flex-col gap-6">
        <div className="text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-indigo-500/20 border border-indigo-500/30">
            <Folder className="w-8 h-8 text-indigo-400" />
          </div>
          <h2 className="text-xl font-semibold text-white">Chào mừng đến AuraSeek</h2>
          <p className="mt-2 text-sm text-white/60">
            Chọn thư mục chứa ảnh của bạn để bắt đầu. AuraSeek sẽ tự động phân tích và lập chỉ mục ảnh.
          </p>
        </div>

        <div className="flex flex-col gap-2">
          <label className="text-xs font-medium text-white/70 uppercase tracking-wider">
            Đường dẫn thư mục nguồn ảnh
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={dir}
              onChange={(e) => setDir(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleConfirm()}
              placeholder={placeholderTxt}
              className="flex-1 rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/30"
            />
          </div>
          {error && <p className="text-xs text-red-400">{error}</p>}
        </div>

        <button
          onClick={handleConfirm}
          disabled={saving || !dir.trim()}
          className="flex items-center justify-center gap-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed px-4 py-2.5 text-sm font-medium text-white transition-colors"
        >
          {saving ? (
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
          ) : (
            <Check className="w-4 h-4" />
          )}
          {saving ? "Đang lưu..." : "Đồng ý & Bắt đầu"}
        </button>
      </div>
    </div>
  );
}
