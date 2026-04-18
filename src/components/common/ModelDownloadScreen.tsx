import { useEffect, useRef } from "react";

export interface ModelDownloadEvent {
  file: string;
  progress: number;    // 0.0 – 1.0 for current file
  message: string;
  done: boolean;
  error: string;
  file_index: number;  // 1-based
  file_total: number;
  bytes_done: number;
  bytes_total: number;
}

interface Props {
  event: ModelDownloadEvent | null;
}

function formatBytes(b: number): string {
  if (b <= 0) return "";
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(0)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

export default function ModelDownloadScreen({ event }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Animated particles background
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles: { x: number; y: number; vx: number; vy: number; r: number; alpha: number }[] = [];
    for (let i = 0; i < 60; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        r: Math.random() * 2 + 0.5,
        alpha: Math.random() * 0.4 + 0.1,
      });
    }

    let raf = 0;
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(99,102,241,${p.alpha})`;
        ctx.fill();
      }
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, []);

  const overall =
    event && event.file_total > 0
      ? ((event.file_index - 1) + event.progress) / event.file_total
      : 0;

  const overallPct = Math.round(overall * 100);
  const filePct = event ? Math.round(event.progress * 100) : 0;

  const FILES_LABELS: Record<string, string> = {
    "vision_tower_aura.onnx":              "Vision Tower (Aura)",
    "text_tower_aura.onnx":                "Text Tower (Aura)",
    "face_recognition_sface_2021dec.onnx": "Face Recognition (SFace)",
    "face_detection_yunet_2022mar.onnx":   "Face Detection (YuNet)",
    "yolo26n-seg.onnx":                    "Object Detection (YOLO)",
    "vocab.txt":                           "Tokenizer Vocab",
    "bpe.codes":                           "BPE Codes",
    "DejaVuSans.ttf":                      "UI Font",
  };

  const displayName = event?.file ? (FILES_LABELS[event.file] ?? event.file) : "";

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center overflow-hidden bg-[#0a0a14]">
      {/* Particle canvas */}
      <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none opacity-60" />

      {/* Glowing orb */}
      <div className="absolute w-96 h-96 rounded-full bg-indigo-600/10 blur-[120px] pointer-events-none" />

      {/* Card */}
      <div className="relative z-10 w-[480px] rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl p-8 shadow-2xl">
        {/* Logo area */}
        <div className="flex flex-col items-center mb-8">
          <div className="relative mb-4">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/40">
              <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
              </svg>
            </div>
            {/* Spinning ring */}
            <div className="absolute -inset-1 rounded-[18px] border-2 border-indigo-500/30 border-t-indigo-500 animate-spin" />
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">AuraSeek</h1>
          <p className="text-sm text-white/50 mt-1">Đang chuẩn bị hệ thống lần đầu</p>
        </div>

        {/* Error state */}
        {event?.error && (
          <div className="mb-4 p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
            ❌ {event.error}
          </div>
        )}

        {/* File info */}
        <div className="mb-3">
          <div className="flex justify-between items-center mb-1.5">
            <span className="text-sm text-white/70 font-medium truncate max-w-[320px]">
              {displayName || "Đang chuẩn bị..."}
            </span>
            <span className="text-sm font-mono text-indigo-400 ml-2 shrink-0">{filePct}%</span>
          </div>
          {/* Per-file progress bar */}
          <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-all duration-300"
              style={{ width: `${filePct}%` }}
            />
          </div>
          {event && event.bytes_total > 0 && (
            <p className="text-xs text-white/30 mt-1 text-right">
              {formatBytes(event.bytes_done)} / {formatBytes(event.bytes_total)}
            </p>
          )}
        </div>

        {/* Overall progress */}
        <div className="mb-5">
          <div className="flex justify-between items-center mb-1.5">
            <span className="text-xs text-white/40">
              Tổng tiến trình
              {event && event.file_total > 0 && (
                <span className="ml-1 text-white/25">
                  ({event.file_index}/{event.file_total} tệp)
                </span>
              )}
            </span>
            <span className="text-xs font-mono text-white/50">{overallPct}%</span>
          </div>
          <div className="h-2 rounded-full bg-white/10 overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-indigo-600 via-violet-500 to-purple-500 transition-all duration-500"
              style={{ width: `${overallPct}%` }}
            />
          </div>
        </div>

        {/* Status message */}
        <p className="text-center text-xs text-white/40 leading-relaxed">
          {event?.message || "Đang kết nối..."}
        </p>

        {/* File list (mini progress indicators) */}
        {event && event.file_total > 0 && (
          <div className="mt-5 grid grid-cols-4 gap-1.5">
            {Array.from({ length: event.file_total }).map((_, idx) => {
              const done = idx < event.file_index - 1;
              const active = idx === event.file_index - 1;
              return (
                <div
                  key={idx}
                  className={`h-1 rounded-full transition-all duration-300 ${
                    done    ? "bg-indigo-500"
                    : active ? "bg-indigo-500/60 animate-pulse"
                    : "bg-white/10"
                  }`}
                />
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
