import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

type GlassCardProps = {
  children: ReactNode;
  className?: string;
};

export function GlassCard({ children, className }: GlassCardProps) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-xl border border-white/10 bg-linear-to-br from-white/5 to-white/0 p-3 shadow-[0_18px_60px_rgba(15,23,42,0.45)] backdrop-blur-2xl dark:border-white/10 dark:from-white/10 dark:to-white/0",
        "transition-shadow duration-300 ease-out",
        "hover:shadow-[0_28px_80px_rgba(15,23,42,0.9)] hover:-translate-y-0.5",
        className,
      )}
    >
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(248,250,252,0.18),transparent_55%),radial-gradient(circle_at_bottom_right,rgba(15,23,42,0.85),transparent_55%)]" />
      <div className="relative z-10">{children}</div>
    </div>
  );
}

