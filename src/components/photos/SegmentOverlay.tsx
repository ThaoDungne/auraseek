/**
 * SegmentOverlay — canvas-based segmentation mask + bbox renderer.
 *
 * Decodes mask_rle (run-length encoded) data for each detected object and
 * paints a semi-transparent coloured fill + crisp border on a canvas that
 * sits on top of an <img> rendered with object-cover or object-contain.
 *
 * Works for both object detection (with mask_rle) and face detection
 * (bbox-only fill, since YuNet produces no segmentation masks).
 */

import { useRef, useEffect } from "react";
import type { DetectedObject, DetectedFace } from "@/types/photo.type";

export interface SegmentOverlayProps {
  /** Detected objects — may carry `mask_rle` for pixel-accurate fills */
  detectedObjects?: DetectedObject[];
  /** Detected faces — highlighted using bbox fill (no mask) */
  detectedFaces?: DetectedFace[];
  /** Original image width in pixels (used for RLE decoding) */
  imgNaturalW: number;
  /** Original image height in pixels */
  imgNaturalH: number;
  /** Rendered display width (clientWidth of the <img>) */
  displayW: number;
  /** Rendered display height (clientHeight of the <img>) */
  displayH: number;
  /** CSS object-fit mode — determines coordinate mapping */
  objectFit?: "cover" | "contain";
  /** Whether to draw face bbox rectangles (default: true) */
  showFaces?: boolean;
  /** Whether to draw object/face labels + confidence (default: true) */
  showLabels?: boolean;
  /** Current zoom/view scale — use this to adjust stroke/font size to stay crisp */
  viewScale?: number;
  /** Whether to render segmentation masks (default: true) */
  showMasks?: boolean;
  /** Whether to render bbox rectangles (objects + faces) (default: true) */
  showBoxes?: boolean;
  /** If set, only this object index will be highlighted (mask + label) */
  activeObjectIndex?: number | null;
  /** If true, chỉ vẽ mask cho activeObjectIndex (khi chưa có index thì không vẽ gì) */
  onlyActive?: boolean;
}

// Decode [offset, length][] RLE into a Uint8Array of 0/1 flags
function decodeRle(rle: [number, number][], total: number): Uint8Array {
  const buf = new Uint8Array(total);
  for (const [off, len] of rle) {
    buf.fill(1, off, Math.min(off + len, total));
  }
  return buf;
}

// Fill + border RGBA pairs for up to 6 objects
const PALETTE: [number, number, number][] = [
  [34,  211, 238], // cyan
  [251, 191,  36], // amber
  [ 74, 222, 128], // green
  [248, 113, 113], // red
  [192, 132, 252], // purple
  [251, 146,  60], // orange
];
const FACE_RGB: [number, number, number] = [167, 139, 250]; // violet

export function SegmentOverlay({
  detectedObjects = [],
  detectedFaces   = [],
  imgNaturalW,
  imgNaturalH,
  displayW,
  displayH,
  objectFit  = "cover",
  showFaces  = true,
  showLabels = true,
  viewScale  = 1,
  showMasks  = true,
  showBoxes  = true,
  activeObjectIndex = null,
  onlyActive = false,
}: SegmentOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || displayW === 0 || displayH === 0 || imgNaturalW === 0 || imgNaturalH === 0) return;

    canvas.width  = displayW;
    canvas.height = displayH;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, displayW, displayH);

    // ── Coordinate transform ─────────────────────────────────────────
    let scaleX: number, scaleY: number, cropX: number, cropY: number;

    if (objectFit === "cover") {
      const s = Math.max(displayW / imgNaturalW, displayH / imgNaturalH);
      scaleX  = scaleY = s;
      cropX   = (imgNaturalW * s - displayW) / 2;
      cropY   = (imgNaturalH * s - displayH) / 2;
    } else {
      // contain: letterboxed
      const s = Math.min(displayW / imgNaturalW, displayH / imgNaturalH);
      scaleX  = scaleY = s;
      cropX   = -(displayW - imgNaturalW * s) / 2;  // negative = padding
      cropY   = -(displayH - imgNaturalH * s) / 2;
    }

    // Map original-image pixel (ox, oy) → display pixel (px, py)
    // px = ox * scaleX - cropX
    // py = oy * scaleY - cropY
    // Inverse: ox = (px + cropX) / scaleX

    const totalPixels = imgNaturalW * imgNaturalH;

    // ── 1. Decode all object masks ────────────────────────────────────
    const masks = detectedObjects.map((obj, i) => ({
      index: i,
      obj,
      rgb: PALETTE[i % PALETTE.length],
      pixels: obj.mask_rle?.length ? decodeRle(obj.mask_rle, totalPixels) : null,
    }));

    const hasActive = typeof activeObjectIndex === "number";
    const masksToRender = onlyActive
      ? (hasActive ? masks.filter(m => m.index === activeObjectIndex) : [])
      : (hasActive ? masks.filter(m => m.index === activeObjectIndex) : masks);

    // ── 2. Fill pixels (single ImageData pass over the display canvas) ─
    if (showMasks && masksToRender.some(m => m.pixels)) {
      const imageData = ctx.createImageData(displayW, displayH);
      const d = imageData.data;

      for (let py = 0; py < displayH; py++) {
        for (let px = 0; px < displayW; px++) {
          const ox = Math.floor((px + cropX) / scaleX);
          const oy = Math.floor((py + cropY) / scaleY);
          if (ox < 0 || ox >= imgNaturalW || oy < 0 || oy >= imgNaturalH) continue;

          const midx = oy * imgNaturalW + ox;
          for (const { pixels, rgb } of masksToRender) {
            if (pixels?.[midx]) {
              const b = (py * displayW + px) * 4;
              d[b]     = rgb[0];
              d[b + 1] = rgb[1];
              d[b + 2] = rgb[2];
              d[b + 3] = 110; // ~43% opacity fill
              break;
            }
          }
        }
      }
      ctx.putImageData(imageData, 0, 0);
    }

    // ── 3. Draw crisp mask borders ────────────────────────────────────
    if (showMasks) {
      for (const { pixels, rgb } of masksToRender) {
        if (!pixels) continue;
        ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.92)`;

        for (let py = 1; py < displayH - 1; py++) {
          for (let px = 1; px < displayW - 1; px++) {
            const ox = Math.floor((px + cropX) / scaleX);
            const oy = Math.floor((py + cropY) / scaleY);
            if (ox < 0 || ox >= imgNaturalW || oy < 0 || oy >= imgNaturalH) continue;
            if (!pixels[oy * imgNaturalW + ox]) continue;

            const isEdge =
              !pixels[oy * imgNaturalW + Math.max(ox - 1, 0)] ||
              !pixels[oy * imgNaturalW + Math.min(ox + 1, imgNaturalW - 1)] ||
              !pixels[Math.max(oy - 1, 0) * imgNaturalW + ox] ||
              !pixels[Math.min(oy + 1, imgNaturalH - 1) * imgNaturalW + ox];

            if (isEdge) {
              const size = Math.max(1, 2 / viewScale);
              ctx.fillRect(px - size / 2, py - size / 2, size, size);
            }
          }
        }
      }
    }

    // ── 3b. Object bbox rectangles (for all objects) ──────────────────
    if (showBoxes) {
      masks.forEach(({ obj, rgb }) => {
        const x = obj.bbox.x * scaleX - cropX;
        const y = obj.bbox.y * scaleY - cropY;
        const w = obj.bbox.w * scaleX;
        const h = obj.bbox.h * scaleY;

        ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.95)`;
        ctx.lineWidth   = Math.max(1, 2 / viewScale);
        ctx.strokeRect(x, y, w, h);
      });
    }

    // ── 4. Face bbox fills (violet) ───────────────────────────────────
    if (showFaces && showBoxes) {
      for (const face of detectedFaces) {
        const x = face.bbox.x * scaleX - cropX;
        const y = face.bbox.y * scaleY - cropY;
        const w = face.bbox.w * scaleX;
        const h = face.bbox.h * scaleY;

        ctx.fillStyle   = `rgba(${FACE_RGB[0]},${FACE_RGB[1]},${FACE_RGB[2]},0.35)`;
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = `rgba(${FACE_RGB[0]},${FACE_RGB[1]},${FACE_RGB[2]},0.9)`;
        ctx.lineWidth   = 2;
        ctx.strokeRect(x, y, w, h);
      }
    }

    // ── 5. Labels ─────────────────────────────────────────────────────
    if (showLabels) {
      const label = (text: string, bx: number, by: number, rgb: [number,number,number]) => {
        const fontSize = Math.max(6, 11 / viewScale);
        ctx.font = `bold ${fontSize}px system-ui, sans-serif`;
        const tw  = ctx.measureText(text).width;
        const pad = 4 / viewScale;
        const lh  = 17 / viewScale;
        const lx  = Math.max(0, Math.min(bx, displayW - tw - pad * 2));
        const ly  = by > lh + 2 / viewScale ? by - lh - 2 / viewScale : by + 2 / viewScale;
        ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.88)`;
        ctx.fillRect(lx, ly, tw + pad * 2, lh);
        ctx.fillStyle = "#fff";
        ctx.fillText(text, lx + pad, ly + lh - 4 / viewScale);
      };

      const labelMasks = activeObjectIndex != null ? masksToRender : masks;

      labelMasks.forEach(({ obj, rgb, pixels }) => {
        if (showMasks && !pixels) return;
        label(
          `${obj.class_name} ${(obj.conf * 100).toFixed(0)}%`,
          obj.bbox.x * scaleX - cropX,
          obj.bbox.y * scaleY - cropY,
          rgb,
        );
      });

      if (showFaces) {
        detectedFaces.forEach(face => {
          label(
            face.name ?? "Face",
            face.bbox.x * scaleX - cropX,
            face.bbox.y * scaleY - cropY,
            FACE_RGB,
          );
        });
      }
    }

  }, [
    detectedObjects,
    detectedFaces,
    imgNaturalW,
    imgNaturalH,
    displayW,
    displayH,
    objectFit,
    showFaces,
    showLabels,
    showMasks,
    showBoxes,
    viewScale,
    activeObjectIndex,
  ]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none"
      style={{ width: "100%", height: "100%" }}
    />
  );
}
