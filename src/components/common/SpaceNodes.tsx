import React, { useEffect, useRef } from "react";

export function SpaceNodes() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let nodes: { x: number; y: number; vx: number; vy: number; radius: number }[] = [];
    const numNodes = 70; // Tuỳ chỉnh mật độ
    const connectionDistance = 140; // Khoảng cách nối dây
    const nodeSpeed = 0.4; // Tốc độ trôi

    const resize = () => {
      const parent = canvas.parentElement;
      canvas.width = parent?.clientWidth || window.innerWidth;
      canvas.height = parent?.clientHeight || 280;
      initNodes();
    };

    const initNodes = () => {
      nodes = [];
      for (let i = 0; i < numNodes; i++) {
        nodes.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * nodeSpeed,
          vy: (Math.random() - 0.5) * nodeSpeed,
          radius: Math.random() * 1.5 + 0.5, // Kích thước hạt hạt sao
        });
      }
    };

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
      ctx.lineWidth = 0.7;

      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        
        node.x += node.vx;
        node.y += node.vy;

        // Bật lại khi chạm biên
        if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1;

        // Vẽ node (ngôi sao)
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fill();

        // Vẽ đường nối các nodes gần nhau
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x - node.x;
          const dy = nodes[j].y - node.y;
          const distSq = dx * dx + dy * dy;

          if (distSq < connectionDistance * connectionDistance) {
            const opacity = 1 - Math.sqrt(distSq) / connectionDistance;
            // Màu nối tuỳ chỉnh (sáng rực)
            ctx.strokeStyle = `rgba(130, 200, 255, ${opacity * 0.5})`;
            ctx.beginPath();
            ctx.moveTo(node.x, node.y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.stroke();
          }
        }
      }
      animationFrameId = requestAnimationFrame(draw);
    };

    window.addEventListener("resize", resize);
    resize();
    draw();

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none opacity-15 mix-blend-screen z-[5]"
    />
  );
}
