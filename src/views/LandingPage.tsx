import { Button } from "@/components/ui/button";

interface LandingPageProps {
  onStart: () => void;
}

export function LandingPage({ onStart }: LandingPageProps) {
  return (
    <div className="relative w-full h-screen overflow-hidden bg-[#020205] flex flex-col items-center justify-between font-['Montserrat']">
      
      {/* Nền video (trung tâm) */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-0">
        <video 
          src="/logo/Started.mp4" 
          autoPlay 
          loop 
          muted 
          playsInline
          className="w-full max-w-[800px] object-contain opacity-90 scale-125 md:scale-150 transform translate-y-[-5%]" 
        />
        {/* Lớp phủ gradient mờ đi để video dễ chịu hơn */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_20%,#020205_70%)]" />
      </div>

      {/* Header (Logo trái, Nút Bắt đầu phải) */}
      <div className="relative z-10 w-full flex items-center justify-between px-8 py-6 md:px-4 md:py-4">
        <div className="flex flex-col items-center gap-1 cursor-default">
          <img src="/logo/Logo.png" alt="AuraSeek" className="w-24 h-24 object-contain drop-shadow-[0_0_15px_rgba(255,255,255,0.2)]" />
        </div>
        <Button 
          onClick={onStart}
          variant="outline" 
          className="rounded-full px-8 py-5 border-white/20 bg-white/5 text-white hover:bg-white/10 hover:border-white/40 transition-all font-medium text-[15px] shadow-[0_0_20px_rgba(255,255,255,0.05)] cursor-pointer"
        >
          Bắt đầu
        </Button>
      </div>
      
      {/* Footer & Slogan */}
      <div className="relative z-10 w-full flex flex-col items-center pb-8 pt-10">
        <p className="text-white/80 text-lg md:text-xl font-light tracking-wide mb-12">
          Tìm mọi thứ qua ống kính – Chuẩn xác, tức thì.
        </p>
        
        {/* Đường line gradient chia cách footer */}
        <div className="w-full h-[1px] bg-gradient-to-r from-transparent via-[#8a2be2]/40 to-transparent shadow-[0_0_15px_#8a2be2] mb-6" />
        
        <p className="text-white/40 text-xs font-light">
          © 2026 Auraseek. All rights reserved.
        </p>
      </div>
    </div>
  );
}
