import { Loader2, AlertTriangle, Info } from "lucide-react";
import { Button } from "./button";

interface ConfirmDialogProps {
  isOpen: boolean;
  title: string;
  description: string;
  confirmText?: string;
  cancelText?: string;
  isDestructive?: boolean;
  isLoading?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  type?: "confirm" | "alert";
}

export function ConfirmDialog({
  isOpen,
  title,
  description,
  confirmText = "Xác nhận",
  cancelText = "Hủy",
  isDestructive = false,
  isLoading = false,
  onConfirm,
  onCancel,
  type = "confirm",
}: ConfirmDialogProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm transition-opacity" 
        onClick={!isLoading ? onCancel : undefined}
      />
      
      {/* Dialog */}
      <div className="relative bg-background border border-border/30 rounded-2xl shadow-2xl max-w-md w-full overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        <div className="p-6">
          <div className="flex gap-4">
            <div className={`shrink-0 p-3 rounded-full flex items-center justify-center h-12 w-12 ${isDestructive ? 'bg-destructive/10 text-destructive' : 'bg-primary/10 text-primary'}`}>
              {isDestructive ? <AlertTriangle className="w-6 h-6" /> : <Info className="w-6 h-6" />}
            </div>
            <div className="space-y-2 mt-1">
              <h3 className="font-semibold text-lg tracking-tight">{title}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {description}
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-muted/30 px-6 py-4 flex items-center justify-end gap-3 flex-row-reverse border-t border-border/20">
          <Button 
            variant={isDestructive ? "destructive" : "default"}
            onClick={onConfirm}
            disabled={isLoading}
            className="min-w-[100px]"
          >
            {isLoading && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            {type === "alert" ? "Đóng" : confirmText}
          </Button>
          
          {type === "confirm" && (
            <Button
              variant="outline"
              onClick={onCancel}
              disabled={isLoading}
            >
              {cancelText}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
