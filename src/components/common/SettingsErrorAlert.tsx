import { AlertCircle } from "lucide-react";

interface SettingsErrorAlertProps {
    error: string | null;
}

export function SettingsErrorAlert({ error }: SettingsErrorAlertProps) {
    if (!error) return null;

    return (
        <div className="rounded-xl border border-destructive/20 bg-destructive/5 p-4 flex gap-2 text-sm text-destructive">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            {error}
        </div>
    );
}

