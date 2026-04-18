import { createContext, useContext, useState, ReactNode } from "react";

type SelectionContextType = {
    selectedIds: Set<string>;
    toggleSelection: (id: string) => void;
    clearSelection: () => void;
};

const SelectionContext = createContext<SelectionContextType | undefined>(undefined);

export function SelectionProvider({ children }: { children: ReactNode }) {
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

    const toggleSelection = (id: string) => {
        setSelectedIds(prev => {
            const newSet = new Set(prev);
            if (newSet.has(id)) newSet.delete(id);
            else newSet.add(id);
            return newSet;
        });
    };

    const clearSelection = () => setSelectedIds(new Set());

    return (
        <SelectionContext.Provider value={{ selectedIds, toggleSelection, clearSelection }}>
            {children}
        </SelectionContext.Provider>
    );
}

export const useSelection = () => {
    const ctx = useContext(SelectionContext);
    if (!ctx) throw new Error("useSelection must be used within SelectionProvider");
    return ctx;
};
