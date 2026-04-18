import type { ReactNode } from "react";
import { createContext, useContext, useMemo, useState } from "react";
import en from "@/data/dictionaries/en.json";
import vi from "@/data/dictionaries/vi.json";

type SupportedLanguage = "en" | "vi";

type DictionaryContextValue = {
  language: SupportedLanguage;
  setLanguage: (language: SupportedLanguage) => void;
  t: (key: string) => string;
};

const DictionaryContext = createContext<DictionaryContextValue | undefined>(
  undefined,
);

const dictionaries: Record<SupportedLanguage, unknown> = {
  en,
  vi,
};

const getNestedValue = (object: unknown, key: string): unknown => {
  if (!object) return undefined;

  return key.split(".").reduce<unknown>((current, part) => {
    if (
      current &&
      typeof current === "object" &&
      part in (current as Record<string, unknown>)
    ) {
      return (current as Record<string, unknown>)[part];
    }

    return undefined;
  }, object);
};

export function DictionaryProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<SupportedLanguage>("en");

  const value = useMemo<DictionaryContextValue>(
    () => ({
      language,
      setLanguage,
      t: (key: string) => {
        const dictionary = dictionaries[language];
        const value = getNestedValue(dictionary, key);

        if (typeof value === "string") {
          return value;
        }

        return key;
      },
    }),
    [language],
  );

  return (
    <DictionaryContext.Provider value={value}>
      {children}
    </DictionaryContext.Provider>
  );
}

export function useDictionary() {
  const context = useContext(DictionaryContext);

  if (!context) {
    throw new Error("useDictionary must be used within a DictionaryProvider");
  }

  return context;
}

