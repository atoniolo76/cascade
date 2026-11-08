import { useState } from "react";

export type View = "chart" | "table";
export const VIEWS = [
  { value: "chart", label: "Chart" },
  { value: "table", label: "Table" },
] as const;

export const REAL = { key: "__real__", label: "Real hardware", color: "var(--fg-2)" } as const;
export const TABLE_MAX_HEIGHT = 560;

/** Series the reader switched off in a legend; color stays with the series either way. */
export function useHiddenKeys() {
  const [hidden, setHidden] = useState<ReadonlySet<string>>(new Set());
  const toggle = (key: string) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (!next.delete(key)) next.add(key);
      return next;
    });
  };
  return [hidden, toggle] as const;
}
