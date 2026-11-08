import { useSyncExternalStore } from "react";

export type ThemePref = "system" | "light" | "dark";

const KEY = "cascade.theme";
const NEXT: Record<ThemePref, ThemePref> = { system: "light", light: "dark", dark: "system" };
const listeners = new Set<() => void>();

function read(): ThemePref {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

let pref = read();

function apply() {
  if (pref === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = pref;
}

apply();

export const themeStore = {
  get: () => pref,
  cycle: () => {
    pref = NEXT[pref];
    try {
      if (pref === "system") localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, pref);
    } catch {
      // Storage blocked: the choice lasts for this tab.
    }
    apply();
    for (const fn of listeners) fn();
  },
  subscribe: (fn: () => void) => {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },
};

export function useThemePref(): ThemePref {
  return useSyncExternalStore(themeStore.subscribe, themeStore.get);
}
