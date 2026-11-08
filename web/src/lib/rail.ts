import { useSyncExternalStore } from "react";

export type RailState = "collapsed" | "expanded";

export const RAIL_WIDTH: Record<RailState, number> = { collapsed: 72, expanded: 240 };

const KEY = "cascade.rail";
const listeners = new Set<() => void>();

export function readRailState(storage: Pick<Storage, "getItem"> | undefined): RailState {
  try {
    return storage?.getItem(KEY) === "expanded" ? "expanded" : "collapsed";
  } catch {
    return "collapsed";
  }
}

export interface ShortcutKey {
  key: string;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey: boolean;
  repeat: boolean;
  defaultPrevented: boolean;
}

export function isRailShortcut(e: ShortcutKey, typing: boolean): boolean {
  return e.key === "[" && !e.metaKey && !e.ctrlKey && !e.altKey && !e.repeat && !e.defaultPrevented && !typing;
}

let state = readRailState(globalThis.localStorage);

export const railStore = {
  get: () => state,
  toggle: () => {
    state = state === "expanded" ? "collapsed" : "expanded";
    try {
      if (state === "expanded") localStorage.setItem(KEY, state);
      else localStorage.removeItem(KEY);
    } catch {
      // Storage blocked: the choice lasts for this tab.
    }
    for (const fn of listeners) fn();
  },
  subscribe: (fn: () => void) => {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },
};

export function useRailState(): RailState {
  return useSyncExternalStore(railStore.subscribe, railStore.get);
}
