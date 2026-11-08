import { useSyncExternalStore } from "react";

/** The kernel under the pointer or keyboard, shared by every chart on a run page so they highlight the same row. */
export interface KernelFocus {
  scope: string;
  index: number;
  source: string;
}

let focus: KernelFocus | null = null;
const listeners = new Set<() => void>();

export function setKernelFocus(next: KernelFocus | null) {
  if (next?.scope === focus?.scope && next?.index === focus?.index && next?.source === focus?.source) return;
  focus = next;
  for (const l of listeners) l();
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function useKernelFocus(scope: string): KernelFocus | null {
  const f = useSyncExternalStore(subscribe, () => focus);
  return f?.scope === scope ? f : null;
}
