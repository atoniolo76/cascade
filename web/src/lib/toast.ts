import { useSyncExternalStore } from "react";
import type { Tone } from "../domain/status";

export interface ToastAction {
  label: string;
  to: string;
}

export interface Toast {
  id: number;
  tone: Tone;
  title: string;
  body?: string;
  action?: ToastAction;
}

const LIFETIME_MS = 6000;
const MAX_SHOWN = 4;
const listeners = new Set<() => void>();
let toasts: readonly Toast[] = [];
let nextId = 1;

function publish(next: readonly Toast[]) {
  toasts = next;
  for (const fn of listeners) fn();
}

export function dismissToast(id: number) {
  publish(toasts.filter((t) => t.id !== id));
}

export function toast(t: Omit<Toast, "id">) {
  const id = nextId++;
  publish([...toasts, { ...t, id }].slice(-MAX_SHOWN));
  setTimeout(() => {
    dismissToast(id);
  }, LIFETIME_MS);
}

export function useToasts(): readonly Toast[] {
  return useSyncExternalStore(
    (fn) => {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    () => toasts,
  );
}
