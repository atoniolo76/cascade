import { useSyncExternalStore } from "react";

const subscribers = new Set<() => void>();
let now = Date.now();
let timer: ReturnType<typeof setInterval> | undefined;

function subscribe(fn: () => void) {
  subscribers.add(fn);
  if (timer === undefined) {
    now = Date.now();
    timer = setInterval(() => {
      now = Date.now();
      for (const s of subscribers) s();
    }, 1000);
  }
  return () => {
    subscribers.delete(fn);
    if (subscribers.size === 0 && timer !== undefined) {
      clearInterval(timer);
      timer = undefined;
    }
  };
}

/** Wall clock that ticks once a second, shared by every subscriber. */
export function useNow(): number {
  return useSyncExternalStore(subscribe, () => now);
}
