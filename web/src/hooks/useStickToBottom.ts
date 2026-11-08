import { useCallback, useLayoutEffect, useRef, useState } from "react";

const SLACK_PX = 12;

/** Keeps a scroll box pinned to its bottom as content grows, until the user scrolls up. */
export function useStickToBottom<T extends HTMLElement>(content: unknown) {
  const ref = useRef<T>(null);
  const pinnedRef = useRef(true);
  const [pinned, setPinned] = useState(true);

  const onScroll = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const next = el.scrollHeight - el.scrollTop - el.clientHeight <= SLACK_PX;
    if (next !== pinnedRef.current) {
      pinnedRef.current = next;
      setPinned(next);
    }
  }, []);

  const jump = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    pinnedRef.current = true;
    setPinned(true);
  }, []);

  useLayoutEffect(() => {
    const el = ref.current;
    if (el && pinnedRef.current) el.scrollTop = el.scrollHeight;
  }, [content]);

  return { ref, onScroll, pinned, jump };
}
