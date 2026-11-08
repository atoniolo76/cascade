import { Kbd } from "../run/SimGrid";

const KEYS: readonly { keys: string[]; label: string }[] = [
  { keys: ["Space"], label: "Play or pause" },
  { keys: ["J", "K", "L"], label: "Shuttle: reverse, stop, forward (repeat to speed up)" },
  { keys: ["←", "→"], label: "Previous or next kernel boundary" },
  { keys: ["Home", "End"], label: "Jump to start or end" },
  { keys: ["+", "−"], label: "Zoom in or out" },
  { keys: ["F"], label: "Zoom to fit (or to the in/out range)" },
  { keys: ["I", "O"], label: "Mark in and out; sets zoom and range stats" },
  { keys: ["Esc"], label: "Clear selection, then the range" },
  { keys: ["⌘/Ctrl", "scroll"], label: "Zoom around the pointer (or pinch)" },
  { keys: ["⇧", "scroll"], label: "Pan (or scroll sideways, or drag)" },
];

export function Shortcuts({ onClose }: { onClose: () => void }) {
  return (
    <div role="dialog" aria-label="Keyboard shortcuts" className="absolute top-10 right-0 z-30 w-[min(340px,calc(100vw-32px))] animate-toast-in rounded-[16px] bg-surface p-4 shadow-pop ring-1 ring-border">
      <div className="mb-2.5 flex items-center justify-between">
        <span className="text-[13px] font-medium text-fg">Timeline shortcuts</span>
        <button type="button" onClick={onClose} className="rounded-[6px] px-1.5 text-[12px] text-muted hover:bg-surface-2 hover:text-fg">
          Esc
        </button>
      </div>
      <ul className="flex flex-col gap-1.5">
        {KEYS.map((k) => (
          <li key={k.label} className="grid grid-cols-[112px_minmax(0,1fr)] items-center gap-2 text-[12px] leading-5 text-fg-2">
            <span className="flex flex-wrap items-center">
              {k.keys.map((key) => (
                <Kbd key={key}>{key}</Kbd>
              ))}
            </span>
            <span>{k.label}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
