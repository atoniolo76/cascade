import { useRef } from "react";
import { Plus } from "lucide-react";
import { ICON } from "../components/styles";
import { cx } from "../components/cx";
import { CheckDot, SourceIcon, Spinner } from "./icons";
import { SOURCES, sourceDef, type Source, type SourceId } from "./store";

/** Step 1: the 2×3 grid of trace sources. A card toggles its source; Upload opens a real file picker. */
export function StepSources({
  sources,
  connect,
  disconnect,
}: {
  sources: Source[];
  connect: (id: SourceId, detail?: string) => void;
  disconnect: (id: SourceId) => void;
}) {
  const file = useRef<HTMLInputElement>(null);

  const toggle = (id: SourceId) => {
    if (sources.some((s) => s.id === id)) disconnect(id);
    else if (id === "upload") file.current?.click();
    else connect(id);
  };

  return (
    <div className="mt-8 grid grid-cols-1 gap-2.5 sm:grid-cols-2">
      {SOURCES.map((def) => (
        <SourceCard
          key={def.id}
          id={def.id}
          source={sources.find((s) => s.id === def.id)}
          onClick={() => {
            toggle(def.id);
          }}
        />
      ))}
      <input
        ref={file}
        type="file"
        accept=".json,.jsonl,.parquet"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f !== undefined) connect("upload", f.name);
          e.target.value = "";
        }}
      />
    </div>
  );
}

function SourceCard({ id, source, onClick }: { id: SourceId; source: Source | undefined; onClick: () => void }) {
  const def = sourceDef(id);
  const connected = source?.status === "connected";
  const connecting = source?.status === "connecting";

  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={source !== undefined}
      className={cx(
        "ob-fade flex h-[68px] items-center gap-3 rounded-[14px] bg-surface px-3.5 text-left shadow-tile ring-1 ring-inset",
        connected ? "ring-(--st-ok-dot)/45" : "ring-border hover:bg-surface-2 hover:ring-border-strong",
      )}
    >
      <SourceIcon id={id} size="lg" />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[14px] leading-5 font-medium text-fg">{def.name}</span>
        <span className={cx("ob-fade block truncate text-[12.5px] leading-5", connected ? "text-(--st-ok-ink)" : "text-muted")}>
          {connected ? (
            <>
              {def.done} · <span className={cx(/\d/.test(source.detail) && "font-mono text-[11.5px]")}>{source.detail}</span>
            </>
          ) : connecting ? (
            "Connecting…"
          ) : (
            "Not connected"
          )}
        </span>
      </span>
      <span className="grid size-5 shrink-0 place-items-center text-faint">
        {connected ? <CheckDot /> : connecting ? <Spinner className="size-4" /> : <Plus className="size-4" {...ICON} />}
      </span>
    </button>
  );
}
