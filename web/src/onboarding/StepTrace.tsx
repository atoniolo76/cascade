import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { ICON } from "../components/styles";
import { cx } from "../components/cx";
import { BrandIcon, CheckDot } from "./icons";
import { TRACES, type TraceDef, type TraceId } from "./store";

const PROFILER = ["modal run accelsim_trace.py", "--s3-endpoint <your workspace's :9000 URL>", "--s3-bucket cascade", "--model facebook/opt-125m"].join(" \\\n    ");

/** Step 1: the two samples every workspace opens with, and how to send your own. */
export function StepTrace({ trace, choose }: { trace: TraceId | null; choose: (t: TraceId) => void }) {
  const [own, setOwn] = useState(false);
  return (
    <>
      <div className="mt-8 grid gap-2.5" role="radiogroup" aria-label="Trace">
        {TRACES.map((t) => (
          <TraceCard key={t.id} def={t} selected={trace === t.id} onClick={() => { choose(t.id); }} />
        ))}
      </div>
      <div className="mt-2.5 rounded-[14px] border border-border">
        <button
          type="button"
          aria-expanded={own}
          onClick={() => { setOwn((o) => !o); }}
          className="flex h-[60px] w-full items-center gap-3 rounded-[14px] px-3.5 text-left"
        >
          <BrandIcon brand="modal" size="lg" />
          <span className="min-w-0 flex-1">
            <span className="block text-[14px] leading-5 font-medium text-fg">Profile your own</span>
            <span className="block text-[12.5px] leading-5 text-muted">Run the profiler on a Modal GPU. The trace uploads to your workspace.</span>
          </span>
          <ChevronRight className={cx("size-4 shrink-0 text-muted transition-transform duration-200 ease-out", own && "rotate-90")} {...ICON} />
        </button>
        {own && (
          <div className="ob-row-in border-t border-border px-3.5 py-3">
            <pre className="overflow-x-auto rounded-[10px] bg-surface-2 px-3 py-2.5 font-mono text-[12px] leading-5 text-fg-2">{PROFILER}</pre>
            <p className="mt-2 text-[12.5px] leading-5 text-muted">Profiling takes about 30 minutes. Pick a sample to keep going in the meantime.</p>
          </div>
        )}
      </div>
    </>
  );
}

function TraceCard({ def, selected, onClick }: { def: TraceDef; selected: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      onClick={onClick}
      className={cx(
        "ob-fade flex h-[68px] items-center gap-3 rounded-[14px] border bg-surface px-3.5 text-left",
        selected ? "border-accent bg-accent-soft" : "border-border hover:border-border-strong hover:bg-surface-2",
      )}
    >
      <BrandIcon brand="meta" size="lg" />
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-2 text-[14px] leading-5 font-medium text-fg">
          {def.name}
          <span className="rounded-[6px] bg-accent-soft px-1.5 text-[11px] leading-[18px] font-medium text-accent-ink">Sample</span>
        </span>
        <span className="block truncate text-[12.5px] leading-5 text-muted">
          vLLM · profiled on {def.gpu} · <span className="font-mono text-[11.5px]">{def.kernels.toLocaleString("en-US")}</span> kernels traced
        </span>
      </span>
      <span className="grid size-5 shrink-0 place-items-center">{selected ? <CheckDot /> : <span className="size-4 rounded-full border border-border-strong" />}</span>
    </button>
  );
}
