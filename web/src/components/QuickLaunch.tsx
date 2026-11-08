import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Link, useNavigate } from "react-router";
import { Play, Rocket } from "lucide-react";
import { useCreateRun, useGpuConfigs } from "../api/queries";
import type { TraceSummary } from "../api/types";
import { suggestedConfig } from "../domain/configs";
import { gpuLogo } from "../domain/logos";
import { LogoTile } from "./BrandLogo";
import { toast } from "../lib/toast";
import { paths } from "../paths";
import { Button, ErrorNote, Stepper } from "./ui";
import { ICON, buttonClass } from "./styles";

const MAX_COUNT = 64;
const POP_WIDTH = 300;

export function QuickLaunch({ trace, size = "md", variant = "primary" }: { trace: TraceSummary; size?: "md" | "sm"; variant?: "primary" | "secondary" }) {
  const [openAt, setOpenAt] = useState<DOMRect | null>(null);
  const anchor = useRef<HTMLButtonElement>(null);
  useGpuConfigs();
  if (trace.state !== "ready") {
    return (
      <Button size={size} variant={variant} disabled title={`Trace is ${trace.state}; runs need a ready trace`}>
        <Play className="size-3.5" {...ICON} />
        Run
      </Button>
    );
  }
  return (
    <>
      <Button
        ref={anchor}
        size={size}
        variant={variant}
        aria-expanded={openAt !== null}
        aria-haspopup="dialog"
        onClick={(e) => {
          e.stopPropagation();
          setOpenAt(openAt === null ? e.currentTarget.getBoundingClientRect() : null);
        }}
      >
        <Play className="size-3.5" {...ICON} />
        Run
      </Button>
      {openAt !== null && (
        <LaunchPopover
          trace={trace}
          anchor={anchor}
          at={openAt}
          onClose={() => {
            setOpenAt(null);
          }}
        />
      )}
    </>
  );
}

function LaunchPopover({
  trace,
  anchor,
  at,
  onClose,
}: {
  trace: TraceSummary;
  anchor: React.RefObject<HTMLButtonElement | null>;
  at: DOMRect;
  onClose: () => void;
}) {
  const configs = useGpuConfigs();
  const create = useCreateRun();
  const navigate = useNavigate();
  const [count, setCount] = useState(1);
  const pop = useRef<HTMLDivElement>(null);
  const config = configs.data ? suggestedConfig(trace, configs.data) : undefined;

  const left = Math.max(12, Math.min(window.innerWidth - POP_WIDTH - 12, at.right - POP_WIDTH));

  useEffect(() => {
    pop.current?.querySelector<HTMLElement>("[data-autofocus]")?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    const onDown = (e: PointerEvent) => {
      if (e.target instanceof Node && !pop.current?.contains(e.target) && !anchor.current?.contains(e.target)) onClose();
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("scroll", onClose, { passive: true });
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("scroll", onClose);
    };
  }, [anchor, onClose]);

  const launch = () => {
    if (!config) return;
    create.mutate(
      { trace_id: trace.id, entries: [{ gpu_config: config.name, count }] },
      {
        onSuccess: (run) => {
          toast({ tone: "active", title: "Run launched", body: `${config.name} × ${count} · ${run.id}` });
          onClose();
          void navigate(paths.run(run.id));
        },
      },
    );
  };

  return createPortal(
    <div
      ref={pop}
      role="dialog"
      aria-label="Launch a run"
      onClick={(e) => {
        e.stopPropagation();
      }}
      className="fixed z-50 flex animate-toast-in flex-col gap-3 rounded-[16px] bg-surface p-4 text-left shadow-pop ring-1 ring-border"
      style={{ width: POP_WIDTH, top: at.bottom + 8, left }}
    >
      <div>
        <div className="text-[14px] font-medium text-fg">Launch a run</div>
        <div className="text-[13px] text-muted">Suggested config for this {trace.trace_gpu_arch} trace</div>
      </div>
      {config === undefined ? (
        <div className="text-[13px] text-muted">{configs.isPending ? "Loading configs…" : `No config matches ${trace.trace_gpu_arch}.`}</div>
      ) : (
        <div className="flex items-center justify-between gap-3">
          <LogoTile logo={gpuLogo(config.name)} fallback={config.name} />
          <span className="min-w-0 flex-1">
            <span className="block truncate font-mono text-[14px] font-medium text-fg">{config.name}</span>
            <span className="block text-[12px] leading-4 text-muted">{config.core_clock_mhz} MHz · replicas</span>
          </span>
          <Stepper label="replicas" value={count} min={1} max={MAX_COUNT} onChange={setCount} />
        </div>
      )}
      {create.isError && <ErrorNote error={create.error} />}
      <div className="flex items-center gap-2">
        <Button data-autofocus variant="primary" className="flex-1" disabled={config === undefined || create.isPending} onClick={launch}>
          <Rocket className="size-4" {...ICON} />
          {create.isPending ? "Launching…" : `Launch ${count === 1 ? "1 sim" : `${count} sims`}`}
        </Button>
        <Link to={paths.newRun(trace.id)} className={buttonClass({ variant: "ghost" })}>
          More options
        </Link>
      </div>
    </div>,
    document.body,
  );
}
