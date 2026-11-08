import { useState } from "react";
import { S3_BUCKET } from "../config";
import { Status } from "./meters";
import { CopyButton } from "./ui";

const ENDPOINT_KEY = "cascade.s3Endpoint";

function initialEndpoint(): string {
  try {
    const saved = localStorage.getItem(ENDPOINT_KEY);
    if (saved !== null) return saved;
  } catch {
    // Storage blocked: fall through to the defaults.
  }
  if (import.meta.env.VITE_S3_ENDPOINT !== undefined) return import.meta.env.VITE_S3_ENDPOINT;
  return ["localhost", "127.0.0.1"].includes(window.location.hostname) ? "http://localhost:9000" : "";
}

function profilerCommand({ endpoint, model, layers }: { endpoint: string; model: string; layers: string }) {
  return [
    "modal run accelsim_trace.py",
    `--s3-endpoint ${endpoint === "" ? "<s3-endpoint>" : endpoint}`,
    `--s3-bucket ${S3_BUCKET}`,
    "--s3-addressing-style path",
    `--model ${model}`,
    `--layers ${layers}`,
  ].join(" \\\n    ");
}

const FIELD =
  "h-10 w-full min-w-0 rounded-[12px] bg-surface px-3 font-mono text-[13px] text-fg ring-1 ring-border-strong outline-none ring-inset placeholder:font-sans placeholder:text-faint focus:ring-2 focus:ring-accent";

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <li className="grid grid-cols-[28px_minmax(0,1fr)] gap-x-3 gap-y-2">
      <span className="grid size-7 place-items-center rounded-[9px] bg-surface text-[13px] font-medium text-fg ring-1 ring-border">{n}</span>
      <div className="flex min-w-0 flex-col gap-2.5 pt-0.5">
        <div className="text-[15px] font-medium text-fg">{title}</div>
        {children}
      </div>
    </li>
  );
}

export function SendTraceGuide({ waiting }: { waiting: boolean }) {
  const [endpoint, setEndpoint] = useState(initialEndpoint);
  const [model, setModel] = useState("facebook/opt-125m");
  const [layers, setLayers] = useState("model.decoder.layers.10.self_attn");
  const command = profilerCommand({ endpoint: endpoint.trim(), model: model.trim(), layers: layers.trim() });

  return (
    <ol className="flex flex-col gap-7">
      <Step n={1} title="Point the profiler at this app's S3 endpoint">
        <p className="text-[14px] text-muted">
          Cascade serves an S3 bucket named <code className="font-mono text-[13px] text-fg-2">{S3_BUCKET}</code>. On Sail it is the URL exposed for port 9000.
        </p>
        <div className="flex items-center gap-1.5">
          <input
            aria-label="S3 endpoint"
            value={endpoint}
            placeholder="https://…sail.box"
            spellCheck={false}
            onChange={(e) => {
              setEndpoint(e.target.value);
              try {
                localStorage.setItem(ENDPOINT_KEY, e.target.value);
              } catch {
                // Storage blocked: the value lasts for this tab.
              }
            }}
            className={FIELD}
          />
          <CopyButton value={endpoint} label="S3 endpoint" className="size-10 rounded-[12px]" />
        </div>
      </Step>
      <Step n={2} title="Run the profiler on a GPU">
        <div className="grid gap-2 sm:grid-cols-2">
          <label className="flex min-w-0 flex-col gap-1">
            <span className="text-[13px] text-muted">Model</span>
            <input value={model} spellCheck={false} onChange={(e) => { setModel(e.target.value); }} className={FIELD} />
          </label>
          <label className="flex min-w-0 flex-col gap-1">
            <span className="text-[13px] text-muted">Layers</span>
            <input value={layers} spellCheck={false} onChange={(e) => { setLayers(e.target.value); }} className={FIELD} />
          </label>
        </div>
        <div className="relative rounded-[14px] bg-surface ring-1 ring-border">
          <pre className="overflow-x-auto py-3.5 pr-12 pl-4 font-mono text-[12.5px] leading-6 text-fg-2">{command}</pre>
          <CopyButton value={command} label="profiler command" className="absolute top-2.5 right-2.5" />
        </div>
        <p className="text-[13px] text-muted">
          Add <code className="font-mono text-[12px] text-fg-2">--list-layers</code> to print the model's layer names first.
        </p>
      </Step>
      <Step n={3} title="Watch it land">
        <p className="text-[14px] text-muted">The trace shows up here with its first file and counts kernels as they upload. You get a notice when it is ready to simulate.</p>
        {waiting && <Status tone="pending" live label="Waiting for the first file" />}
      </Step>
    </ol>
  );
}
