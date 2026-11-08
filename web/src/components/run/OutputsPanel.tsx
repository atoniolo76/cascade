import { useState } from "react";
import { Download, ExternalLink, FolderOpen } from "lucide-react";
import { downloadFile, request } from "../../api/client";
import { useRunFiles } from "../../api/queries";
import { formatBytes, plural } from "../../domain/format";
import { groupRunFiles, type ConfigFiles } from "../../domain/files";
import type { Series } from "../../domain/stats";
import { openInPerfetto } from "../../lib/perfetto";
import { toast } from "../../lib/toast";
import { paths } from "../../paths";
import { Button, ErrorNote, Panel, PanelHeader, Skeleton } from "../ui";
import { ICON } from "../styles";
import { FileRows } from "./Files";

const enc = encodeURIComponent;
const NO_KERNELS = "Available once a sim has simulated a kernel";

export function OutputsPanel({ runId, live, hasKernels, series }: { runId: string; live: boolean; hasKernels: boolean; series: readonly Series[] }) {
  const files = useRunFiles(runId, { live });
  const groups = groupRunFiles(files.data ?? [], series.map((s) => s.key));
  return (
    <Panel className="min-w-0">
      <PanelHeader
        icon={<FolderOpen {...ICON} />}
        title="Outputs"
        subtitle="What each sim's simulator wrote, uploaded when the sim finishes"
        actions={<PerfettoActions runId={runId} hasKernels={hasKernels} />}
      />
      {files.isError && files.data === undefined ? (
        <div className="p-5">
          <ErrorNote error={files.error} />
        </div>
      ) : files.data === undefined ? (
        <div aria-hidden className="flex flex-col gap-2 p-5">
          <Skeleton className="h-4 w-48" />
          <Skeleton className="h-4 w-72" />
        </div>
      ) : groups.length === 0 ? (
        <p className="px-5 py-4 text-[13px] text-muted">{live ? "No files yet; each sim uploads its outputs when it finishes." : "No sim in this run uploaded output files."}</p>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(360px,1fr))] gap-3 p-5">
          {groups.map((g) => (
            <ConfigCard key={g.config} group={g} series={series.find((s) => s.key === g.config)} />
          ))}
        </div>
      )}
    </Panel>
  );
}

function ConfigCard({ group, series }: { group: ConfigFiles; series: Series | undefined }) {
  return (
    <div className="min-w-0 rounded-[14px] bg-card px-4 py-3.5 ring-1 ring-border">
      <div className="flex items-center gap-2 text-[12px] text-fg-2">
        <span aria-hidden className="h-[7px] w-3 shrink-0 rounded-[2px]" style={{ background: series?.color ?? "var(--series-ref)" }} />
        <span className="truncate font-mono">{series?.label ?? group.config}</span>
        <span className="shrink-0 text-muted">· {plural(group.replicas.length, "replica")}</span>
      </div>
      <div className="mt-2 flex max-h-[520px] flex-col gap-2.5 overflow-y-auto overscroll-contain">
        {group.replicas.map((r) => (
          <section key={r.simId} aria-label={`Replica ${r.replica}`} className="min-w-0">
            <div className="flex items-baseline gap-2 border-b border-border pb-1 text-[12px]">
              <span className="shrink-0 font-medium text-fg">Replica {r.replica}</span>
              <span title={r.simId} className="min-w-0 truncate font-mono text-[11px] text-faint">
                {r.simId}
              </span>
              <span className="ml-auto shrink-0 text-muted tabular-nums">
                {plural(r.files.length, "file")} · {formatBytes(r.bytes)}
              </span>
            </div>
            <FileRows simId={r.simId} files={r.files} />
          </section>
        ))}
      </div>
    </div>
  );
}

function PerfettoActions({ runId, hasKernels }: { runId: string; hasKernels: boolean }) {
  const [opening, setOpening] = useState(false);
  const path = `/api/runs/${enc(runId)}/perfetto`;
  const fileName = `${runId}-perfetto.json`;
  const fail = (title: string) => (error: unknown) => {
    toast({ tone: "bad", title, body: error instanceof Error ? error.message : "Something went wrong" });
  };
  return (
    <>
      <Button
        size="sm"
        disabled={!hasKernels}
        title={hasKernels ? "The run's simulated kernels as a Chrome trace JSON, one track per sim" : NO_KERNELS}
        onClick={() => {
          downloadFile({ path, filename: fileName }).catch(fail("Couldn't download the Perfetto trace"));
        }}
      >
        <Download className="size-3.5" {...ICON} />
        Perfetto JSON
      </Button>
      <Button
        size="sm"
        disabled={!hasKernels || opening}
        title={hasKernels ? "Open the run's simulated kernels in ui.perfetto.dev, in a new tab" : NO_KERNELS}
        onClick={() => {
          setOpening(true);
          openInPerfetto(() => request(path).then((res) => res.arrayBuffer()), {
            title: `Cascade ${runId}`,
            fileName,
            url: new URL(paths.run(runId), window.location.origin).href,
          })
            .catch(fail("Couldn't open Perfetto"))
            .finally(() => {
              setOpening(false);
            });
        }}
      >
        <ExternalLink className="size-3.5" {...ICON} />
        {opening ? "Opening…" : "Open in Perfetto"}
      </Button>
    </>
  );
}
