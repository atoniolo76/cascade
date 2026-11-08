import { useEffect, useState } from "react";
import { Download, Eye, FileArchive, FileBraces, FileSpreadsheet, FileText, type LucideIcon } from "lucide-react";
import { downloadFile, request } from "../../api/client";
import type { SimFile } from "../../api/types";
import { formatBytes, formatInt } from "../../domain/format";
import { PREVIEW_LINES, isGzip, isPreviewable, readHead, type TextHead } from "../../domain/files";
import { toast } from "../../lib/toast";
import { Button, Dialog, ErrorNote, Skeleton } from "../ui";
import { ICON } from "../styles";
import { cx } from "../cx";

const enc = encodeURIComponent;
const filePath = (simId: string, name: string) => `/api/sims/${enc(simId)}/files/${enc(name)}`;
const MINI = "inline-grid size-7 shrink-0 place-items-center rounded-[8px] text-faint transition-colors duration-150 hover:bg-surface-2 hover:text-fg";

function fileIcon(name: string): LucideIcon {
  const base = name.toLowerCase().replace(/\.gz$/, "");
  if (base.endsWith(".csv")) return FileSpreadsheet;
  if (base.endsWith(".json")) return FileBraces;
  return isGzip(name) ? FileArchive : FileText;
}

const errorMessage = (error: unknown) => (error instanceof Error ? error.message : "Something went wrong");

function download(simId: string, name: string) {
  downloadFile({ path: filePath(simId, name), filename: name }).catch((error: unknown) => {
    toast({ tone: "bad", title: `Couldn't download ${name}`, body: errorMessage(error) });
  });
}

export function FileRows({ simId, files, className }: { simId: string; files: readonly SimFile[]; className?: string }) {
  const [previewing, setPreviewing] = useState<SimFile | null>(null);
  return (
    <>
      <ul className={cx("flex flex-col", className)}>
        {files.map((f) => {
          const Icon = fileIcon(f.name);
          return (
            <li key={f.name} className="flex h-7 min-w-0 items-center gap-2 text-[13px]">
              <Icon className="size-3.5 shrink-0 text-faint" {...ICON} />
              <span title={f.name} className="min-w-0 flex-1 truncate font-mono text-[12px] text-fg-2">
                {f.name}
              </span>
              <span className="shrink-0 text-[12px] text-muted tabular-nums">{formatBytes(f.size)}</span>
              <span className="flex w-[60px] shrink-0 justify-end">
                {isPreviewable(f) && (
                  <button type="button" aria-label={`Preview ${f.name}`} title="Preview" className={MINI} onClick={() => { setPreviewing(f); }}>
                    <Eye className="size-3.5" {...ICON} />
                  </button>
                )}
                <button type="button" aria-label={`Download ${f.name}`} title="Download" className={MINI} onClick={() => { download(simId, f.name); }}>
                  <Download className="size-3.5" {...ICON} />
                </button>
              </span>
            </li>
          );
        })}
      </ul>
      {previewing !== null && (
        <FilePreview
          simId={simId}
          file={previewing}
          onClose={() => {
            setPreviewing(null);
          }}
        />
      )}
    </>
  );
}

type Head = { kind: "loading" } | { kind: "error"; error: unknown } | ({ kind: "ready" } & TextHead);

function FilePreview({ simId, file, onClose }: { simId: string; file: SimFile; onClose: () => void }) {
  const [head, setHead] = useState<Head>({ kind: "loading" });

  useEffect(() => {
    const abort = new AbortController();
    request(filePath(simId, file.name), { signal: abort.signal })
      .then((res) => res.blob())
      .then((blob) => readHead(blob, { gzip: isGzip(file.name) }))
      .then(
        (h) => {
          setHead({ kind: "ready", ...h });
        },
        (error: unknown) => {
          if (!abort.signal.aborted) setHead({ kind: "error", error });
        },
      );
    return () => {
      abort.abort();
    };
  }, [simId, file.name]);

  const shown = head.kind !== "ready" ? null : head.more ? `first ${formatInt(PREVIEW_LINES)} lines` : `${formatInt(head.lines.length)} lines`;
  return (
    <Dialog
      title={<span className="font-mono text-[14px]">{file.name}</span>}
      subtitle={
        <span className="tabular-nums">
          {formatBytes(file.size)}
          {isGzip(file.name) && " gzipped"}
          {shown !== null && ` · ${shown}`}
        </span>
      }
      actions={
        <Button size="sm" onClick={() => { download(simId, file.name); }}>
          <Download className="size-3.5" {...ICON} />
          Download
        </Button>
      }
      onClose={onClose}
    >
      <div className="min-h-0 flex-1 overflow-auto overscroll-contain bg-card py-3">
        {head.kind === "loading" ? (
          <div className="flex flex-col gap-2 px-5">
            {Array.from({ length: 10 }, (_, i) => (
              <Skeleton key={i} className="h-3" />
            ))}
          </div>
        ) : head.kind === "error" ? (
          <div className="px-5">
            <ErrorNote error={head.error} />
          </div>
        ) : head.lines.length === 0 ? (
          <p className="px-5 text-[13px] text-faint">The file is empty.</p>
        ) : (
          <div className="w-max min-w-full font-mono text-[12px] leading-5 text-fg-2">
            {head.lines.map((line, i) => (
              <div key={i} className="flex pr-5">
                <span aria-hidden className="sticky left-0 w-14 shrink-0 bg-card pr-3 text-right text-faint tabular-nums select-none">
                  {i + 1}
                </span>
                <span className="whitespace-pre">{line === "" ? " " : line}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      {head.kind === "ready" && head.more && (
        <p className="border-t border-border px-5 py-2.5 text-[12px] text-muted">Showing the first {formatInt(PREVIEW_LINES)} lines. Download the file for the rest.</p>
      )}
    </Dialog>
  );
}
