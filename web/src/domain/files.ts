import type { RunFile, SimFile } from "../api/types";

export const PREVIEW_MAX_BYTES = 16 * 1024 * 1024;
export const PREVIEW_LINES = 200;
const PREVIEW_LINE_CHARS = 2000;
const TEXT_NAME = /\.(csv|txt|log)(\.gz)?$/i;

export const isGzip = (name: string) => /\.gz$/i.test(name);

export function isPreviewable(file: Pick<SimFile, "name" | "size">): boolean {
  return file.size <= PREVIEW_MAX_BYTES && TEXT_NAME.test(file.name);
}

export interface TextHead {
  lines: string[];
  more: boolean;
}

const clip = (line: string) => {
  const s = line.endsWith("\r") ? line.slice(0, -1) : line;
  return s.length > PREVIEW_LINE_CHARS ? `${s.slice(0, PREVIEW_LINE_CHARS)} …` : s;
};

interface Lines {
  lines: string[];
  partial: string;
}

function feed(acc: Lines, text: string) {
  const parts = (acc.partial + text).split("\n");
  acc.partial = (parts.pop() ?? "").slice(0, PREVIEW_LINE_CHARS + 1);
  for (const p of parts) acc.lines.push(clip(p));
}

/** Stops reading once one line past `maxLines` is in, which is how the caller knows there is more. */
async function drain(stream: ReadableStream<BufferSource>, acc: Lines, maxLines: number): Promise<"ended" | "full"> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  try {
    while (acc.lines.length <= maxLines) {
      const { done, value } = await reader.read();
      if (done) {
        feed(acc, decoder.decode());
        return "ended";
      }
      feed(acc, decoder.decode(value, { stream: true }));
    }
    return "full";
  } finally {
    reader.cancel().catch(() => undefined);
  }
}

/** Offsets past 0 that look like a gzip member header: magic, deflate, no reserved flag bits. */
function memberCandidates(b: Uint8Array): number[] {
  const out: number[] = [];
  for (let i = 1; i + 3 < b.length; i++) {
    if (b[i] === 0x1f && b[i + 1] === 0x8b && b[i + 2] === 8 && ((b[i + 3] ?? 0) & 0xe0) === 0) out.push(i);
  }
  return out;
}

/**
 * Accel-Sim writes some .gz files as many concatenated gzip members, and the browser's
 * DecompressionStream stops at the end of the first one ("junk after end of data"). So each member
 * is inflated on its own. A header-looking byte run inside compressed data is a false boundary;
 * inflating up to it fails its CRC/length check, and the member is retried up to the next one.
 */
async function gunzipHead(blob: Blob, acc: Lines, maxLines: number): Promise<"ended" | "full"> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let start = 0;
  for (const end of [...memberCandidates(bytes), bytes.length]) {
    const trial = { lines: acc.lines.slice(), partial: acc.partial };
    try {
      const member = new Blob([bytes.subarray(start, end)]).stream().pipeThrough(new DecompressionStream("gzip"));
      const r = await drain(member, trial, maxLines);
      Object.assign(acc, trial);
      start = end;
      if (r === "full") return "full";
    } catch (error) {
      if (end === bytes.length) throw error;
    }
  }
  return "ended";
}

export async function readHead(blob: Blob, { gzip, maxLines = PREVIEW_LINES }: { gzip: boolean; maxLines?: number }): Promise<TextHead> {
  const acc: Lines = { lines: [], partial: "" };
  const r = gzip ? await gunzipHead(blob, acc, maxLines) : await drain(blob.stream(), acc, maxLines);
  if (r === "ended" && acc.partial !== "") acc.lines.push(clip(acc.partial));
  return { lines: acc.lines.slice(0, maxLines), more: r === "full" || acc.lines.length > maxLines };
}

export interface ReplicaFiles {
  simId: string;
  replica: number;
  files: RunFile[];
  bytes: number;
}

export interface ConfigFiles {
  config: string;
  replicas: ReplicaFiles[];
}

/** Groups in the run's config order; the server already orders by config, replica, then name. */
export function groupRunFiles(files: readonly RunFile[], configOrder: readonly string[]): ConfigFiles[] {
  const byConfig = new Map<string, Map<string, ReplicaFiles>>();
  for (const f of files) {
    const sims = byConfig.get(f.gpu_config) ?? new Map<string, ReplicaFiles>();
    byConfig.set(f.gpu_config, sims);
    const group = sims.get(f.sim_id) ?? { simId: f.sim_id, replica: f.replica, files: [], bytes: 0 };
    sims.set(f.sim_id, group);
    group.files.push(f);
    group.bytes += f.size;
  }
  const rank = (c: string) => {
    const i = configOrder.indexOf(c);
    return i === -1 ? configOrder.length : i;
  };
  return [...byConfig]
    .map(([config, sims]) => ({ config, replicas: [...sims.values()].toSorted((a, b) => a.replica - b.replica) }))
    .toSorted((a, b) => rank(a.config) - rank(b.config) || a.config.localeCompare(b.config));
}
