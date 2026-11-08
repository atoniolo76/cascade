import type { CSSProperties } from "react";
import type { SimState } from "../api/types";
import { SIM_BLOCK_ORDER, SIM_STATE, TONE_VARS, type Tone } from "../domain/status";
import { cx } from "./cx";

export function Status({ tone, label, live = false, className }: { tone: Tone; label: string; live?: boolean; className?: string }) {
  const v = TONE_VARS[tone];
  return (
    <span className={cx("inline-flex shrink-0 items-center gap-2 text-[13px] leading-5 font-medium whitespace-nowrap", className)} style={{ color: v.ink }}>
      <span aria-hidden className="inline-flex gap-[2px]">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className={cx("h-[10px] w-[6px] rounded-[2px]", live && "animate-block-wave")}
            style={{ background: v.dot, animationDelay: live ? `${i * 160}ms` : undefined }}
          />
        ))}
      </span>
      {label}
    </span>
  );
}

const BLOCK = "size-[10px] shrink-0 rounded-[3px] transition-colors duration-300";

export function BlockMeter({
  value,
  blocks = 10,
  tone = "active",
  label,
  className,
}: {
  value: number;
  blocks?: number;
  tone?: Tone;
  label: string;
  className?: string;
}) {
  const v = Math.max(0, Math.min(1, value));
  const raw = Math.round(v * blocks);
  const filled = v > 0 && raw === 0 ? 1 : v < 1 && raw === blocks ? blocks - 1 : raw;
  return (
    <span role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(v * 100)} className={cx("inline-flex gap-[3px]", className)}>
      {Array.from({ length: blocks }, (_, i) => (
        <span key={i} className={BLOCK} style={{ background: i < filled ? TONE_VARS[tone].dot : "var(--block-off)" }} />
      ))}
    </span>
  );
}

type Counts = Partial<Record<SimState, number>>;

/** Largest-remainder split of the counts into `slots` blocks; a nonzero state always keeps one. */
function allocate(counts: Counts, slots: number): SimState[] {
  const present = SIM_BLOCK_ORDER.filter((s) => (counts[s] ?? 0) > 0);
  const total = present.reduce((n, s) => n + (counts[s] ?? 0), 0);
  if (total <= slots) return present.flatMap((s) => Array<SimState>(counts[s] ?? 0).fill(s));
  const shares = present.map((s) => {
    const exact = ((counts[s] ?? 0) / total) * slots;
    return { s, n: Math.max(1, Math.floor(exact)), rem: exact - Math.floor(exact) };
  });
  let left = slots - shares.reduce((n, x) => n + x.n, 0);
  for (const x of shares.toSorted((a, b) => b.rem - a.rem)) {
    if (left <= 0) break;
    x.n++;
    left--;
  }
  while (left < 0) {
    const biggest = shares.reduce((a, b) => (b.n > a.n ? b : a));
    biggest.n--;
    left++;
  }
  return shares.flatMap((x) => Array<SimState>(x.n).fill(x.s));
}

/** One block per sim colored by state, finished sims first; past `slots` sims the blocks share out proportionally. */
export function SimStateBlocks({ counts, slots = 16, className, style }: { counts: Counts; slots?: number; className?: string; style?: CSSProperties }) {
  const states = allocate(counts, slots);
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const summary = SIM_BLOCK_ORDER.flatMap((s) => ((counts[s] ?? 0) > 0 ? [`${counts[s] ?? 0} ${SIM_STATE[s].label.toLowerCase()}`] : [])).join(", ");
  return (
    <span role="img" aria-label={`${total} sims: ${summary}`} title={summary} className={cx("inline-flex flex-wrap gap-[3px]", className)} style={style}>
      {states.map((s, i) => (
        <span
          key={i}
          className={cx(BLOCK, SIM_STATE[s].live && "animate-pulse-soft")}
          style={{ background: s === "pending" ? "var(--block-off)" : TONE_VARS[SIM_STATE[s].tone].dot }}
        />
      ))}
    </span>
  );
}
