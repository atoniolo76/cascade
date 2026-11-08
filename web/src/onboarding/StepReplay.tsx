import { TrendingDown } from "lucide-react";
import { ICON } from "../components/styles";
import { cx } from "../components/cx";
import { CheckDot, QueuedDot, Spinner } from "./icons";
import { checklist, fleetSize, type OnboardingState } from "./store";

/** Step 4 and the done state: the live checklist, and the result banner once the replay is in. */
export function StepReplay({ state }: { state: OnboardingState }) {
  const result = state.replay.result;
  return (
    <>
      <ul className="mt-8">
        {checklist(state).map((item) => (
          <li key={item.key} className="flex items-start gap-3 border-b border-border py-3 last:border-b-0">
            <span className="mt-px">
              {item.state === "done" ? <CheckDot /> : item.state === "active" ? <Spinner className="m-0.5 size-4" /> : <QueuedDot />}
            </span>
            <span className="min-w-0 flex-1">
              <span className={cx("block text-[14px] leading-5", item.state === "queued" ? "text-muted" : "text-fg")}>
                {item.label}
                {item.state === "done" && item.detail !== "" && item.key !== "replay" && <span className="text-muted"> · {item.detail}</span>}
              </span>
              {(item.state === "active" || item.key === "replay") && item.state !== "queued" && (
                <span key={item.detail.split(" ")[0]} className="ob-row-in mt-0.5 block font-mono text-[12px] leading-4 text-muted">
                  {item.detail}
                </span>
              )}
            </span>
          </li>
        ))}
      </ul>
      {state.step === "done" && result !== null && (
        <div className="ob-row-in mt-4 flex items-center gap-2.5 rounded-[12px] bg-(--st-ok-bg) px-3.5 py-2.5 text-[13.5px] text-(--st-ok-ink)">
          <TrendingDown className="size-4 shrink-0" {...ICON} />
          <span>
            Replayed <span className="font-mono">1,200</span> requests · p99 <span className="font-mono">{result.vsBaselinePct < 0 ? "−" : "+"}{Math.abs(result.vsBaselinePct)}%</span> vs{" "}
            <span className="font-mono">{fleetSize(state.fleet)}×</span> A100
          </span>
        </div>
      )}
    </>
  );
}
