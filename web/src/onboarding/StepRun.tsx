import { CheckCircle2 } from "lucide-react";
import { ICON } from "../components/styles";
import { cx } from "../components/cx";
import { CheckDot, QueuedDot, Spinner } from "./icons";
import { checklist, resultLine, type OnboardingState } from "./store";

/** Step 3 and the done state: the live checklist, and the result banner once the run is in. */
export function StepRun({ state }: { state: OnboardingState }) {
  return (
    <>
      <ul className="mt-8">
        {checklist(state).map((item) => {
          // The finished run's numbers sit in the banner below, so its row keeps just the label.
          const finished = item.key === "run" && item.state === "done";
          const sub = item.key === "run" && item.state === "active";
          return (
            <li key={item.key} className="flex items-start gap-3 border-b border-border py-3 last:border-b-0">
              <span className="mt-px">{item.state === "done" ? <CheckDot /> : item.state === "active" ? <Spinner className="m-0.5 size-4" /> : <QueuedDot />}</span>
              <span className="min-w-0 flex-1">
                <span className={cx("block text-[14px] leading-5", item.state === "queued" ? "text-muted" : "text-fg")}>
                  {item.label}
                  {!sub && !finished && <span className="text-muted"> · {item.detail}</span>}
                </span>
                {sub && (
                  <span key={item.detail.split(" ")[0]} className="ob-row-in mt-0.5 block font-mono text-[12px] leading-4 text-muted">
                    {item.detail}
                  </span>
                )}
              </span>
            </li>
          );
        })}
      </ul>
      {state.step === "done" && (
        <div className="ob-row-in mt-4 flex items-center gap-2.5 rounded-[12px] bg-(--st-ok-bg) px-3.5 py-2.5 text-[13.5px] text-(--st-ok-ink)">
          <CheckCircle2 className="size-4 shrink-0" {...ICON} />
          <span className="font-mono text-[12.5px]">{resultLine(state)}</span>
        </div>
      )}
    </>
  );
}
