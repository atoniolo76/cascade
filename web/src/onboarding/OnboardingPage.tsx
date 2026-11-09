import { useCallback, useEffect, useReducer, type CSSProperties, type ReactNode } from "react";
import { ArrowRight } from "lucide-react";
import { BrandTile } from "../components/Logo";
import { Button } from "../components/ui";
import { ICON } from "../components/styles";
import { cx } from "../components/cx";
import { Preview } from "./Preview";
import { StepGpus } from "./StepGpus";
import { StepRun } from "./StepRun";
import { StepTrace } from "./StepTrace";
import { canAdvance, initialState, reducer, simCount, STEPS, summary, type OnboardingState } from "./store";
import "./onboarding.css";

/** One run tick; 30 of them make the scripted 3 s run. */
const TICK_MS = 100;

interface Finish {
  /** Where "Open Cascade" goes: the app inside a workspace, Get started on the public site. */
  onFinish: () => void;
  finishLabel?: string;
}

export function OnboardingPage({ onFinish, finishLabel = "Open Cascade" }: Finish) {
  const [state, dispatch] = useReducer(reducer, initialState);
  // The scripted run ticks only while step 3 is running it.
  const running = state.run.status === "running";
  useEffect(() => {
    if (!running) return;
    const t = window.setInterval(() => {
      dispatch({ type: "tick" });
    }, TICK_MS);
    return () => {
      window.clearInterval(t);
    };
  }, [running]);

  const done = state.step === "done";
  const next = useCallback(() => {
    if (done) onFinish();
    else dispatch({ type: "next" });
  }, [done, onFinish]);
  const back = useCallback(() => { dispatch({ type: "back" }); }, []);

  // Enter advances and Esc goes back, unless focus is on a control that already owns the key.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
      const owned = e.target instanceof HTMLElement && e.target.closest("button, a, input, textarea, select") !== null;
      if (e.key === "Enter" && !owned) next();
      else if (e.key === "Escape") back();
    };
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("keydown", onKey); };
  }, [next, back]);

  return (
    <div className="ob min-h-dvh bg-window text-fg-2 lg:grid lg:h-dvh lg:grid-cols-[46fr_54fr]">
      <section className="flex min-h-dvh flex-col md:min-h-0 lg:min-h-0">
        <TopBar step={state.step} />
        <div className="flex flex-1 items-center px-6 py-10 sm:px-12 md:py-16 lg:py-10 xl:px-16">
          <div
            key={state.step}
            className="ob-slide w-full max-w-[540px]"
            style={{ "--ob-dir": state.direction } as CSSProperties}
          >
            <p className="mb-6 inline-flex h-7 max-w-full items-center truncate rounded-[9px] bg-surface px-2.5 font-mono text-[12px] text-muted ring-1 ring-border ring-inset md:hidden">
              {summary(state)}
            </p>
            <StepBody state={state} dispatch={dispatch} next={next} back={back} finishLabel={finishLabel} />
          </div>
        </div>
      </section>
      <section className="hidden p-2.5 md:block lg:h-dvh">
        <Preview state={state} />
      </section>
    </div>
  );
}

function TopBar({ step }: { step: OnboardingState["step"] }) {
  const index = step === "done" ? STEPS.length : step;
  return (
    <header className="flex h-16 shrink-0 items-center gap-4 px-6 sm:px-8">
      <span className="flex items-center gap-2.5">
        <BrandTile className="size-8 rounded-[10px] [&>svg]:size-[18px]" />
        <span className="text-[15px] font-medium text-fg">Cascade</span>
      </span>
      <div className="ml-auto flex items-center gap-4 sm:gap-8">
        <ol className="flex gap-1.5" aria-label={`Step ${index} of ${STEPS.length}`}>
          {STEPS.map((s) => (
            <li
              key={s}
              className={cx(
                "ob-fade h-[3px] w-5 rounded-full sm:w-8",
                s < index || step === "done" ? "bg-accent" : s === index ? "bg-fg" : "bg-block-off",
              )}
            />
          ))}
        </ol>
        <span className="text-[13px] whitespace-nowrap text-muted tabular-nums">{step === "done" ? "All set" : `Step ${step} of ${STEPS.length}`}</span>
      </div>
    </header>
  );
}

interface StepProps {
  state: OnboardingState;
  dispatch: (a: Parameters<typeof reducer>[1]) => void;
  next: () => void;
  back: () => void;
  finishLabel: string;
}

function StepBody({ state, dispatch, next, back, finishLabel }: StepProps) {
  const ready = canAdvance(state);
  switch (state.step) {
    case 1:
      return (
        <>
          <Headline lead="Pick a trace.">Cascade simulates a run you really profiled. Start with a sample, or send your own.</Headline>
          <StepTrace trace={state.trace} choose={(trace) => { dispatch({ type: "chooseTrace", trace }); }} />
          <Actions>
            <PrimaryButton disabled={!ready} onClick={next}>
              Continue
            </PrimaryButton>
          </Actions>
        </>
      );
    case 2: {
      const sims = simCount(state.entries);
      return (
        <>
          <Headline lead="Pick GPUs to simulate.">Each config runs in Accel-Sim. Add replicas to check the result holds.</Headline>
          {state.trace !== null && (
            <StepGpus trace={state.trace} entries={state.entries} setCount={(config, count) => { dispatch({ type: "setCount", config, count }); }} />
          )}
          <Actions>
            <PrimaryButton disabled={!ready} onClick={next}>
              {sims === 0 ? "Launch" : `Launch ${sims} simulation${sims === 1 ? "" : "s"}`}
            </PrimaryButton>
            <QuietButton onClick={back}>Back</QuietButton>
          </Actions>
        </>
      );
    }
    case 3:
      return (
        <>
          <Headline lead="Watch it run.">Cascade forks a sandbox per simulation and compares the result to the real run.</Headline>
          <StepRun state={state} />
          <Actions>
            <PrimaryButton disabled onClick={next} arrow={false}>
              Working…
            </PrimaryButton>
            <QuietButton onClick={back}>Back</QuietButton>
          </Actions>
        </>
      );
    case "done":
      return (
        <>
          <Headline lead="You’re set up.">Your first run finished. Send your own trace whenever you&rsquo;re ready.</Headline>
          <StepRun state={state} />
          <Actions>
            <PrimaryButton onClick={next}>{finishLabel}</PrimaryButton>
          </Actions>
        </>
      );
  }
}

function Headline({ lead, children }: { lead: string; children: ReactNode }) {
  return (
    <h1 className="text-[28px] leading-[1.18] font-normal tracking-[-0.02em] text-muted sm:text-[34px]">
      <span className="font-medium text-fg">{lead}</span> {children}
    </h1>
  );
}

function Actions({ children }: { children: ReactNode }) {
  return <div className="mt-8 flex items-center gap-2">{children}</div>;
}

function PrimaryButton({ disabled, onClick, arrow = true, children }: { disabled?: boolean; onClick: () => void; arrow?: boolean; children: ReactNode }) {
  return (
    <Button variant="primary" disabled={disabled} onClick={onClick} className="group h-10 px-4">
      {children}
      {arrow && <ArrowRight className={cx("size-4 transition-transform duration-200 ease-out", !disabled && "group-hover:translate-x-0.5")} {...ICON} />}
    </Button>
  );
}

function QuietButton({ onClick, children }: { onClick: () => void; children: ReactNode }) {
  return (
    <Button variant="ghost" onClick={onClick} className="h-10">
      {children}
    </Button>
  );
}
