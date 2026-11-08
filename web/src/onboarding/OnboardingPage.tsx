import { useCallback, useEffect, useReducer, useRef, type CSSProperties, type ReactNode } from "react";
import { ArrowRight } from "lucide-react";
import { useNavigate } from "react-router";
import { BrandTile } from "../components/Logo";
import { Button } from "../components/ui";
import { ICON } from "../components/styles";
import { cx } from "../components/cx";
import { Preview } from "./Preview";
import { StepFleet } from "./StepFleet";
import { StepReplay } from "./StepReplay";
import { StepRouter } from "./StepRouter";
import { StepSources } from "./StepSources";
import { canAdvance, fleetSize, initialState, reducer, STEPS, summary, type OnboardingState, type SourceId } from "./store";
import "./onboarding.css";

/** How long a source shows as "connecting" before it confirms. */
const CONNECT_MS = 650;
/** One replay tick; 25 of them make the scripted 2.5 s replay. */
const TICK_MS = 100;

export function OnboardingPage() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const timers = useRef(new Set<number>());

  useEffect(() => {
    const live = timers.current;
    return () => { live.forEach((t) => { window.clearTimeout(t); }); };
  }, []);

  const connect = useCallback((id: SourceId, detail?: string) => {
    dispatch({ type: "connect", id, detail });
    const t = window.setTimeout(() => {
      timers.current.delete(t);
      dispatch({ type: "confirm", id });
    }, CONNECT_MS);
    timers.current.add(t);
  }, []);

  // The scripted replay ticks only while step 4 is running it.
  const replaying = state.replay.status === "running";
  useEffect(() => {
    if (!replaying) return;
    const t = window.setInterval(() => { dispatch({ type: "tick" }); }, TICK_MS);
    return () => { window.clearInterval(t); };
  }, [replaying]);

  const navigate = useNavigate();
  const done = state.step === "done";
  const next = useCallback(() => {
    if (done) void navigate("/");
    else dispatch({ type: "next" });
  }, [done, navigate]);
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
            <StepBody state={state} connect={connect} dispatch={dispatch} next={next} back={back} />
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
        <span className="text-[13px] whitespace-nowrap text-muted tabular-nums">{step === "done" ? "All set" : `Step ${step} of 4`}</span>
      </div>
    </header>
  );
}

interface StepProps {
  state: OnboardingState;
  connect: (id: SourceId, detail?: string) => void;
  dispatch: (a: Parameters<typeof reducer>[1]) => void;
  next: () => void;
  back: () => void;
}

function StepBody({ state, connect, dispatch, next, back }: StepProps) {
  const ready = canAdvance(state);
  switch (state.step) {
    case 1:
      return (
        <>
          <Headline lead="Connect a trace source.">Cascade replays what you really ran. Point it at a run, or load the sample.</Headline>
          <StepSources sources={state.sources} connect={connect} disconnect={(id) => { dispatch({ type: "disconnect", id }); }} />
          <Actions>
            <PrimaryButton disabled={!ready} onClick={next}>
              Continue
            </PrimaryButton>
            {!ready && <QuietButton onClick={() => { connect("sample"); }}>Use the sample</QuietButton>}
          </Actions>
        </>
      );
    case 2: {
      const n = fleetSize(state.fleet);
      return (
        <>
          <Headline lead="Pick a fleet.">Start from the GPUs you have, then add the ones you don&rsquo;t.</Headline>
          <StepFleet fleet={state.fleet} setCount={(gpu, count) => { dispatch({ type: "setCount", gpu, count }); }} />
          <Actions>
            <PrimaryButton disabled={!ready} onClick={next}>
              {n === 0 ? "Continue" : `Continue with ${n} GPU${n === 1 ? "" : "s"}`}
            </PrimaryButton>
            <QuietButton onClick={back}>Back</QuietButton>
          </Actions>
        </>
      );
    }
    case 3:
      return (
        <>
          <Headline lead="Choose a router.">This decides which GPU each request lands on. You can change it any time.</Headline>
          <StepRouter
            fleet={state.fleet}
            router={state.router}
            engine={state.engine}
            choose={(router) => { dispatch({ type: "chooseRouter", router }); }}
            setEngine={(engine) => { dispatch({ type: "setEngine", engine }); }}
          />
          <Actions>
            <PrimaryButton disabled={!ready} onClick={next}>
              Run the replay
            </PrimaryButton>
            <QuietButton onClick={back}>Back</QuietButton>
          </Actions>
        </>
      );
    case 4:
      return (
        <>
          <Headline lead="Try it out.">Cascade is replaying your trace on the fleet you built.</Headline>
          <StepReplay state={state} />
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
          <Headline lead="You’re set up.">Cascade ran its first replay. Change the fleet and run it again whenever you like.</Headline>
          <StepReplay state={state} />
          <Actions>
            <PrimaryButton onClick={next}>Open Cascade</PrimaryButton>
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
