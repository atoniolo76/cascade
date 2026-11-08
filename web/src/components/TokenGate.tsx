import { useState, useSyncExternalStore, type ReactNode } from "react";
import { KeyRound } from "lucide-react";
import { checkToken, tokenStore, type TokenCheck } from "../api/client";
import { Button, Card } from "./ui";
import { Logo } from "./Logo";

const PROBLEM: Record<Exclude<TokenCheck, "ok">, string> = {
  rejected: "That token was rejected.",
  unreachable: "Couldn't reach the Cascade API.",
};

export function TokenGate({ children }: { children: ReactNode }) {
  const token = useSyncExternalStore(tokenStore.subscribe, tokenStore.get);
  if (token !== null) return children;
  return <TokenForm />;
}

function TokenForm() {
  const [value, setValue] = useState("");
  const [checking, setChecking] = useState(false);
  const [problem, setProblem] = useState<Exclude<TokenCheck, "ok"> | null>(null);

  const submit = async () => {
    const token = value.trim();
    if (token === "") return;
    setChecking(true);
    const result = await checkToken(token);
    setChecking(false);
    if (result === "ok") tokenStore.set(token);
    else setProblem(result);
  };

  return (
    <div className="grid min-h-dvh place-items-center bg-bg px-4">
      <Card className="w-full max-w-[360px] p-6">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
          className="flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <Logo />
            <span className="text-[15px] font-semibold tracking-tight">Cascade</span>
          </div>
          <div>
            <h1 className="text-[16px] leading-6 font-semibold">Enter access token</h1>
            <p className="mt-1 text-[13px] text-muted">
              Paste the <code className="font-mono text-[12px] text-fg-2">CASCADE_TOKEN</code> for this deployment. It stays in this browser.
            </p>
          </div>
          <label className="flex flex-col gap-1.5">
            <span className="sr-only">Token</span>
            <div className="flex h-9 items-center gap-2 rounded-lg bg-surface px-3 ring-1 ring-border-strong transition-shadow duration-150 ring-inset focus-within:ring-2 focus-within:ring-accent">
              <KeyRound className="size-4 shrink-0 text-faint" />
              <input
                autoFocus
                type="password"
                autoComplete="off"
                spellCheck={false}
                value={value}
                onChange={(e) => {
                  setValue(e.target.value);
                  setProblem(null);
                }}
                placeholder="cascade token"
                className="h-full min-w-0 flex-1 bg-transparent font-mono text-[13px] outline-none placeholder:font-sans placeholder:text-faint"
              />
            </div>
            <span className="min-h-4 text-[12px] leading-4" style={{ color: "var(--st-bad-ink)" }}>
              {problem !== null ? PROBLEM[problem] : ""}
            </span>
          </label>
          <Button type="submit" variant="primary" disabled={checking || value.trim() === ""} className="h-9">
            {checking ? "Checking…" : "Continue"}
          </Button>
        </form>
      </Card>
    </div>
  );
}
