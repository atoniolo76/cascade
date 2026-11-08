import { useState, useSyncExternalStore, type ReactNode } from "react";
import { KeyRound } from "lucide-react";
import { checkToken, tokenStore, type TokenCheck } from "../api/client";
import { Button, Card } from "./ui";
import { BrandTile } from "./Logo";

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
    <div className="grid min-h-dvh place-items-center bg-window px-4">
      <Card className="w-full max-w-[380px] bg-panel p-7 shadow-tile">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
          className="flex flex-col gap-4">
          <div className="flex items-center gap-3">
            <BrandTile />
            <span className="text-[17px] font-medium tracking-tight text-fg">Cascade</span>
          </div>
          <div>
            <h1 className="text-[18px] leading-7 font-medium text-fg">Enter access token</h1>
            <p className="mt-1 text-[14px] text-muted">
              Paste the <code className="font-mono text-[13px] text-fg-2">CASCADE_TOKEN</code> for this deployment. It stays in this browser.
            </p>
          </div>
          <label className="flex flex-col gap-1.5">
            <span className="sr-only">Token</span>
            <div className="flex h-10 items-center gap-2 rounded-[12px] bg-surface px-3 ring-1 ring-border-strong transition-shadow duration-150 ring-inset focus-within:ring-2 focus-within:ring-accent">
              <KeyRound className="size-4 shrink-0 text-faint" strokeWidth={1.5} />
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
                className="h-full min-w-0 flex-1 bg-transparent font-mono text-[14px] text-fg outline-none placeholder:font-sans placeholder:text-faint"
              />
            </div>
            <span className="min-h-4 text-[12px] leading-4" style={{ color: "var(--st-bad-ink)" }}>
              {problem !== null ? PROBLEM[problem] : ""}
            </span>
          </label>
          <Button type="submit" variant="primary" disabled={checking || value.trim() === ""}>
            {checking ? "Checking…" : "Continue"}
          </Button>
        </form>
      </Card>
    </div>
  );
}
