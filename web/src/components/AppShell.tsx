import type { ReactNode } from "react";
import { Link, NavLink, Outlet } from "react-router";
import { LogOut, Plus } from "lucide-react";
import { tokenStore } from "../api/client";
import { Logo } from "./Logo";
import { cx } from "./cx";

const NAV = [
  { to: "/", label: "Traces", end: true },
  { to: "/runs", label: "Runs", end: false },
];

export function AppShell() {
  return (
    <div className="min-h-dvh">
      <header className="sticky top-0 z-30 border-b border-border bg-bg/85 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-[1440px] items-center gap-4 px-4 sm:px-6 lg:px-8">
          <Link to="/" className="flex items-center gap-2 rounded-md">
            <Logo />
            <span className="hidden text-[15px] font-semibold tracking-tight min-[400px]:inline">Cascade</span>
          </Link>
          <nav className="flex items-center gap-1">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  cx(
                    "rounded-md px-2.5 py-1.5 text-[13px] font-medium transition-colors duration-150",
                    isActive ? "bg-surface-2 text-fg" : "text-muted hover:text-fg",
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-1">
            <Link
              to="/runs/new"
              className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-accent px-3 text-[13px] font-medium text-accent-fg shadow-card transition-colors duration-150 hover:bg-accent-hover"
            >
              <Plus className="size-4" />
              <span className="hidden sm:inline">New run</span>
            </Link>
            <button
              type="button"
              title="Forget token"
              aria-label="Forget token"
              onClick={tokenStore.clear}
              className="grid size-8 place-items-center rounded-lg text-muted transition-colors duration-150 hover:bg-surface-2 hover:text-fg"
            >
              <LogOut className="size-4" />
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-[1440px] px-4 pt-6 pb-16 sm:px-6 lg:px-8">
        <Outlet />
      </main>
    </div>
  );
}

export function PageHeader({ title, subtitle, actions, eyebrow }: { title: ReactNode; subtitle?: ReactNode; actions?: ReactNode; eyebrow?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between sm:gap-6">
      <div className="min-w-0 flex-1">
        {eyebrow !== undefined && <div className="mb-1 text-[12px] font-medium text-muted">{eyebrow}</div>}
        <h1 className="text-[20px] leading-7 font-semibold tracking-tight">{title}</h1>
        {subtitle !== undefined && <div className="mt-1 text-[13px] text-muted">{subtitle}</div>}
      </div>
      {actions !== undefined && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}
