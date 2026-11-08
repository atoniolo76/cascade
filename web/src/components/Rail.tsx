import { Fragment, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { Link, useLocation } from "react-router";
import { Activity, ArrowUpRight, BookOpen, GitCompareArrows, Layers, LogOut, Monitor, Moon, PanelLeftClose, PanelLeftOpen, Plus, Sun, type LucideIcon } from "lucide-react";
import { tokenStore } from "../api/client";
import { useHealth } from "../api/queries";
import { DOCS_URL, REPO_URL } from "../config";
import { logoByKey } from "../domain/logos";
import { isTyping } from "../lib/keys";
import { isRailShortcut, railStore, type RailState } from "../lib/rail";
import { themeStore, useThemePref, type ThemePref } from "../lib/theme";
import { paths } from "../paths";
import { LogoGlyph, LogoTile } from "./BrandLogo";
import { BrandTile } from "./Logo";
import { ICON } from "./styles";
import { cx } from "./cx";

const RAIL_ID = "app-rail";
const DRAWER_ID = "app-rail-drawer";
const GITHUB = logoByKey("github");

type NavId = "traces" | "runs" | "compare" | "new";

function activeNav(pathname: string): NavId | null {
  if (pathname === "/runs/new") return "new";
  if (pathname.startsWith("/compare")) return "compare";
  if (pathname.startsWith("/runs")) return "runs";
  if (pathname === "/" || pathname.startsWith("/traces")) return "traces";
  return null;
}

const NAV: readonly { id: NavId; to: string; label: string; Icon: LucideIcon }[] = [
  { id: "traces", to: paths.traces, label: "Traces", Icon: Layers },
  { id: "runs", to: paths.runs, label: "Runs", Icon: Activity },
  { id: "compare", to: paths.compare(), label: "Compare runs", Icon: GitCompareArrows },
  { id: "new", to: paths.newRun(), label: "New run", Icon: Plus },
];

const THEME_META: Record<ThemePref, { value: string; Icon: LucideIcon }> = {
  system: { value: "System", Icon: Monitor },
  light: { value: "Light", Icon: Sun },
  dark: { value: "Dark", Icon: Moon },
};

const ITEM =
  "group relative flex h-10 shrink-0 items-center rounded-[12px] text-muted transition-[color,background-color,box-shadow] duration-150 hover:text-fg";
const ITEM_ACTIVE = "bg-surface text-fg shadow-tile ring-1 ring-border";
const ICON_BOX = "grid size-10 shrink-0 place-items-center [&>svg]:size-[19px]";
const FADE = "opacity-0 transition-opacity duration-150 group-data-[state=expanded]/rail:opacity-100 group-data-[state=expanded]/rail:delay-[140ms]";
const LABEL = cx("flex min-w-0 flex-1 items-center gap-2 overflow-hidden pr-3 text-left text-[14px] leading-5 font-medium whitespace-nowrap", FADE);

function Tip({ children }: { children: ReactNode }) {
  return (
    <span className="pointer-events-none absolute top-1/2 left-full z-50 ml-2.5 hidden -translate-y-1/2 items-center gap-1.5 rounded-[8px] bg-btn px-2 py-1 text-[12px] leading-4 font-medium whitespace-nowrap text-btn-fg opacity-0 transition-opacity duration-150 group-hover:opacity-100 group-focus-visible:opacity-100 md:flex md:group-data-[state=expanded]/rail:hidden">
      {children}
    </span>
  );
}

function Kbd({ children, className }: { children: string; className?: string }) {
  return <kbd className={cx("rounded-[5px] px-1.5 font-mono text-[11px] leading-4 ring-1 ring-current/25", className)}>{children}</kbd>;
}

function Label({ children, hint }: { children: string; hint?: ReactNode }) {
  return (
    <span aria-hidden className={LABEL}>
      <span className="min-w-0 flex-1 overflow-hidden">{children}</span>
      {hint}
    </span>
  );
}

const EXTERNAL_HINT = <ArrowUpRight aria-hidden className="size-3.5 shrink-0 text-faint" {...ICON} />;

function RailLink({ to, label, Icon, active }: { to: string; label: string; Icon: LucideIcon; active: boolean }) {
  return (
    <Link to={to} aria-label={label} aria-current={active ? "page" : undefined} className={cx(ITEM, active && ITEM_ACTIVE)}>
      <span className={ICON_BOX}>
        <Icon {...ICON} />
      </span>
      <Label>{label}</Label>
      <Tip>{label}</Tip>
    </Link>
  );
}

function ExternalLink({ href, label, icon }: { href: string; label: string; icon: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" aria-label={`${label} (opens in a new tab)`} className={ITEM}>
      <span className={ICON_BOX}>{icon}</span>
      <Label hint={EXTERNAL_HINT}>{label}</Label>
      <Tip>{label}</Tip>
    </a>
  );
}

function Divider() {
  return <span aria-hidden className="mx-2.5 h-px w-5 shrink-0 bg-border-strong transition-[width] duration-[180ms] ease-out group-data-[state=expanded]/rail:w-[calc(100%-20px)]" />;
}

function useIsTruncated<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [truncated, setTruncated] = useState(false);
  useLayoutEffect(() => {
    const el = ref.current;
    if (el === null) return;
    const measure = () => {
      setTruncated(el.clientWidth > 0 && el.scrollWidth > el.clientWidth);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => {
      observer.disconnect();
    };
  }, []);
  return [ref, truncated] as const;
}

function Workspace({ name, expanded }: { name: string; expanded: boolean }) {
  const [nameRef, truncated] = useIsTruncated<HTMLSpanElement>();
  return (
    <div role="img" aria-label={`Workspace: ${name}`} tabIndex={expanded ? undefined : 0} className="group relative mb-1 flex h-10 shrink-0 items-center rounded-[12px]">
      <span className="grid size-10 shrink-0 place-items-center">
        <LogoTile logo={undefined} fallback={name} size="lg" />
      </span>
      <span aria-hidden className={cx("flex min-w-0 flex-1 flex-col pr-3", FADE)}>
        <span ref={nameRef} title={truncated ? name : undefined} className="truncate text-[13px] leading-[18px] font-medium text-fg">
          {name}
        </span>
        <span className="truncate text-[11.5px] leading-4 text-muted">Sail workspace</span>
      </span>
      <Tip>{name}</Tip>
    </div>
  );
}

function useWorkspace() {
  const workspace = useHealth().data?.workspace ?? null;
  useEffect(() => {
    if (workspace !== null) document.title = `${workspace} · Cascade`;
  }, [workspace]);
  return workspace;
}

interface Toggle {
  label: string;
  shortcut?: string;
  expanded: boolean;
  controls: string;
  onClick: () => void;
}

function RailPanel({ id, label, state, toggle, className }: { id: string; label: string; state: RailState; toggle: Toggle; className?: string }) {
  const nav = activeNav(useLocation().pathname);
  const workspace = useWorkspace();
  const theme = THEME_META[useThemePref()];
  const expanded = state === "expanded";
  const ToggleIcon = toggle.expanded ? PanelLeftClose : PanelLeftOpen;
  return (
    <nav id={id} aria-label={label} data-state={state} className={cx("group/rail flex-col gap-2.5 px-4 py-4", className)}>
      <Link to={paths.traces} aria-label="Cascade home" className="mb-2 flex shrink-0 items-center gap-3 rounded-[12px]">
        <BrandTile />
        <span aria-hidden className={cx("min-w-0 overflow-hidden text-[15px] leading-5 font-semibold tracking-[-0.01em] whitespace-nowrap text-fg", FADE)}>
          Cascade
        </span>
      </Link>
      {workspace !== null && <Workspace name={workspace} expanded={expanded} />}
      {NAV.map((item) => (
        <Fragment key={item.id}>
          {item.id === "new" && <Divider />}
          <RailLink to={item.to} label={item.label} Icon={item.Icon} active={nav === item.id} />
        </Fragment>
      ))}
      <div className="mt-auto flex flex-col gap-2.5 pt-4">
        <ExternalLink href={REPO_URL} label="GitHub" icon={GITHUB ? <LogoGlyph logo={GITHUB} className="size-[18px]" /> : null} />
        <ExternalLink href={DOCS_URL} label="Docs" icon={<BookOpen {...ICON} />} />
        <button type="button" aria-label={`Theme: ${theme.value.toLowerCase()}`} onClick={themeStore.cycle} className={ITEM}>
          <span className={ICON_BOX}>
            <theme.Icon {...ICON} />
          </span>
          <Label hint={<span className="text-[12px] font-normal text-faint">{theme.value}</span>}>Theme</Label>
          <Tip>{`Theme: ${theme.value.toLowerCase()}`}</Tip>
        </button>
        <button type="button" aria-label="Forget token" onClick={tokenStore.clear} className={ITEM}>
          <span className={ICON_BOX}>
            <LogOut {...ICON} />
          </span>
          <Label>Forget token</Label>
          <Tip>Forget token</Tip>
        </button>
        <Divider />
        <button type="button" aria-label={toggle.label} aria-expanded={toggle.expanded} aria-controls={toggle.controls} onClick={toggle.onClick} className={ITEM}>
          <span className={ICON_BOX}>
            <ToggleIcon {...ICON} />
          </span>
          <Label hint={toggle.shortcut !== undefined && <Kbd className="font-normal text-faint">{toggle.shortcut}</Kbd>}>{toggle.label}</Label>
          <Tip>
            {toggle.label}
            {toggle.shortcut !== undefined && <Kbd className="opacity-70">{toggle.shortcut}</Kbd>}
          </Tip>
        </button>
      </div>
    </nav>
  );
}

function DesktopRail({ state }: { state: RailState }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!isRailShortcut(e, isTyping(e.target))) return;
      e.preventDefault();
      railStore.toggle();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
    };
  }, []);
  const expanded = state === "expanded";
  return (
    <RailPanel
      id={RAIL_ID}
      label="Main"
      state={state}
      toggle={{ label: expanded ? "Collapse sidebar" : "Expand sidebar", shortcut: "[", expanded, controls: RAIL_ID, onClick: railStore.toggle }}
      className="fixed inset-y-0 left-0 z-30 hidden w-(--rail-w) transition-[width] duration-[180ms] ease-out md:flex"
    />
  );
}

/** Mounted means open, like ui.tsx's Dialog. Following a link closes it. */
function Drawer({ onClose }: { onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (d !== null && !d.open) d.showModal();
  }, []);
  return (
    <dialog
      ref={ref}
      aria-label="Menu"
      onClose={onClose}
      onKeyDown={(e) => {
        e.stopPropagation();
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget || (e.target instanceof Element && e.target.closest("a") !== null)) onClose();
      }}
      className="m-0 h-dvh max-h-none w-[min(264px,calc(100vw-48px))] max-w-none animate-drawer-in rounded-r-[22px] bg-panel p-0 shadow-pop ring-1 ring-border backdrop:bg-black/30 md:hidden"
    >
      <RailPanel id={DRAWER_ID} label="Menu" state="expanded" toggle={{ label: "Close menu", expanded: true, controls: DRAWER_ID, onClick: onClose }} className="flex h-full" />
    </dialog>
  );
}

function PhoneBar() {
  const nav = activeNav(useLocation().pathname);
  const [open, setOpen] = useState(false);
  return (
    <header className="sticky top-0 z-30 flex h-14 items-center gap-1.5 bg-window/90 px-3 backdrop-blur-md md:hidden">
      <Link to={paths.traces} aria-label="Cascade home" className="mr-1 rounded-[12px]">
        <BrandTile />
      </Link>
      <nav aria-label="Main" className="flex items-center gap-1">
        {NAV.map((item) => (
          <Link
            key={item.id}
            to={item.to}
            aria-label={item.label}
            aria-current={nav === item.id ? "page" : undefined}
            className={cx(ITEM, nav === item.id && ITEM_ACTIVE)}
          >
            <span className={ICON_BOX}>
              <item.Icon {...ICON} />
            </span>
          </Link>
        ))}
      </nav>
      <button
        type="button"
        aria-label="Open menu"
        aria-expanded={open}
        aria-controls={DRAWER_ID}
        onClick={() => {
          setOpen(true);
        }}
        className={cx(ITEM, "ml-auto")}
      >
        <span className={ICON_BOX}>
          <PanelLeftOpen {...ICON} />
        </span>
      </button>
      {open && (
        <Drawer
          onClose={() => {
            setOpen(false);
          }}
        />
      )}
    </header>
  );
}

export function Rail({ state }: { state: RailState }) {
  return (
    <>
      <PhoneBar />
      <DesktopRail state={state} />
    </>
  );
}
