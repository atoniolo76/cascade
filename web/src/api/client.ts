const TOKEN_KEY = "cascade.token";
const listeners = new Set<() => void>();

function readToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

function writeToken(token: string | null) {
  try {
    if (token === null) localStorage.removeItem(TOKEN_KEY);
    else localStorage.setItem(TOKEN_KEY, token);
  } catch {
    // Storage blocked: the token lives only as long as this tab.
  }
  memoryToken = token;
  for (const fn of listeners) fn();
}

let memoryToken = readToken();

export const tokenStore = {
  get: (): string | null => memoryToken,
  set: (token: string) => {
    writeToken(token);
  },
  clear: () => {
    writeToken(null);
  },
  subscribe: (fn: () => void) => {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },
};

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function detailMessage(body: unknown, fallback: string): string {
  if (typeof body !== "object" || body === null || !("detail" in body)) return fallback;
  const { detail } = body;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const msgs = detail.flatMap((d: unknown) =>
      typeof d === "object" && d !== null && "msg" in d && typeof d.msg === "string" ? [d.msg] : [],
    );
    if (msgs.length > 0) return msgs.join("; ");
  }
  return fallback;
}

export async function request(path: string, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers);
  const token = tokenStore.get();
  if (token !== null) headers.set("Authorization", `Bearer ${token}`);
  if (init?.body !== undefined) headers.set("Content-Type", "application/json");
  const res = await fetch(path, { ...init, headers });
  if (res.ok) return res;
  if (res.status === 401) tokenStore.clear();
  const body: unknown = await res.json().catch(() => null);
  throw new ApiError(res.status, detailMessage(body, `${res.status} ${res.statusText}`));
}

/** The server is the typed pydantic side of the contract in types.ts; its JSON is trusted as T. */
export async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await request(path, init);
  const body: unknown = await res.json();
  return body as T;
}

export async function apiText(path: string): Promise<string> {
  const res = await request(path);
  return res.text();
}

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(url);
  }, 1000);
}

export function eventsUrl(runId: string): string {
  return `/api/runs/${encodeURIComponent(runId)}/events?token=${encodeURIComponent(tokenStore.get() ?? "")}`;
}

export type TokenCheck = "ok" | "rejected" | "unreachable";

export async function checkToken(token: string): Promise<TokenCheck> {
  try {
    const res = await fetch("/api/gpu-configs", { headers: { Authorization: `Bearer ${token}` } });
    if (res.ok) return "ok";
    return res.status === 401 ? "rejected" : "unreachable";
  } catch {
    return "unreachable";
  }
}

export async function downloadFile({ path, filename }: { path: string; filename: string }) {
  const res = await request(path);
  saveBlob(await res.blob(), filename);
}

export function saveText({ text, filename, type }: { text: string; filename: string; type: string }) {
  saveBlob(new Blob([text], { type }), filename);
}
