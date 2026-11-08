/** The Get started page links to `<app>/#token=<token>`. The fragment never reaches a server. */
export function splitTokenFragment(hash: string): { token: string | null; rest: string } {
  const params = new URLSearchParams(hash.replace(/^#/, ""));
  const token = params.get("token")?.trim() ?? "";
  if (token === "") return { token: null, rest: hash };
  params.delete("token");
  const rest = params.toString();
  return { token, rest: rest === "" ? "" : `#${rest}` };
}

interface UrlParts {
  pathname: string;
  search: string;
  hash: string;
}

interface History {
  readonly state: unknown;
  replaceState(data: unknown, unused: string, url: string): void;
}

/** Takes the token out of the URL so it is not left in history, bookmarks or screenshots. */
export function takeTokenFromUrl(location: UrlParts, history: History): string | null {
  const { token, rest } = splitTokenFragment(location.hash);
  if (token === null) return null;
  history.replaceState(history.state, "", `${location.pathname}${location.search}${rest}`);
  return token;
}
