/** What the send-a-trace command uses: the user's own value, else the app's reported endpoint. */
export function chooseEndpoint({
  typed,
  reported,
  fallback,
}: {
  typed: string | null;
  reported: string | null | undefined;
  fallback: string;
}): string {
  return typed ?? reported ?? fallback;
}
