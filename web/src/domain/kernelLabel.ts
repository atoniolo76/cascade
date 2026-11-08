import { kernelDisplayName } from "./kernelName";

/** The timeline's kernel label: the same simplified name the charts show. */
export function kernelLabel(name: string): string {
  return kernelDisplayName(name);
}
