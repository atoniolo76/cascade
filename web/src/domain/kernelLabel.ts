import { kernelDisplayName, simplifyKernelName } from "./kernelName";

/** The timeline's kernel label: the same simplified name the charts show. */
export function kernelLabel(name: string): string {
  return kernelDisplayName(name);
}

/** The function name alone, for labels inside narrow clips. */
export function kernelBase(name: string): string {
  return simplifyKernelName(name).base;
}
