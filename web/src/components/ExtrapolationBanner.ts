import { createElement } from "react";
import { EXTRAPOLATION_BANNER } from "../domain/scenario.ts";

export function ExtrapolationBanner({ counts }: { counts: Record<string, number> }) {
  if (!Object.entries(counts).some(([key, count]) => key.startsWith("extrapolated") && count > 0)) return null;
  return createElement("div", { role: "alert", className: "rounded-[14px] border border-amber-400/50 bg-amber-400/10 px-4 py-3 text-[15px] font-medium text-fg" }, EXTRAPOLATION_BANNER);
}
