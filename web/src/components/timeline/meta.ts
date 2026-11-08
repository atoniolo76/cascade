import { gpuLogo, modelLogo, type Logo } from "../../domain/logos";
import type { Track } from "../../domain/timeline";

export interface TrackMeta {
  logo: Logo | undefined;
  sub?: string | undefined;
}

/** NVIDIA for simulated configs, the trace's model org on the real hardware track. */
export function trackMeta(track: Track, model: string | null, sub?: string): TrackMeta {
  const s = track.source;
  if (s.kind === "real") return { logo: modelLogo(model), sub: "measured" };
  return { logo: gpuLogo(s.gpuConfig), sub };
}
