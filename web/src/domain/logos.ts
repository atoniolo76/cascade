import manifest from "../assets/logos/logos.json";

export interface Logo {
  key: string;
  name: string;
  color: string | null;
  mono: string;
  /** The color logo reads poorly on the white tile, so the tile draws the mono one. */
  tileMono: boolean;
}

const files = import.meta.glob<string>("../assets/logos/*.svg", { eager: true, query: "?url", import: "default" });
const fileUrl = (name: string) => files[`../assets/logos/${name}`];

const { _match: match, ...entries } = manifest;

const LOGOS = new Map<string, Logo>(
  Object.entries(entries).flatMap(([key, e]) => {
    const mono = fileUrl(e.mono);
    if (mono === undefined) return [];
    const color = e.color === null ? null : (fileUrl(e.color) ?? null);
    return [[key, { key, name: e.name, color, mono, tileMono: "tile" in e && e.tile === "mono" }]];
  }),
);

export function logoByKey(key: string): Logo | undefined {
  return LOGOS.get(key);
}

const MODEL_ORGS = Object.entries(match.model_org);
const GPU_PREFIXES = Object.entries(match.gpu);

/** A Hugging Face model id like `meta-llama/Llama-3.1-8B` resolves by its org, ignoring case. */
export function modelLogo(modelId: string | null): Logo | undefined {
  const org = modelId?.split("/")[0]?.toLowerCase();
  if (org === undefined || org === "") return undefined;
  const hit = MODEL_ORGS.find(([o]) => o.toLowerCase() === org);
  return hit === undefined ? undefined : LOGOS.get(hit[1]);
}

/** A GPU name ("NVIDIA H100 80GB HBM3") or a sim config ("SM90_H100") resolves by its vendor prefix. */
export function gpuLogo(name: string | null): Logo | undefined {
  const upper = name?.toUpperCase();
  if (upper === undefined) return undefined;
  const hit = GPU_PREFIXES.find(([prefix]) => upper.startsWith(prefix.toUpperCase()));
  return hit === undefined ? undefined : LOGOS.get(hit[1]);
}
