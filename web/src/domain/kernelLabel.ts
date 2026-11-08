/** The one place clip labels shorten kernel names: the nested names of an Itanium-mangled `_ZN…` name, otherwise the name as is. */
export function kernelLabel(name: string): string {
  if (!name.startsWith("_ZN")) return name;
  const parts: string[] = [];
  let i = 3;
  while (i < name.length) {
    const m = /^\d+/.exec(name.slice(i));
    if (!m) break;
    const len = Number(m[0]);
    i += m[0].length;
    parts.push(name.slice(i, i + len));
    i += len;
  }
  return parts.length > 0 ? parts.join("::") : name;
}
