import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { gzipSync } from "node:zlib";
import type { RunFile } from "../api/types";
import { groupRunFiles, isPreviewable, readHead } from "./files.ts";

const text = (n: number) => Array.from({ length: n }, (_, i) => `row ${i + 1}`).join("\n") + "\n";

const blob = (s: string | Uint8Array<ArrayBuffer>) => new Blob([s]);
const gz = (s: string | Uint8Array<ArrayBuffer>, level = 6) => new Uint8Array(gzipSync(s, { level }));

describe("readHead", () => {
  it("returns a short file whole", async () => {
    assert.deepEqual(await readHead(blob("a,b\n1,2\n3,4"), { gzip: false }), { lines: ["a,b", "1,2", "3,4"], more: false });
  });

  it("stops after maxLines and says there is more", async () => {
    const head = await readHead(blob(text(500)), { gzip: false, maxLines: 200 });
    assert.equal(head.lines.length, 200);
    assert.equal(head.lines[199], "row 200");
    assert.equal(head.more, true);
  });

  it("does not claim more for a file of exactly maxLines lines", async () => {
    const head = await readHead(blob(text(200)), { gzip: false, maxLines: 200 });
    assert.equal(head.lines.length, 200);
    assert.equal(head.more, false);
  });

  it("decompresses gzip and strips CRLF", async () => {
    assert.deepEqual(await readHead(blob(gz("x\r\ny\r\n")), { gzip: true }), { lines: ["x", "y"], more: false });
  });

  it("reads every member of a multi-member gzip, past a header-looking run inside one", async () => {
    const fake = new Uint8Array([0x78, 0x1f, 0x8b, 0x08, 0x00, 0x79, 0x0a]);
    const file = new Blob([gz("head\nsplit "), gz(fake, 0), gz(text(3))]);
    assert.deepEqual(await readHead(file, { gzip: true }), { lines: ["head", "split x\x1f\ufffd\x08\x00y", "row 1", "row 2", "row 3"], more: false });
  });

  it("stops inflating members once it has enough lines", async () => {
    const file = new Blob(Array.from({ length: 300 }, (_, i) => gz(`line ${i}\n`)));
    const head = await readHead(file, { gzip: true, maxLines: 200 });
    assert.equal(head.lines.length, 200);
    assert.equal(head.lines[199], "line 199");
    assert.equal(head.more, true);
  });

  it("clips a very long line", async () => {
    const { lines } = await readHead(blob(`${"9,".repeat(5000)}\n`), { gzip: false });
    assert.equal(lines.length, 1);
    assert.equal(lines[0], `${"9,".repeat(1000)} …`);
  });
});

describe("isPreviewable", () => {
  it("takes small text, csv and log files, gzipped or not", () => {
    for (const name of ["kernel-stats.csv", "exit-status.txt", "accelsim.log.gz", "perf_counter.csv.gz"]) assert.ok(isPreviewable({ name, size: 1024 }), name);
    assert.ok(!isPreviewable({ name: "opt125m_H100_perfetto.json.gz", size: 1024 }));
    assert.ok(!isPreviewable({ name: "perf_counter.csv.gz", size: 17 * 1024 * 1024 }));
  });
});

describe("groupRunFiles", () => {
  const f = (gpu_config: string, replica: number, name: string, size = 10): RunFile => ({
    name,
    size,
    content_type: "text/plain",
    created_at: "2026-11-07T00:00:00Z",
    sim_id: `sim-${gpu_config}-${replica}`,
    gpu_config,
    replica,
  });

  it("groups by the run's config order, then replica", () => {
    const groups = groupRunFiles([f("A100", 1, "x.txt"), f("H100", 2, "a.txt"), f("H100", 1, "a.txt", 5), f("H100", 1, "b.txt", 7)], ["H100", "A100"]);
    assert.deepEqual(
      groups.map((g) => [g.config, g.replicas.map((r) => [r.replica, r.files.map((x) => x.name), r.bytes])]),
      [
        ["H100", [[1, ["a.txt", "b.txt"], 12], [2, ["a.txt"], 10]]],
        ["A100", [[1, ["x.txt"], 10]]],
      ],
    );
  });
});
