import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { KernelStat, SimState, SimSummary } from "../api/types";
import {
  alignScale,
  buildRunTracks,
  buildTimeline,
  clipIndexAt,
  clipSpan,
  formatTimecode,
  groupKernels,
  kernelRange,
  rangeTime,
  readTrack,
  shuttle,
  stepBoundary,
  boundaries,
  ticks,
  type RunTimelineInput,
  type Track,
} from "./timeline.ts";

function sim(id: string, gpu_config: string, replica: number, state: SimState = "succeeded"): SimSummary {
  return {
    id,
    run_id: "run-a",
    gpu_config,
    replica,
    state,
    error: null,
    launcher_ref: null,
    created_at: "2026-11-07T00:00:00Z",
    started_at: null,
    finished_at: null,
    last_heartbeat_at: null,
    exit_code: null,
    kernels_done: 0,
    kernels_total: 4,
    current_kernel: null,
    latest: { gpu_tot_sim_cycle: null, gpu_tot_sim_insn: null, gpu_tot_ipc: null, sim_rate_ips: null },
    log_tail: [],
  };
}

function stat(sim_id: string, kernel_index: number, ns: number | null): KernelStat {
  return {
    sim_id,
    kernel_index,
    kernel_name: `k${kernel_index}`,
    kernel_launch_uid: kernel_index + 1,
    gpu_sim_cycle: ns === null ? null : ns * 2,
    gpu_sim_insn: 100,
    gpu_ipc: 1.5,
    sim_time_ns: ns,
    stats: {},
  };
}

function input(sims: SimSummary[], stats: KernelStat[], real: { kernel_index: number; real_ns: number | null }[] | null = null): RunTimelineInput {
  return {
    run: { id: "run-a", trace_id: "t1", entries: [{ gpu_config: "H100", count: 3 }, { gpu_config: "H200", count: 1 }], sims },
    trace: { id: "t1", model: "facebook/opt-125m", source_gpu: "NVIDIA H100" },
    kernels: groupKernels(stats),
    real: real?.map((r) => ({ ...r, kernel_name: `k${r.kernel_index}`, sim_ns: {} })) ?? null,
    colorOf: (c) => `color-${c}`,
  };
}

const byId = (tracks: Track[], id: string) => {
  const t = tracks.find((x) => x.id === id);
  assert.ok(t, `track ${id} exists`);
  return t;
};

describe("buildRunTracks", () => {
  it("lays a sim's kernels end to end in kernel order", () => {
    const tracks = buildRunTracks(input([sim("s1", "H100", 1)], [stat("s1", 1, 30), stat("s1", 0, 10), stat("s1", 2, 5)]));
    const t = byId(tracks, "sim:s1");
    assert.deepEqual(
      t.clips.map((c) => [c.kernelIndex, c.start, c.end]),
      [
        [0, 0, 10],
        [1, 10, 40],
        [2, 40, 45],
      ],
    );
    assert.equal(t.total, 45);
  });

  it("uses the per-kernel median across replicas for the config track", () => {
    const sims = [sim("a", "H100", 1), sim("b", "H100", 2), sim("c", "H100", 3)];
    const stats = [stat("a", 0, 10), stat("b", 0, 40), stat("c", 0, 20), stat("a", 1, 7), stat("b", 1, 9)];
    const t = byId(buildRunTracks(input(sims, stats)), "cfg:run-a:H100");
    assert.deepEqual(
      t.clips.map((c) => c.end - c.start),
      [20, 8],
    );
  });

  it("ends a running sim with a pending clip sized like the same kernel elsewhere", () => {
    const sims = [sim("a", "H100", 1, "running"), sim("b", "H100", 2)];
    const stats = [stat("a", 0, 10), stat("b", 0, 12), stat("b", 1, 50)];
    const tracks = buildRunTracks(input(sims, stats));
    const a = byId(tracks, "sim:a");
    assert.equal(a.total, 10);
    const last = a.clips.at(-1);
    assert.deepEqual([last?.pending, last?.kernelIndex, last?.start, last?.end], [true, 1, 10, 60]);
    assert.equal(byId(tracks, "sim:b").clips.at(-1)?.pending, undefined);
    assert.equal(byId(tracks, "cfg:run-a:H100").clips.at(-1)?.pending, true);
  });

  it("adds a real hardware track only when real timing exists", () => {
    assert.equal(buildRunTracks(input([sim("a", "H100", 1)], [])).some((t) => t.source.kind === "real"), false);
    const tracks = buildRunTracks(input([sim("a", "H100", 1)], [], [{ kernel_index: 0, real_ns: 5 }, { kernel_index: 1, real_ns: null }]));
    const real = byId(tracks, "real:t1");
    assert.deepEqual(
      real.clips.map((c) => [c.start, c.end]),
      [[0, 5]],
    );
  });

  it("skips kernels with no simulated time but keeps later indexes", () => {
    const t = byId(buildRunTracks(input([sim("a", "H100", 1)], [stat("a", 0, 10), stat("a", 1, null), stat("a", 2, 4)])), "sim:a");
    assert.deepEqual(
      t.clips.map((c) => c.kernelIndex),
      [0, 2],
    );
  });
});

describe("buildTimeline", () => {
  it("shows real hardware once for two runs of the same trace", () => {
    const real = [{ kernel_index: 0, real_ns: 5 }];
    const a = input([sim("a", "H100", 1)], [stat("a", 0, 10)], real);
    const b = { ...input([sim("b", "H200", 1)], [stat("b", 0, 40)], real), run: { ...a.run, id: "run-b", sims: [sim("b", "H200", 1)] } };
    const tl = buildTimeline([a, b]);
    assert.equal(tl.tracks.filter((t) => t.source.kind === "real").length, 1);
    assert.equal(tl.duration, 40);
  });
});

describe("lookups", () => {
  const track = byId(buildRunTracks(input([sim("a", "H100", 1)], [stat("a", 0, 10), stat("a", 1, 30), stat("a", 2, 5)])), "sim:a");

  it("finds the clip under a time", () => {
    assert.equal(clipIndexAt(track.clips, -1), -1);
    assert.equal(clipIndexAt(track.clips, 0), 0);
    assert.equal(clipIndexAt(track.clips, 10), 1);
    assert.equal(clipIndexAt(track.clips, 39.9), 1);
    assert.equal(clipIndexAt(track.clips, 1000), 2);
  });

  it("reads the playhead in time mode", () => {
    const align = alignScale([track]);
    const r = readTrack(track, 25, "time", align);
    assert.equal(r?.clip.kernelIndex, 1);
    assert.equal(r.fraction, 0.5);
    assert.equal(readTrack(track, 99, "time", align)?.done, true);
  });

  it("aligns kernel i to slot i and fills it by relative duration", () => {
    const other = byId(buildRunTracks(input([sim("b", "H100", 1)], [stat("b", 0, 20), stat("b", 1, 15)])), "sim:b");
    const align = alignScale([track, other]);
    assert.equal(align.slots, 3);
    const c0 = track.clips[0];
    assert.ok(c0);
    assert.deepEqual(clipSpan(c0, "align", align), [0, 0.5]);
    const r = readTrack(track, 0.25, "align", align);
    assert.equal(r?.fraction, 0.5);
    assert.equal(r.elapsed, 5);
  });

  it("steps between kernel boundaries", () => {
    const pts = boundaries(track, "time");
    assert.deepEqual(pts, [0, 10, 40, 45]);
    assert.equal(stepBoundary(pts, 10, 1), 40);
    assert.equal(stepBoundary(pts, 10, -1), 0);
    assert.equal(stepBoundary(pts, 45, 1), null);
  });

  it("sums the time each track spends on an in/out kernel range", () => {
    const align = alignScale([track]);
    const range = kernelRange(track, 5, 41, "time", align);
    assert.deepEqual(range, [1, 2]);
    assert.deepEqual(rangeTime(track, [1, 2]), { ns: 35, kernels: 2 });
  });
});

describe("ruler and readouts", () => {
  it("picks 1-2-5 steps and the unit that fits the step", () => {
    const t = ticks({ start: 0, end: 10_000 }, 1000, "time", 100);
    assert.deepEqual(
      t.major.slice(0, 3).map((m) => m.label),
      ["0 µs", "1 µs", "2 µs"],
    );
    const fine = ticks({ start: 0, end: 500 }, 1000, "time", 100);
    assert.equal(fine.major[1]?.label, "50 ns");
  });

  it("labels kernel slots in align mode", () => {
    const t = ticks({ start: 0, end: 40 }, 800, "align", 100);
    assert.deepEqual(
      t.major.slice(0, 3).map((m) => m.label),
      ["#0", "#5", "#10"],
    );
  });

  it("shows as many timecode decimals as a pixel resolves", () => {
    assert.equal(formatTimecode(1_234_567, 1000), "1,235 µs");
    assert.equal(formatTimecode(1_234_567, 1), "1,234.567 µs");
    assert.equal(formatTimecode(1_234_567, 50), "1,234.57 µs");
  });

  it("shuttles like an editor", () => {
    assert.equal(shuttle(0, "l"), 1);
    assert.equal(shuttle(1, "l"), 2);
    assert.equal(shuttle(8, "l"), 8);
    assert.equal(shuttle(4, "j"), -1);
    assert.equal(shuttle(-1, "j"), -2);
    assert.equal(shuttle(-2, "k"), 0);
  });
});
