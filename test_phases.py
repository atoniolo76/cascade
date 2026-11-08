"""CPU-only native observability checks: python3 -m unittest -v test_phases."""

import json
import os
import runpy
import sqlite3
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import accelsim_trace as a


class PhasesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.home())
        self.work = Path(self.temp.name)
        self.db = self.work / "profile.sqlite"
        with sqlite3.connect(self.db) as con:
            con.executescript("""
                CREATE TABLE StringIds(id INTEGER, value TEXT);
                INSERT INTO StringIds VALUES(1,'kernel'),(2,'cudaLaunchKernel'),
                    (3,'layer [1,768]');
                CREATE TABLE NVTX_EVENTS(start,end,text,textId,globalTid,domainId);
                INSERT INTO NVTX_EVENTS VALUES(100,200,'vllm_step_2',NULL,16777223,1),
                    (110,190,NULL,3,16777223,2);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
                    start,end,correlationId,nameId,globalTid);
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
                    (90,120,1,2,16777223),(110,130,2,2,16777223),
                    (100,200,3,2,16777224);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                    start,end,streamId,deviceId,correlationId,globalPid,
                    gridX,gridY,gridZ,blockX,blockY,blockZ,registersPerThread,
                    staticSharedMemory,dynamicSharedMemory,mangledName,shortName);
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                    (120,150,1,0,1,16777216,1,1,1,32,1,1,32,0,0,1,1);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(
                    start,end,streamId,deviceId,correlationId,globalPid,bytes,value);
                INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES
                    (140,160,1,0,2,16777216,128,0);
                CREATE TABLE META_DATA_CAPTURE(name TEXT, value TEXT, other INTEGER);
                INSERT INTO META_DATA_CAPTURE VALUES
                    ('PROCESS_0:ENVIRONMENT_VARIABLE:HF_TOKEN','secret',1),
                    ('another','ENVIRONMENT_VARIABLE:OTHER',2),('version','ok',3);
            """)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def fake_log():
        steps = [{"step": i, "repeat": i // 2, "warmup": i < 2,
                  "wall_start_ns": 1000 + i * 200, "wall_end_ns": 1100 + i * 200}
                 for i in range(4)]
        log = "\n".join(
            f"INFO Iteration({i}): {int(i % 2 == 0)} context requests, 6 context tokens, "
            f"{i % 2} generation requests, 1 generation tokens, iteration elapsed time: 0.10 ms"
            for i in range(4))
        return log, steps

    def test_parser_and_mismatch(self):
        log, steps = self.fake_log()
        # Exercise real JSON step-log serialization as well as log text.
        path = self.work / "steps.json"
        path.write_text(json.dumps(steps))
        rows, diag = a._parse_iterations(log, json.loads(path.read_text()))
        self.assertFalse(diag["iteration_step_mismatch"])
        self.assertEqual([r["kind"] for r in rows], ["prefill", "decode"] * 2)
        self.assertEqual(rows[2]["wall_ns"], 100)
        self.assertEqual(rows[2]["repeat"], 1)
        self.assertTrue(a._parse_iterations(log, steps[:-1])[1]["iteration_step_mismatch"])
        self.assertTrue(a._parse_iterations(log + log, steps)[1]["iteration_step_mismatch"])
        self.assertEqual(a._parse_iterations("", []), ([], {
            "iteration_count": 0, "step_count": 0, "iteration_step_mismatch": False}))

    def test_exports_and_reconciliation(self):
        nvtx = a._nsys_nvtx(self.db)
        self.assertEqual(nvtx[1]["text"], "layer [1,768]")
        steps = a._nsys_steps(self.db)
        kernels = a._nsys_kernels(self.db)
        a._assign_steps(kernels, steps)
        events = a._nsys_events(self.db, 0, steps)
        apis = a._nsys_api_calls(self.db, 0, steps)
        stats = a._phase_nsys_stats(nvtx, events, apis)
        self.assertEqual(stats[2], {"gpu_busy_ns": 40, "cuda_api_ns": 30,
                                    "other_host_ns": 70, "nsys_span_ns": 100})
        rows, _ = a._parse_iterations(*self.fake_log())
        profiled = [{**r, "wall_ns": 120, "elapsed_ms": .12} for r in rows]
        table = a._reconcile({"baseline": rows, "profiled": profiled}, stats)
        self.assertEqual(table[0]["profiled_gpu_busy_ns_n"], 1)
        self.assertEqual(table[1]["profiled_gpu_busy_ns_n"], 0)
        self.assertEqual(table[0]["delta_wall_ns_mean_ns"], 20)
        self.assertAlmostEqual(table[0]["delta_elapsed_ms_mean_ns"], 20000)
        self.assertEqual(table[0]["baseline_wall_ns_n"], 1)
        self.assertEqual(a._distribution([0, 100])["p95"], 95)
        a._write_csv(self.work / "reconcile.csv", table)
        self.assertIn("delta_wall_ns_mean_ns", (self.work / "reconcile.csv").read_text())
        self.assertIsNone(a._reconcile({}, {})[0]["delta_wall_ns_mean_pct"])
        self.assertEqual(a._phase_nsys_stats([], [], []), {})

    def test_scrubbing_and_missing_table(self):
        rep = self.work / "profile.nsys-rep"
        rep.write_text("unscrubbed")
        a._scrub_nsys(self.work)
        self.assertFalse(rep.exists())
        with sqlite3.connect(self.db) as con:
            self.assertEqual(con.execute("SELECT * FROM META_DATA_CAPTURE").fetchall(),
                             [("version", "ok", 3)])
            con.execute("DROP TABLE META_DATA_CAPTURE")
        a._scrub_nsys(self.work)
        rep.write_text("unscrubbed")
        with patch("builtins.print") as warn:
            a._scrub_nsys(self.work, True)
            self.assertIn("unscrubbed environment", warn.call_args.args[0])
        self.assertTrue(rep.exists())

    def test_environment(self):
        with patch.dict(os.environ, {"MODAL_ID": "x", "some_secret": "x", "API_KEY": "x",
                                    "HF_TOKEN": "hf", "HUGGING_FACE_HUB_TOKEN": "hf2",
                                    "CUDA_INJECTION64_PATH": "nvbit"}, clear=True):
            env = a._nsys_env()
        self.assertFalse(set(env) & {"MODAL_ID", "some_secret", "API_KEY", "CUDA_INJECTION64_PATH"})
        self.assertEqual(env["HF_TOKEN"], "hf")
        self.assertEqual(env["HUGGING_FACE_HUB_TOKEN"], "hf2")

    def test_nsys_flags(self):
        with patch.object(a.subprocess, "run") as run:
            run.return_value.stdout = "--cuda-event-trace --cuda-graph-trace"
            cmd = a._nsys_base_cmd(capture_range=True, cuda_events=False)
            self.assertIn("--cuda-event-trace=false", cmd)
            self.assertIn("--cuda-graph-trace=node", cmd)
            self.assertIn("--capture-range=cudaProfilerApi", cmd)
            self.assertIn("--capture-range-end=stop", cmd)
            self.assertIn("--cuda-event-trace=true", a._nsys_base_cmd())
            run.return_value.stdout = ""
            self.assertNotIn("--cuda-event-trace=false", a._nsys_base_cmd(cuda_events=False))

    def test_clock_thread_can_stop(self):
        sampler = a.ClockSampler(self.work / "clocks.csv")
        with patch.object(sampler, "run", return_value=None):
            sampler.start()
            sampler.stop()
        self.assertTrue((self.work / "clocks.csv").exists())

    def test_corrected_timeline(self):
        log, baseline = self.fake_log()
        iterations, _ = a._parse_iterations(log, baseline)
        # Different warmup length shifts global IDs; repeat-local positions still match.
        profiled = [{"step": 10, "repeat": 0}, {"step": 20, "repeat": 1},
                    {"step": 21, "repeat": 1}]
        nvtx = a._nsys_nvtx(self.db)
        # The fake sqlite's captured step 2 corresponds to profiled global step 20.
        nvtx[0]["text"] = "vllm_step_20"
        events = a._nsys_events(self.db, 0, {})
        apis = a._nsys_api_calls(self.db, 0, {})
        stats = a._phase_nsys_stats(nvtx, events, apis)
        ranges = {20: {"start": 100, "end": 200}}
        kernels = a._nsys_kernels(self.db)
        kernels.append({**kernels[0], "start": 115, "end": 125, "grid": "2x3x4"})
        metrics = {"requests": [{"repeat": 1, "metrics": {
            "queued_ts": 1, "scheduled_ts": 2, "first_token_ts": 3, "last_token_ts": 5,
            "num_generation_tokens": 3}}, {"repeat": 0, "metrics": {
                "queued_ts": 1, "scheduled_ts": 1, "first_token_ts": 2, "last_token_ts": 2,
                "num_generation_tokens": 1}}, {"repeat": 2, "metrics": None}]}
        meta = {"run_name": "fake", "model": "opt", "prompt": "hello", "max_tokens": 3,
                "repeats": 2, "exec_mode": "eager", "gpu_name": "H100",
                "sm_clock_mhz_median": 1980, "mem_clock_mhz": 2619,
                "vllm_version": "0.21.0", "layerwise_nvtx": False}
        result = a._corrected_timeline(meta, baseline, profiled, iterations, metrics,
                                       stats, ranges, kernels)
        self.assertEqual(set(result), {"meta", "steps", "requests", "summary"})
        self.assertEqual(set(result["meta"]), set(meta) | {"clock"})
        step_keys = {"step", "repeat", "warmup", "kind", "num_reqs", "num_scheduled_tokens",
                     "baseline_wall_ns", "profiled_wall_ns", "gpu_busy_ns", "cuda_api_ns",
                     "host_ns", "kernels", "baseline_wall_start_ns", "baseline_wall_end_ns",
                     "baseline_gap_before_ns", "profiled_wall_start_ns", "profiled_wall_end_ns",
                     "profiled_gap_before_ns"}
        for s in result["steps"]:
            self.assertEqual(set(s), step_keys)
        for s in result["steps"][:2] + result["steps"][3:]:
            for field in ("profiled_wall_ns", "gpu_busy_ns", "cuda_api_ns", "host_ns"):
                self.assertIsNone(s[field])
            self.assertEqual(s["kernels"], [])
        captured = result["steps"][2]
        self.assertEqual(captured["step"], 2)
        self.assertEqual(captured["profiled_wall_ns"], 100)
        self.assertEqual(captured["host_ns"], 60)
        self.assertEqual(captured["cuda_api_ns"], 30)
        self.assertEqual(captured["num_reqs"], 1)
        self.assertEqual(captured["num_scheduled_tokens"], 7)
        self.assertEqual([k["seq"] for k in captured["kernels"]], [0, 1])
        self.assertEqual([k["start_offset_ns"] for k in captured["kernels"]], [15, 20])
        self.assertEqual(captured["kernels"][0]["grid"], [2, 3, 4])
        self.assertEqual(captured["kernels"][0]["block"], [32, 1, 1])
        self.assertEqual(captured["kernels"][0]["duration_ns"], 10)
        self.assertEqual(set(captured["kernels"][0]), {
            "seq", "name", "grid", "block", "duration_ns", "start_offset_ns"})
        req = result["requests"][0]
        self.assertEqual(set(req), {"repeat", "warmup", "queued_ms", "ttft_ms", "itl_ms_mean",
                                    "itl_ms_p50", "e2e_ms", "num_output_tokens",
                                    "first_step_start_ns", "last_step_end_ns", "arrival_ns",
                                    "queued_ns", "first_token_ns", "last_token_ns",
                                    "pre_first_step_ns", "post_last_step_ns"})
        self.assertEqual((req["queued_ms"], req["ttft_ms"], req["itl_ms_mean"], req["e2e_ms"]),
                         (1000, 2000, 1000, 4000))
        self.assertIsNone(req["itl_ms_p50"])
        self.assertTrue(result["requests"][1]["warmup"])
        self.assertIsNone(result["requests"][1]["itl_ms_mean"])
        self.assertIsNone(result["requests"][2]["ttft_ms"])
        summary = result["summary"]["prefill"]
        self.assertEqual(set(summary), {"n", "baseline_wall_ns", "gpu_busy_ns", "host_ns",
                                        "profiler_overhead_pct", "baseline_gap_before_ns",
                                        "profiled_gap_before_ns"})
        self.assertEqual(summary["baseline_wall_ns"], {"mean": 100, "p50": 100, "p95": 100})
        self.assertEqual(summary["host_ns"], {"mean": 60, "p50": 60})
        self.assertEqual(summary["gpu_busy_ns"], {"mean": 40, "p50": 40})
        self.assertEqual(summary["profiler_overhead_pct"], 0)
        self.assertIsNone(result["summary"]["decode"]["host_ns"]["mean"])
        self.assertEqual(result["summary"]["decode"]["baseline_wall_ns"]["mean"], 100)
        shorter = [{**r, "wall_end_ns": r["wall_start_ns"] + 80} for r in baseline]
        changed = a._corrected_timeline(meta, shorter, profiled, iterations, metrics,
                                        stats, ranges, kernels)
        self.assertEqual(changed["summary"]["prefill"]["profiler_overhead_pct"], 25)
        self.assertEqual(changed["steps"][2]["host_ns"], 40)
        baseline[0].update(num_reqs=2, num_scheduled_tokens=9, kind="mixed")
        fallback = a._corrected_timeline(meta, baseline, profiled, [], {}, {}, {}, [])
        self.assertEqual(fallback["steps"][0]["kind"], "mixed")
        self.assertEqual(fallback["steps"][0]["num_reqs"], 2)
        self.assertEqual(fallback["steps"][0]["num_scheduled_tokens"], 9)
        empty = a._corrected_timeline(meta, [], [], [], {}, {}, {}, [])
        self.assertEqual(empty, {"meta": result["meta"], "steps": [], "requests": [], "summary": {}})
        with self.assertRaises(ValueError):
            a._corrected_timeline(meta, baseline, profiled, iterations, {}, stats, ranges,
                                  [{**kernels[0], "grid": "invalid"}])

    def test_corrected_timeline_monotonic_gaps(self):
        meta = {k: None for k in ("run_name", "model", "prompt", "max_tokens", "repeats",
                                  "exec_mode", "gpu_name", "sm_clock_mhz_median", "mem_clock_mhz",
                                  "vllm_version", "layerwise_nvtx")}
        baseline = [
            {"step": 0, "repeat": 0, "warmup": True, "kind": "prefill",
             "wall_start_ns": 1010, "wall_end_ns": 1020,
             "monotonic_start_ns": 10, "monotonic_end_ns": 20},
            {"step": 1, "repeat": 1, "warmup": False, "kind": "prefill",
             "wall_start_ns": 1100, "wall_end_ns": 1120,
             "monotonic_start_ns": 100, "monotonic_end_ns": 120},
            {"step": 2, "repeat": 1, "warmup": False, "kind": "decode",
             "wall_start_ns": 1150, "wall_end_ns": 1170,
             "monotonic_start_ns": 150, "monotonic_end_ns": 170},
            {"step": 3, "repeat": 1, "warmup": False, "kind": "decode",
             "wall_start_ns": 1210, "wall_end_ns": 1230,
             "monotonic_start_ns": 210, "monotonic_end_ns": 230}]
        profiled = [{**r, "step": r["step"] + 10, "monotonic_start_ns": r["monotonic_start_ns"] + 500,
                     "monotonic_end_ns": r["monotonic_end_ns"] + 505} for r in baseline[:-1]]
        metrics = {"clock": {"step_clock": "monotonic", "wall_minus_monotonic_ns": 1000,
                              "how": "fake"}, "requests": [{"repeat": 1, "metrics": {
                                  "arrival_time": 1080e-9, "queued_ts": 90e-9,
                                  "first_token_ts": 115e-9, "last_token_ts": 225e-9},
                                  "generate_return_monotonic_ns": 250}]}
        result = a._corrected_timeline(meta, baseline, profiled, [], metrics, {}, {}, [],
                                       {"clock": {"wall_minus_monotonic_ns": 2000}})
        self.assertEqual([s["baseline_gap_before_ns"] for s in result["steps"]], [None, None, 30, 40])
        self.assertEqual([s["profiled_gap_before_ns"] for s in result["steps"]], [None, None, 25, None])
        self.assertEqual(result["steps"][1]["profiled_wall_start_ns"], 600)
        # New profiled timestamps exist even for uncaptured steps; old nsys fields stay null.
        self.assertIsNone(result["steps"][1]["profiled_wall_ns"])
        self.assertIsNone(result["steps"][3]["profiled_wall_end_ns"])
        req = result["requests"][0]
        self.assertEqual({k: req[k] for k in ("first_step_start_ns", "last_step_end_ns",
                         "arrival_ns", "queued_ns", "first_token_ns", "last_token_ns",
                         "pre_first_step_ns", "post_last_step_ns")}, {
                             "first_step_start_ns": 100, "last_step_end_ns": 230,
                             "arrival_ns": 80, "queued_ns": 90, "first_token_ns": 115,
                             "last_token_ns": 225, "pre_first_step_ns": 20, "post_last_step_ns": 20})
        self.assertEqual(result["meta"]["clock"]["wall_minus_monotonic_ns"], 1000)
        self.assertEqual(result["meta"]["clock"]["profiled_wall_minus_monotonic_ns"], 2000)
        self.assertEqual(result["summary"]["decode"]["baseline_gap_before_ns"], {
            "mean": 35, "p50": 35, "p95": 39.5})
        self.assertEqual(result["summary"]["decode"]["profiled_gap_before_ns"]["mean"], 25)
        self.assertIsNone(result["summary"]["prefill"]["baseline_gap_before_ns"]["mean"])
        legacy = [{k: v for k, v in r.items() if not k.startswith("monotonic_")} for r in baseline]
        converted = a._corrected_timeline(meta, legacy, [], [], metrics, {}, {}, [])
        self.assertEqual(converted["steps"][2]["baseline_wall_start_ns"], 150)
        missing = a._corrected_timeline(meta, legacy, [], [], {"requests": [{"repeat": 5}]}, {}, {}, [])
        self.assertIsNone(missing["steps"][2]["baseline_wall_start_ns"])
        self.assertIsNone(missing["requests"][0]["post_last_step_ns"])

    def test_workload_repeats_and_profiler_order(self):
        @dataclass
        class Stats:
            num_generation_tokens: int = 2
            arrival_time: float = 1000
            queued_ts: float = 1
            scheduled_ts: float = 2
            first_token_ts: float = 3
            last_token_ts: float = 4
            first_token_latency: float = 2

        @dataclass
        class Histogram:
            count: int = 1
            sum: float = 2
            buckets: dict = None

        calls, kwargs = [], {}
        output = SimpleNamespace(metrics=Stats(), request_id="r", prompt="hello",
                                 outputs=[SimpleNamespace(text="world")])

        class FakeLLM:
            def __init__(self, **kw):
                kwargs.update(kw)
                self.llm_engine = SimpleNamespace(step=lambda: calls.append("step"))

            def collective_rpc(self, fn):
                pass

            def generate(self, prompts, params):
                self.llm_engine.step()
                self.llm_engine.step()
                return [output]

            def start_profile(self):
                calls.append("start")

            def stop_profile(self):
                calls.append("stop")

            def get_metrics(self):
                return [Histogram(buckets={"+Inf": 1})]

        fake_modules = {
            "torch": SimpleNamespace(nn=SimpleNamespace(Module=object),
                                     cuda=SimpleNamespace(synchronize=lambda: calls.append("sync"))),
            "torch_hook": SimpleNamespace(TorchModelHookWrapper=object,
                                          hook_nvbit_to_layer=lambda *args: None),
            "vllm": SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kw: kw),
            "nvtx": SimpleNamespace(push_range=lambda name, **kw: calls.append(name),
                                    pop_range=lambda **kw: None),
        }
        metrics, steps = self.work / "metrics.json", self.work / "steps.json"
        args = ["vllm_trace.py", "--model", "fake", "--prompts", '["hello"]',
                "--trace-scope", "steps", "--iteration-details", "--layerwise-nvtx",
                "--repeats", "3", "--cuda-profiler-from-repeat", "1", "--metrics-out",
                str(metrics), "--step-log", str(steps)]
        with patch.dict(sys.modules, fake_modules), patch.object(sys, "argv", args), \
                patch.dict(os.environ, {}, clear=True):
            runpy.run_path(str(Path(__file__).with_name("vllm_trace.py")), run_name="__main__")
        self.assertEqual(kwargs["profiler_config"], {"profiler": "cuda"})
        self.assertFalse(kwargs["disable_log_stats"])
        self.assertTrue(kwargs["enable_layerwise_nvtx_tracing"])
        self.assertEqual(calls.count("start"), 1)
        self.assertEqual(calls.count("stop"), 1)
        self.assertLess(calls.index("start"), calls.index("vllm_step_2"))
        self.assertGreater(calls.index("start"), calls.index("vllm_step_1"))
        self.assertEqual(calls[-1], "stop")
        recorded = json.loads(steps.read_text())
        self.assertEqual([r["step"] for r in recorded], list(range(6)))
        self.assertEqual([r["repeat"] for r in recorded], [0, 0, 1, 1, 2, 2])
        self.assertEqual([r["warmup"] for r in recorded], [True, True, False, False, False, False])
        data = json.loads(metrics.read_text())
        self.assertEqual(len(data["requests"]), 3)
        self.assertEqual(data["requests"][0]["ttft_s"], 1)
        self.assertEqual(data["metrics"][0]["buckets"], {"+Inf": 1})
        self.assertEqual(data["clock"]["step_clock"], "monotonic")
        self.assertIsInstance(data["clock"]["wall_minus_monotonic_ns"], int)
        self.assertGreaterEqual(data["clock"]["sample_bracket_ns"], 0)
        self.assertGreaterEqual(data["requests"][-1]["generate_return_monotonic_ns"],
                                recorded[-1]["monotonic_end_ns"])
        self.assertEqual(calls.count("sync"), 13)  # startup plus two per step
        for r in recorded:
            self.assertGreaterEqual(r["monotonic_end_ns"], r["monotonic_start_ns"])
            self.assertGreaterEqual(r["pre_step_sync_ns"], 0)
            self.assertGreaterEqual(r["post_step_sync_ns"], 0)
        calls.clear()
        with patch.dict(sys.modules, fake_modules), patch.object(sys, "argv", args + ["--no-step-sync"]), \
                patch.dict(os.environ, {}, clear=True):
            runpy.run_path(str(Path(__file__).with_name("vllm_trace.py")), run_name="__main__")
        self.assertEqual(calls.count("sync"), 1)  # LLM initialization sync is unchanged
        for r in json.loads(steps.read_text()):
            self.assertEqual(r["pre_step_sync_ns"], 0)
            self.assertEqual(r["post_step_sync_ns"], 0)


if __name__ == "__main__":
    unittest.main()
