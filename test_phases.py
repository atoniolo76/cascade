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
                    ('ENVIRONMENT_VARIABLE:HF_TOKEN','secret',1),
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
            self.assertIn("--cuda-event-trace=true", a._nsys_base_cmd())
            run.return_value.stdout = ""
            self.assertNotIn("--cuda-event-trace=false", a._nsys_base_cmd(cuda_events=False))

    def test_clock_thread_can_stop(self):
        sampler = a.ClockSampler(self.work / "clocks.csv")
        with patch.object(sampler, "run", return_value=None):
            sampler.start()
            sampler.stop()
        self.assertTrue((self.work / "clocks.csv").exists())

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
                                     cuda=SimpleNamespace(synchronize=lambda: None)),
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


if __name__ == "__main__":
    unittest.main()
