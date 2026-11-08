import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cascade_profiler import Workload, profile
from cascade_profiler import pipeline
from tests import test_phases


class WorkloadTest(unittest.TestCase):
    def test_round_trip_and_defaults(self):
        workload = Workload("fake", ["one", "two"], llm_kwargs={"dtype": "float16"},
                            sampling_kwargs={"top_p": 0.8})
        self.assertEqual(Workload.from_dict(json.loads(json.dumps(workload.to_dict()))), workload)
        self.assertEqual(workload.sampling_params_kwargs(), {
            "temperature": 0.0, "seed": 0, "ignore_eos": True, "max_tokens": 32, "top_p": 0.8})
        self.assertEqual(workload.merged_llm_kwargs({"async_scheduling": False})["dtype"], "float16")

    def test_invalid_workloads_and_conflicts(self):
        for kwargs in ({"prompts": []}, {"repeats": 0}, {"exec_mode": "wrong"},
                       {"llm_kwargs": {"enforce_eager": False}},
                       {"llm_kwargs": {"async_scheduling": True}}, {"driver": lambda: None}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                Workload(**{"model": "fake", "prompts": ["hello"], **kwargs})
        with self.assertRaisesRegex(ValueError, "disable_log_stats.*conflicts"):
            profile(Workload("fake", ["hello"], llm_kwargs={"disable_log_stats": True}), ".")

    def test_driver_validation(self):
        self.assertIsNone(Workload("fake", ["hi"]).resolve_driver())
        self.assertIs(Workload("fake", [], driver="json:loads").resolve_driver(), json.loads)
        for driver in ("no_such_driver:run", "json:no_such_function", "json:__name__", "__main__:profile"):
            with self.subTest(driver=driver), self.assertRaises(ValueError):
                Workload("fake", ["hi"], driver=driver).resolve_driver()


class PipelineTest(unittest.TestCase):
    def setUp(self):
        test_phases.PhasesTest.setUp(self)
        self.calls = []

    def tearDown(self):
        test_phases.PhasesTest.tearDown(self)

    def fake_run(self, command, **kwargs):
        if command[0] == "custom-nsys":
            return SimpleNamespace(stdout="--cuda-event-trace --cuda-graph-trace", returncode=0)
        if command[0] == "git":
            return SimpleNamespace(stdout="fake-sha\n", returncode=0)
        if command[0] == "nvidia-smi":
            return SimpleNamespace(stdout="H100, 570\n", returncode=0)
        raise AssertionError(command)

    def fake_popen(self, command, **kwargs):
        work = kwargs["cwd"]
        self.calls.append((command, kwargs["env"], work.name))
        if "--workload-json" in command:
            config = Workload.from_dict(json.loads(Path(command[command.index("--workload-json") + 1]).read_text()))
            self.assertEqual(config.model, "fake")
            log, steps = test_phases.PhasesTest.fake_log()
            local_steps = (set(map(int, command[command.index("--trace-steps-in-repeat") + 1].split(",")))
                           if "--trace-steps-in-repeat" in command and command[command.index("--trace-steps-in-repeat") + 1]
                           else set())
            for r in steps:
                r["step_in_repeat"] = r["step"] % 2
                r["traced"] = (work.name == "traced" and r["repeat"] == 1
                               and (not local_steps or r["step_in_repeat"] in local_steps))
                r.update(first_kernel_id=r["step"] + 1, end_kernel_id=r["step"] + 2)
            # Spinlock commands override these flags at the end.
            def last_value(flag):
                return command[max(i for i, arg in enumerate(command) if arg == flag) + 1]
            Path(last_value("--step-log")).write_text(json.dumps(steps))
            Path(last_value("--metrics-out")).write_text(json.dumps({"requests": []}))
            if work.name == "profiled":
                shutil.copyfile(self.db, work / "profile.sqlite")
                (work / "profile.nsys-rep").write_text("unscrubbed")
            if work.name == "traced":
                spin = work / "spinlock_detection"
                spin.mkdir(exist_ok=True)
                (spin / "spinlock_instructions.txt").write_text("fake spinlocks")
        else:
            self.assertTrue(command[0].endswith("post-traces-processing"))
            traces = work / "traces"
            traces.mkdir(exist_ok=True)
            (traces / "kernelslist.g").write_text("kernel-1.tracez\n")
            log = "post-processed\n"
        return SimpleNamespace(stdout=iter(log.splitlines(keepends=True)), returncode=0,
                               wait=lambda: 0, kill=lambda: None)

    def patches(self):
        return patch.multiple(pipeline.ClockSampler, start=lambda s: None, stop=lambda s: None)

    def test_three_phase_orchestration(self):
        with patch.object(subprocess, "Popen", side_effect=self.fake_popen), \
                patch.object(subprocess, "run", side_effect=self.fake_run), self.patches(), \
                patch.dict(os.environ, {"MODAL_ID": "hidden", "MY_SECRET": "hidden",
                                        "CUDA_INJECTION64_PATH": "old.so"}):
            result = profile(Workload("fake", ["hello"], repeats=2), self.work / "runs", run_name="explicit",
                             phases=("profiled", "baseline"), trace_steps=[0], accelsim_root="/tools/accelsim",
                             nsys_bin="custom-nsys", no_step_sync=True, gc_between_repeats=False)
        self.assertEqual([stage for _, _, stage in self.calls],
                         ["baseline", "profiled", "traced", "traced", "traced", "traced"])
        baseline, profiled = self.calls[:2]
        self.assertEqual(baseline[0][1:3], ["-m", "cascade_profiler.runner"])
        self.assertNotIn("--cuda-profiler-from-repeat", baseline[0])
        self.assertEqual(profiled[0][:2], ["custom-nsys", "profile"])
        self.assertIn("--capture-range-end=stop", profiled[0])
        self.assertIn("--cuda-event-trace=false", profiled[0])
        self.assertIn("--cuda-profiler-from-repeat", profiled[0])
        for command, env, _ in self.calls:
            self.assertFalse(set(env) & {"CUDA_INJECTION64_PATH", "MODAL_ID", "MY_SECRET"})
            if "--workload-json" in command:
                self.assertIn("--no-step-sync", command)
                self.assertIn("--no-gc-between-repeats", command)
        traced_command = self.calls[-2][0]
        self.assertEqual(traced_command[0], "/tools/accelsim/util/tracer_nvbit/others/torch_hook/run.sh")
        self.assertIn("CUDA_INJECTION64_PATH=/tools/accelsim/util/tracer_nvbit/tracer_tool/tracer_tool.so", traced_command)
        self.assertEqual(traced_command[traced_command.index("--trace-steps-in-repeat") + 1], "0")
        self.assertEqual(traced_command[traced_command.index("--trace-repeat") + 1], "1")
        self.assertTrue(any("spinlock_tool.so" in arg for arg in self.calls[2][0]))
        manifest = json.loads(result.paths["run.json"].read_text())
        self.assertEqual(manifest["run_name"], "explicit")
        self.assertEqual(manifest["phases"], ["baseline", "profiled", "traced"])
        self.assertFalse(manifest["traced_timing_used"])
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["git_sha"], "fake-sha")
        self.assertEqual(set(result.stage_wall_s), {"baseline", "profiled", "traced", "post_process"})
        self.assertTrue(result.paths["corrected_timeline.json"].exists())
        self.assertTrue(result.paths["baseline_iterations.jsonl"].exists())
        self.assertTrue(result.paths["profiled_iterations.jsonl"].exists())
        self.assertTrue(result.paths["traced/traces/kernelslist.g"].exists())
        self.assertNotIn("profiled/profile.nsys-rep", result.paths)
        traced_steps = json.loads(result.paths["traced/trace_steps.json"].read_text())
        self.assertEqual([(r["repeat"], r["step_in_repeat"], r["step"]) for r in traced_steps if r["traced"]], [(1, 0, 2)])

    def test_baseline_only_and_failed_run(self):
        with patch.object(subprocess, "Popen", side_effect=self.fake_popen), \
                patch.object(subprocess, "run", side_effect=self.fake_run), self.patches():
            result = profile(Workload("fake", ["hello"], repeats=1), self.work / "runs", phases=("baseline",))
        self.assertNotIn("corrected_timeline.json", result.paths)
        with patch.object(subprocess, "Popen", side_effect=RuntimeError("subprocess failure")), \
                patch.object(subprocess, "run", side_effect=self.fake_run), self.patches():
            with self.assertRaisesRegex(RuntimeError, "subprocess failure"):
                profile(Workload("fake", ["hello"]), self.work / "runs", run_name="failed")
        manifest = json.loads((self.work / "runs/failed/run.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("baseline", manifest["stage_wall_s"])

    def test_layerwise_opt_in_and_report_retention(self):
        with patch.object(subprocess, "Popen", side_effect=self.fake_popen), \
                patch.object(subprocess, "run", side_effect=self.fake_run), self.patches():
            result = profile(Workload("fake", ["hello"], repeats=2), self.work / "runs",
                             nsys_bin="custom-nsys", layerwise_nvtx=True, keep_nsys_rep=True)
        self.assertNotIn("--layerwise-nvtx", self.calls[0][0])
        self.assertIn("--layerwise-nvtx", self.calls[1][0])
        self.assertIn("profiled/profile.nsys-rep", result.paths)
        self.assertEqual(set(result.stage_wall_s), {"baseline", "profiled"})
        self.assertNotIn("--no-step-sync", self.calls[0][0])
        self.assertNotIn("--no-gc-between-repeats", self.calls[0][0])

    def test_trace_all_and_unreached_selection(self):
        with patch.object(subprocess, "Popen", side_effect=self.fake_popen), \
                patch.object(subprocess, "run", side_effect=self.fake_run), self.patches():
            result = profile(Workload("fake", ["hello"], repeats=2), self.work / "runs",
                             phases=(), trace_steps=[])
            self.assertNotIn("corrected_timeline.json", result.paths)
            records = json.loads(result.paths["traced/trace_steps.json"].read_text())
            self.assertEqual([(r["repeat"], r["step_in_repeat"]) for r in records if r["traced"]],
                             [(1, 0), (1, 1)])
            with self.assertRaisesRegex(ValueError, "not reached"):
                profile(Workload("fake", ["hello"], repeats=2), self.work / "runs",
                        run_name="unreached", phases=(), trace_steps=[50])
        manifest = json.loads((self.work / "runs/unreached/run.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(set(manifest["stage_wall_s"]), {"traced"})

    def test_validation_before_launch(self):
        invalid = ({"phases": ()}, {"phases": ("invalid",)}, {"trace_steps": [-1]},
                   {"trace_steps": [0], "trace_repeat": 99}, {"run_name": "../escape"})
        with patch.object(subprocess, "Popen") as launch:
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    profile(Workload("fake", ["hello"]), self.work, **kwargs)
            with self.assertRaisesRegex(ValueError, "cannot import driver"):
                profile(Workload("fake", ["hello"], driver="unknown_driver:run"), self.work)
        launch.assert_not_called()
