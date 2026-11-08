import json
import shutil
import tempfile
import unittest
from pathlib import Path

from accelsim_local import core_clock, parse_log, simulate_accelsim
from cascade_profiler.align import build_alignment, read_launches, write_alignment


class SimulationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.home())
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.run = self.work / "run"
        self.traces = self.run / "traced/traces"
        self.traces.mkdir(parents=True)
        def kernel(seq, name):
            return {"seq": seq, "name": name, "grid": [1, 1, 1], "block": [32, 1, 1]}
        self.timeline = {"meta": {"run_name": "fixture", "sm_clock_mhz_median": 1800, "gpu_name": "NVIDIA H100 80GB HBM3"}, "steps": [
            {"repeat": 1, "step": 10, "kind": "decode", "kernels": [kernel(0, "_Za"), kernel(1, "_Zb"), kernel(2, "_Zc")]},
            {"repeat": 1, "step": 11, "kind": "decode", "kernels": [kernel(0, "_Za")]},
            {"repeat": 2, "step": 20, "kind": "decode", "kernels": [kernel(0, "_Za")]},
            {"repeat": 3, "step": 30, "kind": "prefill", "kernels": [kernel(0, "_Za")]}]}
        self.trace_steps = [{"repeat": 1, "step": 2, "step_in_repeat": 0, "traced": True,
                             "first_kernel_id": 1, "end_kernel_id": 4},
                            {"repeat": 1, "step": 3, "step_in_repeat": 1, "traced": True,
                             "first_kernel_id": 4, "end_kernel_id": 5}]
        names = ["_Za", "_Zb", "_Zc", "_Za"]
        specs = [{"name": name, "cycles": cycles, "mode": mode} for name, cycles, mode in
                 zip(names, [100, 200, 300, 140], ["ok", "fail", "timeout", "ok"])]
        rows = ["header"]
        for uid, spec in enumerate(specs, 1):
            filename = f"kernel-{uid}-ctx_0xabc.trace.xz"
            rows.append(f"{filename}, {spec['name']}, 1, 1, 1, 1, 32, 1, 1, 32, 100, 100")
            (self.traces / (filename.removesuffix(".xz") + "z")).write_text(json.dumps(spec))
        (self.traces / "stats_ctx_0xabc").write_text("\n".join(rows))
        (self.traces / "kernelslist.g").write_text("MemcpyHtoD,0x100,4\n" + "\n".join(p.name for p in self.traces.glob("*.tracez")))
        (self.run / "corrected_timeline.json").write_text(json.dumps(self.timeline))
        (self.run / "traced/trace_steps.json").write_text(json.dumps(self.trace_steps))
        (self.run / "run.json").write_text(json.dumps({"trace_repeat": 1}))
        self.alignment = write_alignment(self.run)
        self.root = self.work / "framework"
        cfg = self.root / "gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100"
        cfg.mkdir(parents=True)
        (cfg / "gpgpusim.config").write_text("# -gpgpu_clock_domains 12:1:1:1\n-gpgpu_clock_domains 1980:3960:3960:3106\n")
        (cfg / "config_fermi_islip.icnt").write_text("fake companion config")
        tracecfg = self.root / "gpu-simulator/configs/tested-cfgs/SM90_H100"
        tracecfg.mkdir(parents=True)
        (tracecfg / "trace.config").write_text("# trace config\n")
        (self.root / "gpu-simulator/setup_environment.sh").write_text("export CASCADE_TEST_RUNTIME=ready\nreturn 0\n")
        for target in (self.root / "gpu-simulator/bin/release/accel-sim.out",
                       self.root / "util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"):
            target.parent.mkdir(parents=True)
            shutil.copyfile(Path(__file__).with_name("fake_accelsim.py"), target)
            target.chmod(0o755)

    def test_alignment_and_mismatches(self):
        self.assertTrue(all(s["exact"] for s in self.alignment["steps"]))
        self.assertEqual([k["step"] for k in self.alignment["kernels"]], [10, 10, 10, 11])
        launches = read_launches(self.traces)
        launches[1]["grid"] = [2, 1, 1]
        launches[2]["name"] = "_Zunknown"
        result = build_alignment(self.timeline, self.trace_steps, launches)
        self.assertEqual([k["dims_match"] for k in result["kernels"]], [True, False, True])
        self.assertEqual(result["unmatched"]["n_tracer"], 1)
        self.assertEqual(result["unmatched"]["n_timeline"], 1)
        self.assertFalse(result["steps"][0]["exact"])
        self.assertFalse(result["provenance"]["traced_timing_used"])
        self.assertEqual(build_alignment(self.timeline, [], [])["kernels"], [])
        self.assertEqual(build_alignment({"meta": {}, "steps": []}, self.trace_steps, launches)["steps"][0]["step"], None)
        with self.assertRaisesRegex(ValueError, "range"):
            build_alignment(self.timeline, [{**self.trace_steps[0], "end_kernel_id": None}], launches)

    def test_log_parser(self):
        text = "kernel_name = _Zfoo\nkernel_launch_uid = 1\ngpu_sim_cycle = 12345\ngpu_tot_sim_cycle = 77777\n"
        self.assertEqual(parse_log(text), {"name": "_Zfoo", "kernel_launch_uid": 1, "sim_cycles": 12345})
        for bad in ("", "gpu_sim_cycle = 10", text.replace("= 1\n", "= 1 2\n"),
                    text + "kernel_name = _Zbar\nkernel_launch_uid = 2\ngpu_sim_cycle = 20\n",
                    text + "break due to reaching the maximum cycles"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_log(bad)
        cfg = self.work / "bad.config"
        cfg.write_text("# no clock")
        with self.assertRaises(ValueError):
            core_clock(cfg)

    def test_simulate_and_propagate(self):
        summary = simulate_accelsim(self.run, accelsim_root=self.root, jobs=2,
                                    kernel_timeout_s=0.3, propagate=True)
        self.assertEqual(summary["counts"]["ok"], 2)
        self.assertEqual(summary["counts"]["failed"], 1)
        self.assertEqual(summary["counts"]["timeout"], 1)
        self.assertEqual(summary["counts"]["propagated"], 1)
        self.assertEqual([(r["simulated"], r["total"]) for r in summary["coverage"]], [(1, 3), (1, 1)])
        data = json.loads((self.run / "sim_durations.json").read_text())
        self.assertEqual([r["sim_cycles"] for r in data["kernels"]], [100, 140, 120])
        self.assertEqual(data["kernels"][-1]["source"], "propagated")
        self.assertEqual(data["provenance"]["clock_mhz"], 1980)
        self.assertEqual(data["provenance"]["measured_sm_clock_mhz"], 1800)
        self.assertEqual(data["provenance"]["ambiguous_propagation"][0]["median"], 120)
        listing = self.run / "sim/kernel-1/kernelslist.g"
        self.assertEqual(listing.read_text(), "kernel-1-ctx_0xabc.tracez\n")
        self.assertTrue((listing.parent / "kernel-1-ctx_0xabc.tracez").is_file())
        self.assertTrue((listing.parent / "config_fermi_islip.icnt").is_file())
        self.assertIn("timeout", (self.run / "sim/kernels.csv").read_text())

    def test_postprocess_subset_empty_and_errors(self):
        processed = self.traces / "kernel-1-ctx_0xabc.tracez"
        (self.traces / "kernel-1-ctx_0xabc.trace.xz").write_text(processed.read_text())
        processed.unlink()
        (self.traces / "kernelslist.g").unlink()
        result = simulate_accelsim(self.run, accelsim_root=self.root, jobs=1, uids=[1])
        self.assertEqual(result["counts"]["ok"], 1)
        self.assertTrue(processed.is_file())
        self.assertTrue((self.run / "sim/post_process.log").is_file())
        result = simulate_accelsim(self.run, accelsim_root=self.root, jobs=1, uids=[])
        self.assertEqual(result["counts"]["selected"], 0)
        self.assertEqual(json.loads((self.run / "sim_durations.json").read_text())["kernels"], [])
        for kwargs in ({"jobs": 0}, {"uids": [99]}, {"kernel_timeout_s": -1}, {"gpu_config": "../x"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate_accelsim(self.run, accelsim_root=self.root, **kwargs)

    def test_partial_postprocess_failure_is_not_fatal(self):
        (self.traces / "kernel-4-ctx_0xabc.tracez").unlink()
        postprocessor = self.root / "util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"
        postprocessor.write_text("#!/usr/bin/env python3\nraise SystemExit(2)\n")
        summary = simulate_accelsim(self.run, accelsim_root=self.root, jobs=2, uids=[1, 4])
        self.assertEqual(summary["counts"]["ok"], 1)
        self.assertEqual(summary["counts"]["failed"], 1)
        self.assertIsNotNone(summary["post_process_error"])


if __name__ == "__main__":
    unittest.main()
