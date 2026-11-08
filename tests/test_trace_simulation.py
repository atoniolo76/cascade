import json
import shutil
import unittest

import test_accelsim as fixtures

from cascade.sim.trace_simulation import simulate_traces


class TraceSimulationTest(unittest.TestCase):
    def setUp(self):
        fixtures.SimulationTest.setUp(self)
        shutil.move(self.traces, self.run / "traces")
        self.traces = self.run / "traces"
        (self.run / "corrected_timeline.json").unlink()
        (self.run / "trace_alignment.json").unlink()
        for path in self.traces.glob("*.tracez"):
            payload = path.read_bytes()
            spec = json.loads(payload)
            header = (
                f"-kernel name = {spec['name']}\n-grid dim = (1,1,1)\n"
                "-block dim = (32,1,1)\n-nregs = 32\n"
            )
            path.write_bytes(header.encode() + b"\0" + payload)

    def test_uid_only_results(self):
        summary = simulate_traces(
            self.run,
            gpu_config="SM90_H100",
            accelsim_root=self.root,
            jobs=2,
            kernel_timeout_s=0.3,
            verify_dedup=2,
        )
        self.assertEqual(summary["counts"]["unique_accelsim_runs"], 4)
        self.assertEqual(summary["counts"]["traced_kernel_occurrences"], 4)
        self.assertEqual(summary["counts"]["ok"], 2)
        self.assertEqual(summary["counts"]["failed"], 1)
        self.assertEqual(summary["counts"]["timeout"], 1)
        self.assertEqual(summary["dedup_verification"]["max_absolute_cycles"], 40)
        data = json.loads((self.run / "sim_durations.json").read_text())
        self.assertEqual(data["provenance"]["scope"], "tracer_uid_only")
        self.assertIsNone(data["provenance"]["alignment"])
        self.assertCountEqual([r["tracer_uid"] for r in data["kernels"]], [1, 4])
        self.assertTrue(all("step" not in r and "seq" not in r for r in data["kernels"]))
        self.assertFalse((self.run / "trace_alignment.json").exists())
        self.assertFalse((self.run / "corrected_timeline.json").exists())

    def test_empty_and_invalid(self):
        listing = self.traces / "kernelslist.g"
        original = listing.read_text()
        listing.write_text("")
        result = simulate_traces(self.run, gpu_config="SM90_H100", accelsim_root=self.root)
        self.assertEqual(result["counts"]["unique_accelsim_runs"], 0)
        self.assertEqual(json.loads((self.run / "sim_durations.json").read_text())["kernels"], [])
        listing.write_text(original + "\n" + original.splitlines()[1] + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            simulate_traces(self.run, gpu_config="SM90_H100", accelsim_root=self.root)
        listing.write_text(original)
        for kwargs in ({"jobs": 0}, {"verify_dedup": -1}, {"dedup": 1}, {"kernel_timeout_s": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate_traces(self.run, gpu_config="SM90_H100", **kwargs)
        (self.traces / "kernel-1-ctx_0xabc.tracez").write_bytes(b"truncated")
        with self.assertRaisesRegex(ValueError, "kernel name"):
            simulate_traces(self.run, gpu_config="SM90_H100", accelsim_root=self.root)
