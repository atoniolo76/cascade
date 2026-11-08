import json
import unittest

import test_accelsim as simulation

from cascade.sim.accelsim import (
    choose_gpu_config,
    dedup_groups,
    duration_records,
    simulate_accelsim,
    trace_header,
)


class AdvancedTest(unittest.TestCase):
    def setUp(self):
        simulation.SimulationTest.setUp(self)

    def test_dedup_and_verification(self):
        files = {}
        for uid in (1, 4):
            path = self.traces / f"kernel-{uid}-ctx_0xabc.tracez"
            payload = path.read_bytes()
            header = (
                f"-kernel name = _Za\n-kernel id = {uid}\n-cuda stream id = {uid}\n"
                "-grid dim = (1,1,1)\n-block dim = (32,1,1)\n"
                "-shmem = 0\n-nregs = 32\n-binary version = 90\n"
            )
            path.write_bytes(header.encode() + b"\0" + payload)
            files[uid] = path
        kernels = [k for k in self.alignment["kernels"] if k["tracer_uid"] in files]
        groups = dedup_groups(kernels, files)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["members"], [1, 4])
        self.assertEqual(trace_header(files[1])["nregs"], "32")
        result = simulate_accelsim(self.run, accelsim_root=self.root, jobs=2, uids=[1, 4])
        self.assertEqual(result["counts"]["unique_accelsim_runs"], 1)
        self.assertEqual(result["counts"]["traced_kernel_occurrences"], 2)
        data = json.loads((self.run / "sim_durations.json").read_text())
        self.assertEqual([k["sim"] for k in data["kernels"]], ["exact", "dedup:1"])
        self.assertEqual([k["sim_cycles"] for k in data["kernels"]], [100, 100])
        result = simulate_accelsim(
            self.run, accelsim_root=self.root, jobs=2, uids=[1, 4], verify_dedup=1
        )
        self.assertEqual(result["counts"]["unique_accelsim_runs"], 2)
        self.assertEqual(result["dedup_verification"]["max_absolute_cycles"], 40)
        self.assertAlmostEqual(result["dedup_verification"]["max_relative"], 0.4)
        self.assertEqual(result["provenance_labels"], {"exact": 2})
        files[4].write_bytes(files[4].read_bytes().replace(b"nregs = 32", b"nregs = 48"))
        self.assertEqual(len(dedup_groups(kernels, files)), 2)
        self.assertEqual(len(dedup_groups(kernels, files, False)), 2)

    def test_interpolation_and_generic_keys(self):
        self.timeline["steps"][0]["context_length"] = 8
        self.timeline["steps"][1]["context_length"] = 16
        template = self.timeline["steps"][2]
        template["context_length"] = 12
        for step, context in ((21, 24), (22, 4)):
            self.timeline["steps"].append({**template, "step": step, "context_length": context})
        self.timeline["steps"].append(
            {**template, "step": 23, "kernels": [{**template["kernels"][0], "grid": [2, 1, 1]}]}
        )
        self.timeline["steps"].append(
            {**template, "step": 24, "kernels": [{**template["kernels"][0], "seq": 9}]}
        )
        results = [
            {"uid": 1, "cycles": 100, "status": "ok"},
            {"uid": 4, "cycles": 140, "status": "ok"},
        ]
        records, ambiguous = duration_records(
            self.timeline, self.alignment, results, 1980, "interp"
        )
        propagated = {r["step"]: r for r in records if r["source"] == "propagated"}
        self.assertEqual(propagated[20]["sim_cycles"], 120)
        self.assertEqual(propagated[20]["sim"], "interp:1,4")
        self.assertEqual(propagated[21]["sim_cycles"], 140)
        self.assertEqual(propagated[21]["sim"], "interp:4,4:clamped")
        self.assertEqual(propagated[22]["sim_cycles"], 100)
        self.assertNotIn(23, propagated)
        self.assertNotIn(24, propagated)
        self.assertTrue(ambiguous)
        records, _ = duration_records(self.timeline, self.alignment, results, 1980, "identical")
        self.assertTrue(next(r for r in records if r["step"] == 20)["sim"].startswith("identical:"))

    def test_gpu_mapping(self):
        self.assertEqual(choose_gpu_config("NVIDIA A100-SXM4-80GB")[0], "SM80_A100")
        self.assertEqual(choose_gpu_config("unknown", "CUSTOM")[0], "CUSTOM")
        with self.assertRaisesRegex(ValueError, "explicitly"):
            choose_gpu_config("unknown")


if __name__ == "__main__":
    unittest.main()
