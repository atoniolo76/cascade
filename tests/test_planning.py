import unittest

from cascade_profiler import Workload, trace_plan


class PlanningTest(unittest.TestCase):
    def test_trace_plan(self):
        workload = Workload("Qwen/second-model", ["not tokenized from text"], repeats=2)
        steps = [{"repeat": 0, "step": 0, "kind": "prefill", "num_reqs": 1, "num_scheduled_tokens": 1}]
        steps += [{"repeat": 1, "step": 100 + n, "kind": "decode" if n else "prefill",
                   "num_reqs": 1, "num_scheduled_tokens": 9999 if not n else 1} for n in range(13)]
        iterations = [{"step": 100, "kind": "prefill", "ctx_reqs": 1, "ctx_tokens": 4}]
        self.assertEqual(trace_plan(workload, baseline_steps=steps, iterations=iterations), [(1, 0), (1, 4), (1, 12)])
        with self.assertRaisesRegex(ValueError, "baseline_steps"):
            trace_plan(workload)
        with self.assertRaises(ValueError):
            trace_plan(workload, (-1,), baseline_steps=steps)
