import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import accelsim_local
from cascade_profiler import Workload


class LocalSupportTest(unittest.TestCase):
    def test_cpu_image_is_repo_local(self):
        recipe = Mock()
        for method in ("entrypoint", "apt_install", "uv_pip_install", "env", "run_commands", "add_local_python_source"):
            getattr(recipe, method).return_value = recipe
        modal = SimpleNamespace(Image=SimpleNamespace(from_registry=Mock(return_value=recipe)))
        with patch.dict(sys.modules, {"modal": modal}):
            accelsim_local.accelsim_image()
        self.assertEqual(recipe.add_local_python_source.call_args.args, ("accelsim_local", "cascade_profiler"))
        self.assertEqual(recipe.env.call_args.args[0]["GPGPUSIM_BRANCH"], accelsim_local.GPGPUSIM_REF)
        self.assertIn("post-traces-processing", " ".join(recipe.run_commands.call_args.args))

    def test_app_e2e_modes(self):
        tree = ast.parse(Path("accelsim_trace.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "e2e_main")
        node.decorator_list = []
        profile = Mock()
        profile.with_options.return_value.remote.return_value = {"run_name": "fixture", "stage_wall_s": {}}
        simulate = Mock()
        simulate.remote.return_value = {"wall_s": 1, "coverage": [], "counts": {}, "provenance_labels": {}}
        scope = {"Workload": Workload, "profile_e2e": profile, "simulate": simulate, "print": Mock()}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "accelsim_trace.py", "exec"), scope)
        scope["e2e_main"](run_name="fixture", timing_only=True)
        self.assertIsNone(profile.with_options.return_value.remote.call_args.args[2])
        simulate.remote.assert_not_called()
        scope["e2e_main"](run_name="fixture", profile_only=True)
        simulate.remote.assert_not_called()
        scope["e2e_main"](run_name="fixture", verify_dedup=2)
        simulate.remote.assert_called_once_with("fixture", gpu_config=None, propagate="none", dedup=True, verify_dedup=2)


if __name__ == "__main__":
    unittest.main()
