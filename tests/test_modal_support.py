import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cascade_profiler
from cascade_profiler import modal_support as support


class ModalSupportTest(unittest.TestCase):
    def test_image_recipes(self):
        recipe = Mock()
        for method in ("entrypoint", "uv_pip_install", "apt_install", "env", "run_commands", "add_local_python_source"):
            getattr(recipe, method).return_value = recipe
        modal = SimpleNamespace(Image=SimpleNamespace(from_registry=Mock(return_value=recipe)))
        with patch.dict(sys.modules, {"modal": modal}):
            support.image()
            self.assertIn("cascade_profiler", recipe.add_local_python_source.call_args.args)
            recipe.reset_mock()
            support.accelsim_image()
        self.assertEqual(recipe.env.call_args.args[0]["GPGPUSIM_BRANCH"], support.GPGPUSIM_REF)
        self.assertEqual(recipe.env.call_args.args[0]["CPLUS_INCLUDE_PATH"],
                         f"{support.ROOT}/util/tracer_nvbit/tracer_tool")
        commands = " ".join(recipe.run_commands.call_args.args)
        self.assertIn("source", commands)
        self.assertIn("setup_environment.sh release", commands)
        self.assertIn("post-traces-processing", commands)
        self.assertIn("-pthread $(BUILD_DIR)/*.o -lzstd", commands)

    def test_registration_and_commit_on_failure(self):
        options = []
        def decorator(**kwargs):
            options.append(kwargs)
            return lambda fn: fn
        app = SimpleNamespace(function=decorator)
        volume = SimpleNamespace(reload=Mock(), commit=Mock())
        result = SimpleNamespace(run_dir=Path("/data/fixture"), stage_wall_s={"baseline": 1})
        with patch.object(cascade_profiler, "profile", return_value=result) as profile, \
                patch.object(cascade_profiler, "simulate_accelsim", return_value={"coverage": []}) as simulate:
            functions = support.register(app, volume, gpu="H100!", image="GPU", sim_image="CPU", hf_cache="cache")
            response = functions.profile({"model": "fake", "prompts": ["hello"]}, "fixture", trace_steps=[0])
            self.assertEqual(response["run_name"], "fixture")
            profile.assert_called_once()
            self.assertEqual(functions.simulate("fixture"), {"coverage": []})
            self.assertEqual(simulate.call_args.kwargs["jobs"], 32)
            profile.side_effect = RuntimeError("failed")
            with self.assertRaisesRegex(RuntimeError, "failed"):
                functions.profile({"model": "fake", "prompts": ["hello"]})
            with self.assertRaises(ValueError):
                functions.simulate("../escape")
        self.assertEqual(volume.commit.call_count, 3)
        self.assertEqual(volume.reload.call_count, 3)
        self.assertTrue(all(o["serialized"] for o in options))
        self.assertEqual(options[0]["gpu"], "H100!")
        self.assertNotIn("gpu", options[1])
        self.assertEqual(options[0]["volumes"]["/root/.cache/huggingface"], "cache")


if __name__ == "__main__":
    unittest.main()
