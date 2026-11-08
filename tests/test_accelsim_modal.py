import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from cascade.sim import modal_support as support


def test_image_recipe():
    recipe = Mock()
    for method in (
        "entrypoint",
        "uv_pip_install",
        "apt_install",
        "env",
        "run_commands",
        "add_local_python_source",
    ):
        getattr(recipe, method).return_value = recipe
    modal = SimpleNamespace(Image=SimpleNamespace(from_registry=Mock(return_value=recipe)))
    with patch.dict(sys.modules, {"modal": modal}):
        support.accelsim_image()
    assert recipe.env.call_args.args[0]["GPGPUSIM_BRANCH"] == support.GPGPUSIM_REF
    assert (
        recipe.env.call_args.args[0]["CPLUS_INCLUDE_PATH"]
        == f"{support.ROOT}/util/tracer_nvbit/tracer_tool"
    )
    commands = " ".join(recipe.run_commands.call_args.args)
    assert "-pthread $(BUILD_DIR)/*.o -lzstd" in commands
    assert "setup_environment.sh release" in commands
    assert recipe.add_local_python_source.call_args.args == ("cascade",)


def test_registration():
    options = []

    def decorator(**kwargs):
        options.append(kwargs)
        return lambda fn: fn

    app = SimpleNamespace(function=decorator)
    volume = SimpleNamespace(reload=Mock(), commit=Mock())
    with patch("cascade.sim.accelsim.simulate_accelsim", return_value={}) as simulate:
        remote = support.register(app, volume, image="CPU", mount="/traces", run_subdir="phases")
        assert remote("fixture") == {}
        assert simulate.call_args.args == (Path("/traces/phases/fixture"),)
        assert simulate.call_args.kwargs["jobs"] == 32
        with patch(
            "cascade.sim.trace_simulation.simulate_traces", return_value={"legacy": True}
        ) as legacy:
            assert remote("old", gpu_config="SM90_H100", traces_only=True) == {"legacy": True}
            assert legacy.call_args.args == (Path("/traces/phases/old"),)
        simulate.side_effect = RuntimeError("failed")
        with pytest.raises(RuntimeError, match="failed"):
            remote("fixture")
        for name in ("", "../escape", ".", ".."):
            with pytest.raises(ValueError):
                remote(name)
    assert volume.commit.call_count == 3
    assert volume.reload.call_count == 3
    assert options[0]["serialized"]
    assert "gpu" not in options[0]
    assert options[0]["volumes"] == {"/traces": volume}
    with pytest.raises(ValueError, match="absolute"):
        support.register(app, volume, image="CPU", mount="relative")
    for subdir in ("../escape", "/absolute"):
        with pytest.raises(ValueError, match="inside"):
            support.register(app, volume, image="CPU", run_subdir=subdir)
