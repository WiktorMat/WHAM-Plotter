from __future__ import annotations

from pathlib import Path

from wham.config import DecayModeSplitCfg, load_config
from wham.decay_modes import category_config, decay_mode_categories


def _with_split(workspace: dict, *, level: int, channel: str, column: str = "decayModePNet"):
    cfg = load_config(workspace["yaml"])
    return cfg.model_copy(update={
        "decay_mode_split": DecayModeSplitCfg(
            level=level, channel=channel, column=column
        )
    })


def test_level_one_has_only_inclusive_plot(workspace: dict) -> None:
    cfg = _with_split(workspace, level=1, channel="tt")
    assert decay_mode_categories(cfg) == ()


def test_semileptonic_levels_two_and_three_are_identical(workspace: dict) -> None:
    level2 = _with_split(workspace, level=2, channel="mt")
    level3 = _with_split(workspace, level=3, channel="mt")
    assert [c.name for c in decay_mode_categories(level2)] == ["mu-pi", "mu-rho", "mu-a1"]
    assert decay_mode_categories(level2) == decay_mode_categories(level3)
    assert "decayModePNet_2" in level2.required_columns()
    assert "decayModePNet_1" not in level2.required_columns()


def test_tautau_level_two_and_three_are_cumulative(workspace: dict) -> None:
    level2 = _with_split(workspace, level=2, channel="tt")
    level3 = _with_split(workspace, level=3, channel="tt")
    cats2 = decay_mode_categories(level2)
    cats3 = decay_mode_categories(level3)
    assert [c.name for c in cats2] == [
        "pi-anything", "rho-anything", "a1-anything",
        "anything-pi", "anything-rho", "anything-a1",
    ]
    assert cats3[:len(cats2)] == cats2
    assert len(cats3) == 15
    a1rho = next(c for c in cats3 if c.name == "a1-rho")
    assert a1rho.output_subdir == Path("split_by_decay_modes/a1-rho")
    assert "decayModePNet_1 == 10" in a1rho.cut
    assert "decayModePNet_2 == 2" in a1rho.cut


def test_legacy_decay_mode_column_and_category_config(workspace: dict) -> None:
    cfg = _with_split(workspace, level=3, channel="et", column="decayMode")
    rho = next(c for c in decay_mode_categories(cfg) if c.name == "e-rho")
    assert rho.cut == "decayMode_2 == 1"
    split_cfg = category_config(cfg, rho)
    assert split_cfg.name == f"{cfg.name}__dm__e-rho"
    assert split_cfg.output_dir == cfg.resolved_output_dir() / "e-rho"
    assert "decayMode_2 == 1" in split_cfg.selection
