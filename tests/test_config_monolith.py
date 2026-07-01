from pathlib import Path

from rlrestore.common.config import Config, MonolithCfg, load_config

REPO = Path(__file__).resolve().parents[1]


def test_monolith_config_loads_with_right_types():
    cfg = load_config(REPO / "configs" / "monolith.yaml")
    assert isinstance(cfg, Config)
    assert cfg.monolith is not None
    m = cfg.monolith
    assert isinstance(m, MonolithCfg)
    # Spec values
    assert m.loss == "charbonnier"
    assert isinstance(m.eps, float) and m.eps == 1e-3
    assert m.optimizer == "adam"
    assert isinstance(m.lr, float) and m.lr == 2e-4
    assert m.lr_drop_every == 8
    assert isinstance(m.lr_drop_factor, float) and m.lr_drop_factor == 0.5
    assert isinstance(m.weight_decay, float) and m.weight_decay == 1e-4
    assert isinstance(m.grad_clip_norm, float) and m.grad_clip_norm == 0.5
    assert m.batch_size == 64
    assert m.steps_per_epoch == 8000
    assert m.epochs == 60
    assert m.patience == 6
    assert m.severities == ["mild", "moderate", "severe"]
    assert m.robust_extras is False
    assert m.val_pairs == 256
    assert m.depth == 14 and m.width == 64


def test_monolith_config_has_no_agent():
    cfg = load_config(REPO / "configs" / "monolith.yaml")
    assert cfg.agent is None  # monolith config is agent-free


def test_existing_configs_still_load_without_monolith():
    for name in ("smoke.yaml", "full.yaml", "full_vanilla.yaml"):
        cfg = load_config(REPO / "configs" / name)
        assert cfg.monolith is None  # absent -> None, mirrors agent
        assert cfg.agent is not None  # agent configs unaffected
