from pathlib import Path

from rlrestore.common.config import Config, load_config

REPO = Path(__file__).resolve().parents[1]


def test_load_smoke_config():
    cfg = load_config(REPO / "configs" / "smoke.yaml")
    assert isinstance(cfg, Config)
    assert cfg.tools.epochs == 1
    assert cfg.tools.steps_per_epoch == 30
    assert cfg.name == "smoke"


def test_load_full_config():
    cfg = load_config(REPO / "configs" / "full.yaml")
    assert cfg.tools.epochs == 40
    assert cfg.tools.steps_per_epoch == 4000
    assert cfg.tools.batch_size == 64
    assert cfg.data.severity == "moderate"


def test_full_and_smoke_share_schema():
    smoke = load_config(REPO / "configs" / "smoke.yaml")
    full = load_config(REPO / "configs" / "full.yaml")
    assert type(smoke) is type(full)


def test_load_config_coerces_quoted_floats(tmp_path):
    src = (REPO / "configs" / "smoke.yaml").read_text().replace("lr: 0.1", 'lr: "1e-4"')
    p = tmp_path / "c.yaml"
    p.write_text(src)
    cfg = load_config(p)
    assert isinstance(cfg.tools.lr, float) and cfg.tools.lr == 1e-4
