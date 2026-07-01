from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class DataCfg:
    root: str
    severity: str
    val_pairs: int
    max_train_images: int  # 0 = all


@dataclass(frozen=True)
class ToolsCfg:
    epochs: int
    steps_per_epoch: int
    batch_size: int
    lr: float
    lr_drop_every: int  # epochs
    lr_drop_factor: float
    momentum: float
    weight_decay: float
    grad_clip_norm: float
    robust_extras: bool
    optimizer: str  # "sgd" (paper) | "adam" (fallback)


@dataclass(frozen=True)
class AgentCfg:
    """DQN agent training hyper-parameters (stability_spec §2, infra_spec §3-§4).

    All float fields (lr, gamma, epsilon_end) are coerced from yaml scalars.
    patience is intentionally absent — runner uses env_steps as the only budget.
    """

    env_steps: int
    warmup_steps: int
    replay_capacity: int
    batch_episodes: int
    train_every: int
    target_sync_every: int
    chunk_size: int
    eval_pairs: int
    epsilon_end: float
    epsilon_anneal_steps: int
    lr: float
    gamma: float
    use_double_dqn: bool
    # F5 remedy (stability_spec §5): global grad-norm clip over all agent params
    # incl. LSTM. None = paper-faithful no-clip (kept available for ablation).
    grad_clip_norm: Optional[float] = None


_AGENT_FLOAT_FIELDS = {"lr", "gamma", "epsilon_end", "grad_clip_norm"}


@dataclass(frozen=True)
class MonolithCfg:
    """Monolithic-CNN baseline hyper-parameters (docs/plan3/monolith_spec.md).

    The single end-to-end DnCNN-style control for the headline comparison.
    All float fields are coerced from yaml scalars in load_config.
    """

    loss: str  # "charbonnier" (default) | "mse"
    eps: float  # Charbonnier epsilon
    optimizer: str  # "adam"
    lr: float
    lr_drop_every: int  # epochs
    lr_drop_factor: float
    weight_decay: float
    grad_clip_norm: float
    batch_size: int
    steps_per_epoch: int
    epochs: int  # safety cap; early stopping is the judge
    patience: int
    severities: list  # training severity classes, sampled uniformly
    robust_extras: bool
    val_pairs: int
    depth: int
    width: int


_MONOLITH_FLOAT_FIELDS = {"eps", "lr", "lr_drop_factor", "weight_decay", "grad_clip_norm"}


@dataclass(frozen=True)
class Config:
    name: str
    seed: int
    data: DataCfg
    tools: ToolsCfg
    agent: Optional[AgentCfg] = None  # None when absent — Plan-1 configs/tests stay valid
    monolith: Optional[MonolithCfg] = None  # None when absent — mirrors agent


_TOOL_FLOAT_FIELDS = {"lr", "lr_drop_factor", "momentum", "weight_decay", "grad_clip_norm"}


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    tools_raw = {
        k: float(v) if k in _TOOL_FLOAT_FIELDS else v for k, v in raw["tools"].items()
    }
    agent_cfg: AgentCfg | None = None
    if "agent" in raw:
        agent_raw = {
            k: float(v) if k in _AGENT_FLOAT_FIELDS and v is not None else v
            for k, v in raw["agent"].items()
        }
        agent_cfg = AgentCfg(**agent_raw)
    monolith_cfg: MonolithCfg | None = None
    if "monolith" in raw:
        monolith_raw = {
            k: float(v) if k in _MONOLITH_FLOAT_FIELDS else v
            for k, v in raw["monolith"].items()
        }
        monolith_cfg = MonolithCfg(**monolith_raw)
    return Config(
        name=raw["name"],
        seed=raw["seed"],
        data=DataCfg(**raw["data"]),
        tools=ToolsCfg(**tools_raw),
        agent=agent_cfg,
        monolith=monolith_cfg,
    )
