"""Tests for RestoreEnv (Task A1).

All tool interactions use fake modules — no real checkpoints required except
test_frozen_tools which saves a tiny real SmallToolCNN in the trainer's format.

Fake tool zoo
-------------
_Identity         : returns input unchanged (PSNR unchanged)
_PlusBias(delta)  : adds a constant (positive = degrade, negative = may improve)
_TowardClean      : moves pixel values toward 0 (used when clean=zeros)
_AlwaysWorse      : adds large positive constant — always worsens PSNR vs zero-clean
"""


import numpy as np
import pytest
import torch
from torch import nn

from rlrestore.agent.env import RestoreEnv, load_tools

# ────────────────────────── fake tool modules ───────────────────────────────


class _Identity(nn.Module):
    """Does nothing — PSNR stays constant."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()


class _PlusBias(nn.Module):
    """Adds `delta` to every pixel."""

    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.delta


class _ScaleTowardZero(nn.Module):
    """Multiplies by `factor` (0 < factor < 1) — shrinks absolute value toward 0.

    When clean=zeros and degraded=+offset, each application improves PSNR
    monotonically because |current - clean| = |current| decreases.
    """

    def __init__(self, factor: float = 0.5):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor


# ────────────────────────── shared helpers ──────────────────────────────────

_N_TOOLS = 12


def _make_image_pair():
    """Return (clean, degraded) as 63x63 float32 HWC [0,1] arrays."""
    rng = np.random.default_rng(42)
    clean = rng.random((63, 63, 3), dtype=np.float64).astype(np.float32)
    degraded = np.clip(clean + 0.15 * rng.standard_normal((63, 63, 3)), 0.0, 1.0).astype(
        np.float32
    )
    return clean, degraded


def _neutral_tools(n: int = _N_TOOLS) -> list[nn.Module]:
    return [_Identity() for _ in range(n)]


def _degrading_tools(n: int = _N_TOOLS) -> list[nn.Module]:
    """Tools that add a large positive constant — always worsen PSNR vs clean=0."""
    return [_PlusBias(0.5) for _ in range(n)]


def _make_env(
    tools=None,
    training_mode: bool = True,
    seed: int = 0,
    images=None,
) -> RestoreEnv:
    if tools is None:
        tools = _neutral_tools()
    if images is None:
        clean, _deg = _make_image_pair()
        images = [clean]
    return RestoreEnv(
        images=images,
        tools=tools,
        device=torch.device("cpu"),
        seed=seed,
        training_mode=training_mode,
    )


# ────────────────────────── test (a): pitfall P4 ────────────────────────────


def test_p4_forced_terminal_reward_not_zeroed():
    """Three tool steps each give positive reward; reward at step 3 is NOT zeroed.

    Construction: clean = zeros, degraded = 0.3 (uniform offset).
    Tool = _ScaleTowardZero(0.5) — each step halves distance to 0 (= clean).
    PSNR(0.3, 0) < PSNR(0.15, 0) < PSNR(0.075, 0) → monotonically increasing.
    This is guaranteed by construction, independent of make_training_pair.
    """
    from rlrestore.common.metrics import psnr as _psnr

    clean = np.zeros((63, 63, 3), dtype=np.float32)
    degraded = np.full((63, 63, 3), 0.3, dtype=np.float32)

    # Verify our construction gives improving PSNR
    p0 = _psnr(degraded, clean)
    p1 = _psnr(degraded * 0.5, clean)
    p2 = _psnr(degraded * 0.25, clean)
    assert p1 > p0 and p2 > p1, "Construction check failed — PSNR not monotonically improving"

    tools = [_ScaleTowardZero(0.5) for _ in range(_N_TOOLS)]
    env = RestoreEnv(
        images=[clean],  # image pool (won't be used after we inject state)
        tools=tools,
        device=torch.device("cpu"),
        seed=0,
        training_mode=True,
    )
    env.reset(seed=0)

    # Inject our known (clean, degraded) pair directly into the env state
    env._clean = clean
    env._current = degraded.copy()
    env._initial_psnr = _psnr(degraded, clean)
    env._current_psnr = env._initial_psnr
    env._step_count = 0

    rewards = []
    terminated = False
    for _ in range(3):
        obs, reward, terminated, info = env.step(0)  # always tool 0
        rewards.append(reward)

    assert all(r > 0 for r in rewards), f"Expected all positive rewards, got {rewards}"
    assert terminated, "Expected terminated=True at step 3"
    assert rewards[2] > 0, (
        f"Pitfall P4: reward at forced terminal was zeroed! Got {rewards[2]}"
    )


# ────────────────────────── test (b): STOP action ───────────────────────────


def test_stop_semantics():
    """STOP: reward=0.0, terminated=True, prev_action_onehot all-zeros shape (12,)."""
    env = _make_env()
    env.reset(seed=0)

    obs, reward, terminated, info = env.step(12)  # STOP

    img_obs, prev_oh = obs
    assert reward == 0.0, f"STOP reward must be 0.0, got {reward}"
    assert terminated is True, "STOP must set terminated=True"
    assert prev_oh.shape == (_N_TOOLS,), (
        f"prev_action_onehot must be shape (12,), got {prev_oh.shape}"
    )
    assert np.all(prev_oh == 0.0), "STOP must produce all-zeros prev_action_onehot"


# ────────────────────────── test (c): prev-action one-hot ───────────────────


def test_prev_action_onehot_after_tool():
    """After tool k (k<12), one-hot has 1.0 at position k, shape (12,)."""
    for k in [0, 5, 11]:
        env_k = _make_env()
        env_k.reset(seed=0)
        obs, _r, _t, _i = env_k.step(k)
        _img, prev_oh = obs
        assert prev_oh.shape == (_N_TOOLS,), (
            f"prev_action_onehot shape wrong after tool {k}: {prev_oh.shape}"
        )
        assert prev_oh[k] == 1.0, f"Expected 1.0 at index {k}, got {prev_oh[k]}"
        assert np.sum(prev_oh) == 1.0, (
            f"One-hot sum must be 1, got {np.sum(prev_oh)} for tool {k}"
        )


def test_prev_action_onehot_at_reset():
    """At reset (t=0), prev_action_onehot should be all zeros."""
    env = _make_env()
    obs = env.reset(seed=0)
    _img, prev_oh = obs
    assert prev_oh.shape == (_N_TOOLS,)
    assert np.all(prev_oh == 0.0), "prev_action_onehot at t=0 must be all zeros"


# ────────────────────────── test (d): early abort training_mode only ────────


def test_early_abort_training_mode_only():
    """Early abort fires in training_mode but NOT in validation mode.

    Construction: clean=zeros, degraded=0.3, tools add 0.5 -> psnr(0.8, 0) < psnr(0.3, 0).
    The tool makes things worse; training_mode env terminates after step 1;
    validation env runs all 3 steps (step_count >= 3).
    """
    from rlrestore.common.metrics import psnr as _psnr

    clean = np.zeros((63, 63, 3), dtype=np.float32)
    degraded = np.full((63, 63, 3), 0.3, dtype=np.float32)
    initial_psnr = _psnr(degraded, clean)

    # Verify our degrading tool actually lowers PSNR
    after_tool = degraded + 0.5  # PlusBias(0.5) applied once
    assert _psnr(after_tool, clean) < initial_psnr, "Construction: tool must lower PSNR"

    tools = _degrading_tools()  # PlusBias(0.5)

    def _setup_env(training_mode: bool) -> RestoreEnv:
        env = RestoreEnv(
            images=[clean],
            tools=tools,
            device=torch.device("cpu"),
            seed=0,
            training_mode=training_mode,
        )
        env.reset(seed=0)
        env._clean = clean
        env._current = degraded.copy()
        env._initial_psnr = initial_psnr
        env._current_psnr = initial_psnr
        env._step_count = 0
        return env

    # ── training env: should terminate after 1 step with negative reward ──
    train_env = _setup_env(training_mode=True)
    obs1, reward1, terminated1, _ = train_env.step(0)
    assert terminated1 is True, "training_mode: early abort should terminate on first bad step"
    assert reward1 < 0, f"training_mode: degrading step should give negative reward, got {reward1}"

    # ── validation env: should NOT terminate early (runs all 3 steps) ──
    val_env = _setup_env(training_mode=False)
    _, _, t1, _ = val_env.step(0)
    _, _, t2, _ = val_env.step(0)
    _, _, t3, _ = val_env.step(0)

    assert t1 is False, "val_env: step 1 should NOT terminate early"
    assert t2 is False, "val_env: step 2 should NOT terminate early"
    assert t3 is True, "val_env: step 3 should terminate (forced terminal at step_count>=3)"


# ────────────────────────── test (e): determinism ───────────────────────────

def test_determinism_same_seed():
    """Two envs reset with the same seed produce identical obs and step results."""
    clean, _deg = _make_image_pair()
    images = [clean]
    tools = _neutral_tools()

    env_a = RestoreEnv(images=images, tools=tools, device=torch.device("cpu"), seed=0)
    env_b = RestoreEnv(images=images, tools=tools, device=torch.device("cpu"), seed=0)

    obs_a = env_a.reset(seed=7)
    obs_b = env_b.reset(seed=7)

    img_a, oh_a = obs_a
    img_b, oh_b = obs_b
    np.testing.assert_array_equal(img_a, img_b, err_msg="reset: current images differ")
    np.testing.assert_array_equal(oh_a, oh_b, err_msg="reset: prev_action_onehots differ")

    # Step with same action
    for action in [3, 7]:
        obs_a2, r_a, t_a, _ = env_a.step(action)
        obs_b2, r_b, t_b, _ = env_b.step(action)
        np.testing.assert_array_equal(obs_a2[0], obs_b2[0], err_msg=f"step {action}: obs differ")
        assert r_a == r_b, f"step {action}: rewards differ ({r_a} vs {r_b})"
        assert t_a == t_b


# ────────────────────────── test (f): frozen tools ──────────────────────────

def test_frozen_tools_load(tmp_path):
    """load_tools returns modules with requires_grad=False and in eval mode.

    Saves a real SmallToolCNN in the trainer's best.pt format (dict with 'model' key).
    """
    from rlrestore.tools.models import build_tool
    from rlrestore.tools.specs import TOOL_SPECS

    cfg_name = "test_cfg"
    n_tools = _N_TOOLS

    # Build the directory structure that load_tools expects:
    # checkpoints/tools/<cfg_name>/toolNN/best.pt
    for i in range(n_tools):
        tool_dir = tmp_path / "checkpoints" / "tools" / cfg_name / f"tool{i:02d}"
        tool_dir.mkdir(parents=True)
        model = build_tool(TOOL_SPECS[i])
        torch.save({"model": model.state_dict(), "spec_index": i, "val_psnr": 30.0},
                   tool_dir / "best.pt")

    # Temporarily point load_tools at tmp_path by passing it explicitly
    # load_tools signature: load_tools(cfg_name, device, ckpt_root=Path("checkpoints/tools"))
    loaded = load_tools(cfg_name, torch.device("cpu"), ckpt_root=tmp_path / "checkpoints" / "tools")

    assert len(loaded) == n_tools, f"Expected {n_tools} tools, got {len(loaded)}"

    for i, tool in enumerate(loaded):
        # All parameters must have requires_grad=False
        for name, param in tool.named_parameters():
            assert not param.requires_grad, (
                f"tool[{i}].{name}: requires_grad should be False (frozen)"
            )
        # Module must be in eval mode
        assert not tool.training, f"tool[{i}] should be in eval mode (training={tool.training})"


# ────────────────────────── test: obs structure ─────────────────────────────

def test_obs_is_tuple_of_arrays():
    """obs is a 2-tuple: (current HWC float32, prev_onehot shape (12,))."""
    env = _make_env()
    obs = env.reset(seed=0)
    assert isinstance(obs, tuple) and len(obs) == 2
    img, oh = obs
    assert isinstance(img, np.ndarray), "obs[0] must be numpy array"
    assert img.dtype == np.float32
    assert img.shape == (63, 63, 3)
    assert isinstance(oh, np.ndarray)
    assert oh.shape == (_N_TOOLS,)


def test_step_returns_correct_structure():
    """step returns (obs, reward, terminated, info) with correct types."""
    env = _make_env()
    env.reset(seed=0)
    result = env.step(0)
    assert len(result) == 4
    obs, reward, terminated, info = result
    assert isinstance(obs, tuple) and len(obs) == 2
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(info, dict)


# ────────────────────────── test: post-terminal guard ──────────────────────────


def test_step_after_terminal_raises():
    """Calling step() after episode termination raises RuntimeError."""
    env = _make_env(training_mode=True)
    env.reset(seed=0)
    env.step(12)  # STOP terminates
    with pytest.raises(RuntimeError, match="step\\(\\) called on terminated episode"):
        env.step(0)
