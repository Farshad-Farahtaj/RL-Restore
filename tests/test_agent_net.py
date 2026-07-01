"""Tests for AgentNet (src/rlrestore/agent/net.py).

Tests per task A2 / fidelity_spec §2 + §5 pitfalls P1+P2.

(a) spatial trace  : flatten dim is exactly 384
(b) P2 wiring      : lstm.input_size==44, q_head.out_features==13,
                     forward_step rejects 13-dim prev-action
(c) 1-step consistency : forward_step(hx=None) Q == forward_episode_batch Q
                         on a single-step episode  (atol 1e-6)
(d) P1 LSTM-state  : fresh hx=None == explicit zeros (exact);
                     differs from carried hx from a 2-step rollout
(e) multi-step unroll : 3-step step-by-step with carried hx equals
                        forward_episode_batch on the stacked episode (atol 1e-5)
(f) determinism    : same seed -> identical init -> identical outputs
"""

from __future__ import annotations

import pytest
import torch

from rlrestore.agent.net import AgentNet

# ─────────────────────────── fixtures ────────────────────────────────────────


@pytest.fixture
def net() -> AgentNet:
    """Fresh AgentNet on CPU with a fixed seed."""
    torch.manual_seed(42)
    return AgentNet()


def _rand_img(batch: int = 1, seed: int = 0) -> torch.Tensor:
    """Random (batch, 3, 63, 63) float32 image tensor."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 3, 63, 63, generator=g)


def _rand_oh(batch: int = 1, seed: int = 1) -> torch.Tensor:
    """Random (batch, 12) float32 one-hot-ish vector."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 12, generator=g)


def _zero_oh(batch: int = 1) -> torch.Tensor:
    return torch.zeros(batch, 12)


# ─────────────────────────── (a) spatial trace ────────────────────────────────


def test_flatten_dim_param(net: AgentNet) -> None:
    """(a) fc layer in_features must be 384 (24*4*4 from 63x63 input)."""
    assert net.fc.in_features == 384, (
        f"Expected fc.in_features==384, got {net.fc.in_features}"
    )


def test_flatten_dim_forward(net: AgentNet) -> None:
    """(a) Actual forward confirms flatten produces 384 before fc."""
    activation: dict[str, torch.Tensor] = {}

    def hook(module: torch.nn.Module, inp: tuple, out: torch.Tensor) -> None:
        activation["flat"] = out.detach()

    handle = net.fc.register_forward_pre_hook(
        lambda mod, inp: activation.update({"flat": inp[0].detach()})
    )
    try:
        img = _rand_img(1)
        oh = _zero_oh(1)
        net.forward_step(img, oh, None)
    finally:
        handle.remove()

    assert "flat" in activation
    assert activation["flat"].shape == (1, 384), (
        f"Expected (1,384) before fc, got {activation['flat'].shape}"
    )


def test_conv_output_channels(net: AgentNet) -> None:
    """(a) Verify the conv tower output shape explicitly before flatten."""
    # conv4 output should be (1, 24, 4, 4)
    x = _rand_img(1)
    with torch.no_grad():
        y = net.conv(x)
    assert y.shape == (1, 24, 4, 4), f"Expected (1,24,4,4), got {y.shape}"


# ─────────────────────────── (b) P2 wiring ───────────────────────────────────


def test_lstm_input_size(net: AgentNet) -> None:
    """(b/P2) lstm.input_size must be 44 (32 + 12)."""
    assert net.lstm.input_size == 44, (
        f"Expected lstm.input_size==44, got {net.lstm.input_size}"
    )


def test_q_head_out_features(net: AgentNet) -> None:
    """(b/P2) q_head output must be 13 actions."""
    assert net.q_head.out_features == 13, (
        f"Expected q_head.out_features==13, got {net.q_head.out_features}"
    )


def test_forward_step_rejects_13dim_prev_action(net: AgentNet) -> None:
    """(b/P2) prev-action is 12-dim NOT 13; forward_step must raise on 13-dim."""
    img = _rand_img(1)
    wrong_oh = torch.zeros(1, 13)  # 13-dim — must fail
    with pytest.raises((RuntimeError, ValueError)):
        net.forward_step(img, wrong_oh, None)


def test_forward_step_accepts_12dim_prev_action(net: AgentNet) -> None:
    """(b) forward_step succeeds with correct 12-dim prev-action."""
    img = _rand_img(1)
    oh = _zero_oh(1)
    q, hx = net.forward_step(img, oh, None)
    assert q.shape == (1, 13)


# ─────────────────────────── (c) 1-step consistency ─────────────────────────


def test_forward_step_vs_episode_batch_1step(net: AgentNet) -> None:
    """(c) Single step: forward_step(hx=None) == forward_episode_batch."""
    torch.manual_seed(7)
    img = _rand_img(1, seed=10)  # shape (1, 3, 63, 63)
    oh = _rand_oh(1, seed=11)   # shape (1, 12)

    # forward_step path
    net.eval()
    with torch.no_grad():
        q_step, _ = net.forward_step(img, oh, None)  # (1, 13)

    # forward_episode_batch path: G=1, T=1
    # imgs shape: (G, T, 3, 63, 63) = (1, 1, 3, 63, 63)
    # prev_oh shape: (G, T, 12) = (1, 1, 12)
    imgs_batch = img.unsqueeze(1)    # (1, 1, 3, 63, 63)
    oh_batch = oh.unsqueeze(1)       # (1, 1, 12)
    with torch.no_grad():
        q_batch = net.forward_episode_batch(imgs_batch, oh_batch)  # (1, 13)

    assert q_step.shape == (1, 13)
    assert q_batch.shape == (1, 13)
    assert torch.allclose(q_step, q_batch, atol=1e-6), (
        f"Max diff: {(q_step - q_batch).abs().max().item()}"
    )


# ─────────────────────────── (d) P1 LSTM state ───────────────────────────────


def test_fresh_hx_none_equals_explicit_zeros(net: AgentNet) -> None:
    """(d/P1) hx=None must produce identical Q to explicit zero hx."""
    img = _rand_img(1, seed=20)
    oh = _zero_oh(1)
    net.eval()

    with torch.no_grad():
        q_none, _ = net.forward_step(img, oh, None)

    # Explicit zeros: hx = (h0, c0) both shape (1, 1, 50)
    h0 = torch.zeros(1, 1, 50)
    c0 = torch.zeros(1, 1, 50)
    with torch.no_grad():
        q_zeros, _ = net.forward_step(img, oh, (h0, c0))

    assert torch.equal(q_none, q_zeros), (
        f"hx=None vs explicit zeros differ; max diff: {(q_none - q_zeros).abs().max().item()}"
    )


def test_carried_hx_differs_from_fresh(net: AgentNet) -> None:
    """(d/P1) Carrying hx from a 2-step rollout changes Q at the next step."""
    img_a = _rand_img(1, seed=30)
    img_b = _rand_img(1, seed=31)
    img_c = _rand_img(1, seed=32)
    oh = _zero_oh(1)
    net.eval()

    with torch.no_grad():
        # Run 2 steps to build up state
        _, hx1 = net.forward_step(img_a, oh, None)
        _, hx2 = net.forward_step(img_b, oh, hx1)

        # Q at step 3 with carried hx vs fresh hx
        q_carried, _ = net.forward_step(img_c, oh, hx2)
        q_fresh, _ = net.forward_step(img_c, oh, None)

    assert not torch.equal(q_carried, q_fresh), (
        "Carried hx must produce different Q than fresh hx=None"
    )


# ─────────────────────────── (e) multi-step unroll equivalence ───────────────


def test_3step_stepwise_equals_episode_batch(net: AgentNet) -> None:
    """(e) 3-step step-by-step with carried hx == forward_episode_batch.

    This is the key unroll-equivalence test: the episode_batch must produce
    the SAME per-step Q values as manual step-by-step rollout.
    """
    torch.manual_seed(99)
    imgs = [_rand_img(1, seed=i) for i in range(3)]   # 3 x (1,3,63,63)
    ohs = [_rand_oh(1, seed=i + 10) for i in range(3)]  # 3 x (1,12)

    net.eval()

    # Step-by-step with carried hx
    with torch.no_grad():
        hx = None
        q_steps = []
        for img, oh in zip(imgs, ohs):
            q, hx = net.forward_step(img, oh, hx)
            q_steps.append(q)  # each (1, 13)

    q_steps_cat = torch.cat(q_steps, dim=0)  # (3, 13)

    # forward_episode_batch: G=1, T=3
    imgs_batch = torch.stack(imgs, dim=1)  # (1, 3, 3, 63, 63) — wait, need care
    # imgs[i] is (1, 3, 63, 63); stack along dim=1 -> (1, 3, 3, 63, 63)? No.
    # imgs[i] has shape (1, 3, 63, 63); we want (G=1, T=3, 3, 63, 63).
    # torch.stack(imgs, dim=1) stacks along new dim 1: gives (1, 3, 3, 63, 63).
    # That's (G, T, C, H, W) = (1, 3, 3, 63, 63) — correct.
    ohs_batch = torch.stack(ohs, dim=1)  # (1, 3, 12)

    with torch.no_grad():
        q_batch = net.forward_episode_batch(imgs_batch, ohs_batch)  # (3, 13)

    assert q_batch.shape == (3, 13), f"Expected (3,13), got {q_batch.shape}"
    assert q_steps_cat.shape == (3, 13)
    assert torch.allclose(q_batch, q_steps_cat, atol=1e-5), (
        f"Max diff: {(q_batch - q_steps_cat).abs().max().item()}"
    )


# ─────────────────────────── (f) determinism ─────────────────────────────────


def test_determinism_same_seed(net: AgentNet) -> None:
    """(f) Same seed -> identical initialization -> identical outputs."""
    torch.manual_seed(42)
    net1 = AgentNet()
    torch.manual_seed(42)
    net2 = AgentNet()

    # Verify parameter identity
    for (n1, p1), (n2, p2) in zip(net1.named_parameters(), net2.named_parameters()):
        assert torch.equal(p1, p2), f"Parameter {n1} differs between identically seeded nets"

    # Verify output identity
    img = _rand_img(1, seed=50)
    oh = _zero_oh(1)
    net1.eval()
    net2.eval()
    with torch.no_grad():
        q1, _ = net1.forward_step(img, oh, None)
        q2, _ = net2.forward_step(img, oh, None)

    assert torch.equal(q1, q2), "Identically seeded nets must produce identical outputs"


def test_n_params(net: AgentNet) -> None:
    """n_params() returns a positive integer."""
    n = net.n_params()
    assert isinstance(n, int)
    assert n > 0


# ─────────────────────────── (g) G=2 multi-episode parity ───────────────────

def test_multi_episode_batch_parity_g2() -> None:
    """(g) G=2 episodes, T=3 steps: batch forward == step-by-step per episode.

    Tests that forward_episode_batch produces identical Q values to manual
    step-by-step rollout when processing multiple independent episodes.
    """
    torch.manual_seed(3)
    net = AgentNet()
    net.eval()
    imgs = torch.rand(2, 3, 3, 63, 63)     # G=2 episodes, T=3 steps
    prev = torch.zeros(2, 3, 12)
    prev[0, 1, 4] = 1.0                     # episode 0 took action 4 at t=0
    prev[0, 2, 7] = 1.0
    prev[1, 1, 2] = 1.0                     # episode 1 differs
    prev[1, 2, 9] = 1.0
    with torch.no_grad():
        q_batch = net.forward_episode_batch(imgs, prev).view(2, 3, 13)
        for g in range(2):
            hx = None
            for t in range(3):
                q_step, hx = net.forward_step(
                    imgs[g, t : t + 1], prev[g, t : t + 1], hx
                )
                assert torch.allclose(q_batch[g, t], q_step.squeeze(0), atol=1e-5), (
                    f"episode {g} step {t}: batch and stepwise Q diverge"
                )
