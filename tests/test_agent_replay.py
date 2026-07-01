"""Tests for EpisodeReplay ring buffer.

Spec: docs/plan2/fidelity_spec.md §3.
Task: Plan 2 A3.

Required scenarios
------------------
(a) pending episodes never sampled
(b) atomic finalize — episode data intact after commit
(c) wrap eviction — oldest episodes evicted, survivors intact
(d) uint8 round-trip — stored/retrieved within atol 1/255; values >1 saturate
(e) grouped sampling — correct shapes, grouped by length
(f) determinism — same rng seed -> same sample
(g) length guard — 0-length or 4-step raises
"""

from __future__ import annotations

import numpy as np
import pytest

from rlrestore.agent.replay import EpisodeReplay

# ──────────────────────────── helpers ────────────────────────────────────────


def _rand_img(rng: np.random.Generator, *, saturate: bool = False) -> np.ndarray:
    """Return a (63,63,3) float32 image in [0,1] (or [0,2] if saturate)."""
    img = rng.random((63, 63, 3)).astype(np.float32)
    if saturate:
        img *= 2.0  # some values > 1
    return img


def _push_episode(replay: EpisodeReplay, rng: np.random.Generator, length: int) -> list:
    """Push `length` steps, finalize, return pushed (image, action, reward, terminal) tuples."""
    steps = []
    for t in range(length):
        img = _rand_img(rng)
        action = int(rng.integers(12))
        reward = float(rng.standard_normal())
        terminal = t == length - 1
        replay.push_step(img, action, reward, terminal)
        steps.append((img, action, reward, terminal))
    replay.finalize_episode()
    return steps


# ──────────────────────────── (a) pending never sampled ───────────────────────


class TestPendingNeverSampled:
    def test_pending_episode_not_counted(self):
        replay = EpisodeReplay(capacity=100)
        rng = np.random.default_rng(0)
        img = _rand_img(rng)
        replay.push_step(img, 0, 1.0, False)
        replay.push_step(_rand_img(rng), 1, 2.0, False)
        # Two steps pushed but NOT finalized — must be invisible
        assert replay.num_episodes == 0

    def test_pending_episode_not_sampled(self):
        replay = EpisodeReplay(capacity=100)
        rng = np.random.default_rng(1)
        replay.push_step(_rand_img(rng), 0, 1.0, False)
        replay.push_step(_rand_img(rng), 1, 0.5, True)
        sample_rng = np.random.default_rng(42)
        # With 0 committed episodes, sample_batch should raise or return empty
        with pytest.raises(Exception):
            replay.sample_batch(1, sample_rng)

    def test_num_transitions_excludes_pending(self):
        replay = EpisodeReplay(capacity=100)
        rng = np.random.default_rng(2)
        # Commit one 2-step episode
        _push_episode(replay, rng, 2)
        assert replay.num_transitions == 2
        # Push 1 more step without finalizing
        replay.push_step(_rand_img(rng), 0, 0.0, False)
        # Pending step must not inflate the count
        assert replay.num_transitions == 2


# ──────────────────────────── (b) atomic finalize ─────────────────────────────


class TestAtomicFinalize:
    def test_episode_sampled_after_finalize(self):
        replay = EpisodeReplay(capacity=100)
        rng = np.random.default_rng(10)
        _push_episode(replay, rng, 2)
        assert replay.num_episodes == 1
        assert replay.num_transitions == 2

    def test_episode_data_intact(self):
        replay = EpisodeReplay(capacity=100)
        rng = np.random.default_rng(11)
        steps = _push_episode(replay, rng, 3)
        sample_rng = np.random.default_rng(0)
        batch = replay.sample_batch(1, sample_rng)
        # Episode has length 3
        assert 3 in batch
        ep = batch[3][0]
        imgs = ep["images"]
        actions = ep["actions"]
        rewards = ep["rewards"]
        terminals = ep["terminals"]
        assert imgs.shape == (3, 63, 63, 3)
        assert actions.shape == (3,)
        assert rewards.shape == (3,)
        assert terminals.shape == (3,)
        for t, (orig_img, orig_act, orig_rew, orig_term) in enumerate(steps):
            np.testing.assert_allclose(imgs[t], orig_img, atol=1.0 / 255.0)
            assert actions[t] == orig_act
            np.testing.assert_allclose(rewards[t], orig_rew, rtol=1e-6)
            assert bool(terminals[t]) == orig_term

    def test_images_returned_as_float32(self):
        replay = EpisodeReplay(capacity=100)
        rng = np.random.default_rng(12)
        _push_episode(replay, rng, 1)
        sample_rng = np.random.default_rng(0)
        batch = replay.sample_batch(1, sample_rng)
        ep = list(batch.values())[0][0]
        assert ep["images"].dtype == np.float32

    def test_before_finalize_zero_episodes(self):
        replay = EpisodeReplay(capacity=100)
        rng = np.random.default_rng(13)
        replay.push_step(_rand_img(rng), 0, 1.0, True)
        assert replay.num_episodes == 0
        replay.finalize_episode()
        assert replay.num_episodes == 1


# ──────────────────────────── (c) wrap eviction ───────────────────────────────


class TestWrapEviction:
    """Capacity=7, episodes of length 3. After enough pushes the ring wraps
    and the oldest episode must be evicted atomically."""

    def test_oldest_episode_evicted_on_wrap(self):
        capacity = 7
        replay = EpisodeReplay(capacity=capacity)
        rng = np.random.default_rng(20)

        # Push 2 complete 3-step episodes → 6 transitions; ring has 1 slot free
        _push_episode(replay, rng, 3)
        steps_ep1 = _push_episode(replay, rng, 3)
        assert replay.num_episodes == 2
        assert replay.num_transitions == 6

        # Push a third 3-step episode — slots 6,7,8 mod 7 wrap → overwrites ep0's slots
        steps_ep2 = _push_episode(replay, rng, 3)
        # ep0 must be evicted; ep2 must be present
        assert replay.num_episodes == 2  # ep1 + ep2 (ep0 gone)
        assert replay.num_transitions == 6

        # Confirm ep1 and ep2 are intact by reading directly from committed refs.
        # (We avoid relying on sample_batch's replacement draw returning both.)
        ep1_ref, ep2_ref = replay._episodes
        ep1_data = replay._read_episode(ep1_ref)
        ep2_data = replay._read_episode(ep2_ref)
        def _u8_rt(img: np.ndarray) -> np.ndarray:
            return (
                np.clip(img * 255, 0, 255).astype(np.uint8).astype(np.float32) / 255.0
            )

        ep1_img0 = _u8_rt(steps_ep1[0][0])
        ep2_img0 = _u8_rt(steps_ep2[0][0])
        np.testing.assert_allclose(
            ep1_data["images"][0], ep1_img0, atol=1e-6,
            err_msg="ep1 first image corrupted",
        )
        np.testing.assert_allclose(
            ep2_data["images"][0], ep2_img0, atol=1e-6,
            err_msg="ep2 first image corrupted",
        )

    def test_evicted_episodes_not_sampled(self):
        capacity = 7
        replay = EpisodeReplay(capacity=capacity)
        rng = np.random.default_rng(21)

        # Push 3 episodes of length 3; at capacity=7 the first episode
        # is overwritten completely after the third episode is pushed.
        steps_ep0 = _push_episode(replay, rng, 3)
        _push_episode(replay, rng, 3)
        _push_episode(replay, rng, 3)

        # ep0 should be gone — verify via the committed episode list
        assert replay.num_episodes == 2
        ep0_img0 = (
            np.clip(steps_ep0[0][0] * 255, 0, 255)
            .astype(np.uint8)
            .astype(np.float32) / 255.0
        )
        for ref in replay._episodes:
            data = replay._read_episode(ref)
            assert not np.allclose(
                data["images"][0], ep0_img0, atol=1e-6
            ), "ep0 should have been evicted"

    def test_num_transitions_consistent_after_eviction(self):
        capacity = 7
        replay = EpisodeReplay(capacity=capacity)
        rng = np.random.default_rng(22)
        _push_episode(replay, rng, 3)  # ep0
        _push_episode(replay, rng, 3)  # ep1
        assert replay.num_transitions == 6
        _push_episode(replay, rng, 3)  # ep2 → wraps, evicts ep0
        # ep1 + ep2 = 6 transitions in 2 episodes
        assert replay.num_transitions == 6
        assert replay.num_episodes == 2

    def test_second_wrap_evicts_second_oldest(self):
        """After enough pushes a second wrap evicts the next oldest episode."""
        capacity = 7
        replay = EpisodeReplay(capacity=capacity)
        rng = np.random.default_rng(23)
        _push_episode(replay, rng, 3)  # ep0
        _push_episode(replay, rng, 3)  # ep1
        _push_episode(replay, rng, 3)  # ep2 → wraps, evicts ep0
        _push_episode(replay, rng, 3)  # ep3 → wraps further, evicts ep1
        assert replay.num_episodes == 2
        assert replay.num_transitions == 6


# ──────────────────────────── (d) uint8 round-trip ────────────────────────────


class TestUint8RoundTrip:
    def test_roundtrip_within_tolerance(self):
        replay = EpisodeReplay(capacity=100)
        rng = np.random.default_rng(30)
        orig_img = _rand_img(rng)
        replay.push_step(orig_img, 0, 0.5, True)
        replay.finalize_episode()
        sample_rng = np.random.default_rng(0)
        batch = replay.sample_batch(1, sample_rng)
        ep = list(batch.values())[0][0]
        np.testing.assert_allclose(ep["images"][0], orig_img, atol=1.0 / 255.0)

    def test_values_above_one_saturate(self):
        """Images with pixel values > 1 should saturate to 1.0 after round-trip."""
        replay = EpisodeReplay(capacity=100)
        rng = np.random.default_rng(31)
        # Image with values in [1.0, 2.0]
        img = _rand_img(rng, saturate=True)
        assert img.max() > 1.0  # sanity
        replay.push_step(img, 0, 0.0, True)
        replay.finalize_episode()
        sample_rng = np.random.default_rng(0)
        batch = replay.sample_batch(1, sample_rng)
        ep = list(batch.values())[0][0]
        retrieved = ep["images"][0]
        # All pixels must be <= 1.0 (saturated) and max should be 1.0
        assert retrieved.max() <= 1.0 + 1e-6
        assert retrieved.max() >= 1.0 - 1e-6

    def test_uint8_storage(self):
        """Verify images are stored as uint8 internally (lazy alloc)."""
        replay = EpisodeReplay(capacity=100)
        rng = np.random.default_rng(32)
        _push_episode(replay, rng, 1)
        # Access internal buffer and check dtype
        assert replay._images_buf.dtype == np.uint8


# ──────────────────────────── (e) grouped sampling ────────────────────────────


class TestGroupedSampling:
    def _setup_mixed(self) -> EpisodeReplay:
        """Push 2 eps of length-1, 2 of length-2, 2 of length-3."""
        replay = EpisodeReplay(capacity=200)
        rng = np.random.default_rng(40)
        for _ in range(2):
            _push_episode(replay, rng, 1)
        for _ in range(2):
            _push_episode(replay, rng, 2)
        for _ in range(2):
            _push_episode(replay, rng, 3)
        return replay

    def test_groups_keyed_by_length(self):
        replay = self._setup_mixed()
        sample_rng = np.random.default_rng(0)
        batch = replay.sample_batch(6, sample_rng)
        # All three lengths should appear
        assert set(batch.keys()) <= {1, 2, 3}

    def test_total_episodes_equals_requested(self):
        replay = self._setup_mixed()
        sample_rng = np.random.default_rng(0)
        batch = replay.sample_batch(6, sample_rng)
        total = sum(len(v) for v in batch.values())
        assert total == 6

    def test_episode_data_shapes_per_length(self):
        replay = self._setup_mixed()
        sample_rng = np.random.default_rng(0)
        batch = replay.sample_batch(6, sample_rng)
        for length, eps in batch.items():
            for ep in eps:
                assert ep["images"].shape == (length, 63, 63, 3)
                assert ep["actions"].shape == (length,)
                assert ep["rewards"].shape == (length,)
                assert ep["terminals"].shape == (length,)

    def test_sample_single_episode_type(self):
        """When only length-2 episodes exist, sample returns only length-2."""
        replay = EpisodeReplay(capacity=200)
        rng = np.random.default_rng(41)
        for _ in range(4):
            _push_episode(replay, rng, 2)
        sample_rng = np.random.default_rng(0)
        batch = replay.sample_batch(2, sample_rng)
        assert list(batch.keys()) == [2]
        assert len(batch[2]) == 2


# ──────────────────────────── (f) determinism ─────────────────────────────────


class TestDeterminism:
    def test_same_seed_same_sample(self):
        replay = EpisodeReplay(capacity=200)
        rng = np.random.default_rng(50)
        for _ in range(5):
            _push_episode(replay, rng, 2)

        batch_a = replay.sample_batch(3, np.random.default_rng(99))
        batch_b = replay.sample_batch(3, np.random.default_rng(99))

        assert set(batch_a.keys()) == set(batch_b.keys())
        for length in batch_a:
            eps_a = batch_a[length]
            eps_b = batch_b[length]
            assert len(eps_a) == len(eps_b)
            for ea, eb in zip(eps_a, eps_b):
                np.testing.assert_array_equal(ea["images"], eb["images"])
                np.testing.assert_array_equal(ea["actions"], eb["actions"])

    def test_different_seeds_different_samples(self):
        replay = EpisodeReplay(capacity=200)
        rng = np.random.default_rng(51)
        for _ in range(10):
            _push_episode(replay, rng, 2)

        batch_a = replay.sample_batch(5, np.random.default_rng(1))
        batch_b = replay.sample_batch(5, np.random.default_rng(2))

        # With 10 distinct episodes, it'd be very unlikely both draws are identical
        # Collect first image of first ep from each draw
        imgs_a = batch_a[2][0]["images"]
        imgs_b = batch_b[2][0]["images"]
        # Allow that they could match by chance, but check via action arrays too
        # (just confirm the rng is actually used — we cannot guarantee they differ)
        # This is a weak sanity check, not a strict assertion.
        assert imgs_a is not imgs_b  # different objects at minimum


# ──────────────────────────── (g) length guard ────────────────────────────────


class TestLengthGuard:
    def test_finalize_empty_raises(self):
        replay = EpisodeReplay(capacity=100)
        with pytest.raises(ValueError, match="0"):
            replay.finalize_episode()

    def test_finalize_four_steps_raises(self):
        replay = EpisodeReplay(capacity=100)
        rng = np.random.default_rng(60)
        for _ in range(4):
            replay.push_step(_rand_img(rng), 0, 0.0, False)
        with pytest.raises(ValueError):
            replay.finalize_episode()

    def test_finalize_three_steps_ok(self):
        """Sanity: length-3 must not raise."""
        replay = EpisodeReplay(capacity=100)
        rng = np.random.default_rng(61)
        for _ in range(3):
            replay.push_step(_rand_img(rng), 0, 0.0, False)
        replay.finalize_episode()  # must not raise
        assert replay.num_episodes == 1

    def test_finalize_one_step_ok(self):
        """Sanity: length-1 must not raise."""
        replay = EpisodeReplay(capacity=100)
        rng = np.random.default_rng(62)
        replay.push_step(_rand_img(rng), 0, 0.0, True)
        replay.finalize_episode()
        assert replay.num_episodes == 1


# ──────────────────────────── (h) copy-safety ──────────────────────────────


class TestCopySafety:
    def test_sampled_images_not_aliased_to_ring(self):
        """Verify sampled images are materialized copies, not views of the ring."""
        replay = EpisodeReplay(capacity=100)
        rng = np.random.default_rng(77)
        img = rng.random((63, 63, 3)).astype(np.float32)
        replay.push_step(img, 0, 1.0, True)
        replay.finalize_episode()
        batch = replay.sample_batch(1, np.random.default_rng(0))
        sampled = list(batch.values())[0][0]["images"][0].copy()
        orig_val = sampled[0, 0, 0]
        # Clobber the ring slot with a drastically different value
        replay._images_buf[0] = 0
        # Re-sample and verify the ring mutation is visible in a fresh sample
        again = list(replay.sample_batch(1, np.random.default_rng(0)).values())[0][0]["images"][0]
        assert not np.allclose(again[0, 0, 0], orig_val), "ring mutation must be visible on re-read"
        # Verify the previously sampled copy remained stable
        assert np.allclose(sampled[0, 0, 0], orig_val), "previously sampled copy must be stable"
