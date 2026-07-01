"""Episode ring-buffer for RL-Restore DQN training.

Spec: docs/plan2/fidelity_spec.md §3.

Design
------
- Flat ring arrays lazily allocated on first push:
    _images_buf  : (capacity, 63, 63, 3)  uint8   – pixels stored as (x*255).clip(0,255)
    _actions_buf : (capacity,)             uint8
    _rewards_buf : (capacity,)             float32
    _terminals_buf: (capacity,)            bool

- An episode is only visible to sampling after `finalize_episode()`.
  Steps accumulate in a pending list; finalize writes them atomically.

- Ring wrap: when writing new transitions would collide with the oldest
  committed EpisodeRef, that entire episode is evicted before writing.
  Episodes are never partially valid.

- `sample_batch(n_episodes, rng)` returns a dict keyed by episode length;
  each value is a list of dicts with numpy arrays for images/actions/rewards/
  terminals.  Uses the provided Generator for reproducible draws.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# Image dimensions (must match env contract)
_H, _W, _C = 63, 63, 3
_MIN_EP_LEN = 1
_MAX_EP_LEN = 3


@dataclass
class EpisodeRef:
    """Points to a committed episode inside the flat ring."""

    start: int  # index of first transition in the ring
    length: int  # 1, 2, or 3


class EpisodeReplay:
    """Flat ring buffer that stores complete episodes atomically.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored simultaneously.  The spec
        default is 500_000; tests may use smaller values.
    """

    def __init__(self, capacity: int) -> None:
        self._cap = capacity

        # Lazy allocation – None until first push.
        self._images_buf: np.ndarray | None = None
        self._actions_buf: np.ndarray | None = None
        self._rewards_buf: np.ndarray | None = None
        self._terminals_buf: np.ndarray | None = None

        # Ring write head (next slot to write into).
        self._write_head: int = 0

        # Committed episodes (oldest first).
        self._episodes: deque[EpisodeRef] = deque()

        # Running transition count (incremented on finalize, decremented on eviction).
        self._n_transitions: int = 0

        # Pending (uncommitted) steps accumulated since last finalize.
        self._pending_images: list[np.ndarray] = []  # list of (63,63,3) uint8
        self._pending_actions: list[int] = []
        self._pending_rewards: list[float] = []
        self._pending_terminals: list[bool] = []

    # ─────────────────────────── public write API ────────────────────────────

    def push_step(
        self,
        image_hwc_float: np.ndarray,
        action: int,
        reward: float,
        terminal: bool,
    ) -> None:
        """Accumulate one transition into the pending episode.

        Args:
            image_hwc_float: (63, 63, 3) float32 in any range; clipped to
                             [0, 255] and stored as uint8.
            action: integer action index (stored as uint8).
            reward: float reward.
            terminal: whether this is the last step of the episode.
        """
        self._ensure_allocated()
        img_u8 = (image_hwc_float * 255.0).clip(0, 255).astype(np.uint8)
        self._pending_images.append(img_u8)
        self._pending_actions.append(int(action))
        self._pending_rewards.append(float(reward))
        self._pending_terminals.append(bool(terminal))

    def finalize_episode(self) -> EpisodeRef:
        """Commit the pending steps as a single atomic episode.

        Returns
        -------
        EpisodeRef
            Reference to the newly committed episode.

        Raises
        ------
        ValueError
            If the pending episode length is 0 or > 3.
        """
        length = len(self._pending_images)
        if length < _MIN_EP_LEN or length > _MAX_EP_LEN:
            # Clear pending so the buffer is not left in a broken state.
            self._pending_images.clear()
            self._pending_actions.clear()
            self._pending_rewards.clear()
            self._pending_terminals.clear()
            raise ValueError(
                f"Episode length must be {_MIN_EP_LEN}–{_MAX_EP_LEN}, got {length}."
            )

        self._ensure_allocated()

        # Slots this episode would occupy [write_head, write_head+length).
        start = self._write_head

        # Evict oldest committed episodes whose slots overlap with [start, start+length).
        self._evict_for_write(start, length)

        # Write transitions into the ring.
        for i, (img, act, rew, term) in enumerate(
            zip(
                self._pending_images,
                self._pending_actions,
                self._pending_rewards,
                self._pending_terminals,
            )
        ):
            slot = (start + i) % self._cap
            self._images_buf[slot] = img
            self._actions_buf[slot] = act
            self._rewards_buf[slot] = rew
            self._terminals_buf[slot] = term

        self._write_head = (start + length) % self._cap

        # Clear pending.
        self._pending_images.clear()
        self._pending_actions.clear()
        self._pending_rewards.clear()
        self._pending_terminals.clear()

        ref = EpisodeRef(start=start, length=length)
        self._episodes.append(ref)
        self._n_transitions += length
        return ref

    def patch_last_step(self, reward: float, terminal: bool) -> None:
        """Overwrite reward/terminal of the most recently pushed (pending) step."""
        if not self._pending_rewards:
            raise RuntimeError("no pending step to patch")
        self._pending_rewards[-1] = float(reward)
        self._pending_terminals[-1] = bool(terminal)

    # ─────────────────────────── public read API ─────────────────────────────

    @property
    def num_episodes(self) -> int:
        """Number of committed (visible) episodes."""
        return len(self._episodes)

    @property
    def num_transitions(self) -> int:
        """Total transitions across all committed episodes."""
        return self._n_transitions

    def sample_batch(
        self, n_episodes: int, rng: np.random.Generator
    ) -> dict[int, list[dict[str, np.ndarray]]]:
        """Sample n_episodes episodes, grouped by length.

        Parameters
        ----------
        n_episodes : int
            How many episodes to sample (with replacement from committed pool).
        rng : np.random.Generator
            Deterministic random source — must use `rng.choice`.

        Returns
        -------
        dict[int, list[dict]]
            Keys are episode lengths; values are lists of episode data dicts
            with keys "images" (T,63,63,3) float32, "actions" (T,),
            "rewards" (T,), "terminals" (T,).

        NOTE: returns materialized arrays (dict per episode), not EpisodeRefs —
        deliberate deviation from fidelity_spec §3 wording; spec updated accordingly.

        Raises
        ------
        ValueError
            If there are no committed episodes to sample from.
        """
        if not self._episodes:
            raise ValueError("No committed episodes to sample from.")

        # Convert to list once per call for efficient indexing.
        ep_list = list(self._episodes)
        indices = rng.choice(len(ep_list), size=n_episodes, replace=True)
        selected = [ep_list[i] for i in indices]

        # Group by length.
        result: dict[int, list[dict[str, np.ndarray]]] = {}
        for ref in selected:
            ep_data = self._read_episode(ref)
            result.setdefault(ref.length, []).append(ep_data)

        return result

    # ─────────────────────────── private helpers ─────────────────────────────

    def _ensure_allocated(self) -> None:
        if self._images_buf is None:
            self._images_buf = np.empty((self._cap, _H, _W, _C), dtype=np.uint8)
            self._actions_buf = np.empty((self._cap,), dtype=np.uint8)
            self._rewards_buf = np.empty((self._cap,), dtype=np.float32)
            self._terminals_buf = np.empty((self._cap,), dtype=bool)

    def _slots_of(self, ref: EpisodeRef) -> list[int]:
        """Return the flat ring slot indices occupied by *ref*."""
        return [(ref.start + i) % self._cap for i in range(ref.length)]

    def _evict_for_write(self, start: int, length: int) -> None:
        """Evict committed episodes that overlap with [start, start+length)."""
        new_slots = set((start + i) % self._cap for i in range(length))
        # Pop from the front (oldest first) until no overlap remains.
        while self._episodes:
            oldest = self._episodes[0]
            oldest_slots = set(self._slots_of(oldest))
            if oldest_slots & new_slots:
                self._episodes.popleft()
                self._n_transitions -= oldest.length
            else:
                break

    def _read_episode(self, ref: EpisodeRef) -> dict[str, np.ndarray]:
        """Read one episode from the ring into numpy arrays."""
        slots = self._slots_of(ref)
        images = self._images_buf[slots].astype(np.float32) / 255.0
        actions = self._actions_buf[slots].copy()
        rewards = self._rewards_buf[slots].copy()
        terminals = self._terminals_buf[slots].copy()
        return {
            "images": images,
            "actions": actions,
            "rewards": rewards,
            "terminals": terminals,
        }
