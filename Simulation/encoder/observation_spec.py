"""
Definition of the history encoder's input frame and its frozen normalization constants.

This is the single source of truth shared by all three consumers:
  1. the pretraining data driver
  2. the encoder training script
  3. LatentObsWrapper at PPO / rollout / evaluation time

Keeping it in one place is what prevents pretrain -> PPO normalization drift. The
constants are FROZEN (computed once on the pretraining dataset and file-loaded
thereafter); a running normalizer would silently rescale the encoder's inputs
between pretraining and PPO, and drift over time.

=============================================================================
INPUT FRAME CONTRACT  (must hold identically in every consumer)

    frame_t = [ o_t (29) | aux_t (4) ]        -> 33 dims

      o_t   : pos_est(3), quat(4), gyro(3), vel_xy_est(2), vel_z_est(1),
              prev_action(4), p_err(3), v_err(3), att_err(3), w_err(3)
      aux_t : specific_force_b(3), v_batt_norm

Both halves are latency-delayed by the SAME number of steps. The aux is NOT
interleaved into the actor frame stack - it is a separate block in the env
observation vector - so the actor observation stays a clean prefix of the vector
for any OBS_HISTORY_LEN.

There is deliberately NO separate applied-action channel. `o_t[13:17]` already
holds the POST-EMA action applied at t-1, which is precisely the causal pairing
the encoder needs. Concatenating the applied action again would be a provable
duplicate of those slots shifted by one step.

NOTE ON THE REFERENCE ERROR CHANNELS. p_err / v_err / att_err / w_err are part of
the encoder input on purpose. They are the task specification, so they tell the
encoder what the airframe is being ASKED to do - which is what lets it separate
"this vehicle is under-responsive" from "this vehicle is being asked for a violent
manoeuvre". Without them, a large tracking error during a flip is indistinguishable
from a large tracking error caused by a failing motor.
=============================================================================
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

ACTOR_FRAME_DIM: int = 29          # o_t (must equal ACTOR_SINGLE_OBS_DIM in quad_flip_env.py)
REF_FF_DIM: int = 3                # reference feed-forward block that sits between o_t and aux
AUX_DIM: int = 4                   # specific force (3) + v_batt_norm (1)
ENCODER_IN_DIM: int = ACTOR_FRAME_DIM + AUX_DIM   # 33

# Physical state channels intrinsic to vehicle dynamics (invariant to reference trajectory):
# pos (0:3), quat (3:7), omega (7:10), vel (10:13), specific_force_b (29:32), v_batt_norm (32:33)
PHYS_STATE_INDICES: Tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 29, 30, 31, 32
)
PHYS_STATE_DIM: int = len(PHYS_STATE_INDICES)  # 17

# Applied action channel (stored as prev_action in o_t[13:17])
ACTION_INDICES: Tuple[int, ...] = (13, 14, 15, 16)
ACTION_DIM: int = len(ACTION_INDICES)  # 4

PHYS_STATE_GROUPS: Tuple[Tuple[str, int], ...] = (
    ("delta_p", 3),
    ("delta_q", 4),
    ("delta_w", 3),
    ("delta_v", 3),
    ("delta_f_b", 3),
    ("delta_v_batt", 1),
)

# NOTE the encoder's input is [o_t | aux] and does NOT include the reference feed-forward
# block. That is deliberate and is the reason the block was placed BETWEEN o_t and aux in
# the env vector rather than appended: it lets the actor gain a feed-forward command while
# the frozen 33-dim encoder contract (and every pretrained encoder checkpoint) is unchanged.

# ---------------------------------------------------------------------------------
# ESTIMATOR-DRIFT TARGET GROUPS
#
# Names of the self-supervised targets that measure the estimator's OWN error, in the
# order quad_flip_env.get_priv_targets() appends them. They live here (rather than only in
# the environment) because a second consumer needs to know WHERE they are in the target
# vector: the corrupted-estimate augmentation has to move them in lockstep with the frame
# it corrupts, and it must not guess an offset.
#
# `group_slice()` resolves them by NAME against the environment's declared group list, so
# reordering PRIV_TARGET_GROUPS cannot silently point the augmentation at the wrong dims.
# ---------------------------------------------------------------------------------
EST_DRIFT_GROUPS: Tuple[str, ...] = ("est_drift_p", "est_drift_v")


def group_slice(groups, names: Sequence[str]) -> Optional[Tuple[int, int]]:
    """
    (start, stop) of `names`, in order and adjacent, inside a (name, dim) group list.

    Returns None when the groups are absent (a corpus collected before the drift targets
    existed), so every caller can degrade to "no drift handling" instead of failing.
    """
    names = tuple(names)
    if not names or not groups:
        return None
    idx: Dict[str, Tuple[int, int]] = {}
    off = 0
    for name, dim in groups:
        dim = int(dim)
        idx[str(name)] = (off, off + dim)
        off += dim
    if any(n not in idx for n in names):
        return None
    spans = [idx[n] for n in names]
    start, stop = spans[0][0], spans[-1][1]
    # Must be adjacent and in order, or the returned slice would sweep up other targets.
    if any(spans[i][1] != spans[i + 1][0] for i in range(len(spans) - 1)):
        return None
    if stop - start != sum(s[1] - s[0] for s in spans):
        return None
    return start, stop


DEFAULT_CLIP: float = 10.0
_EPS: float = 1e-8
_STD_FLOOR: float = 1e-6


def frame_from_env_obs(
    env_obs: np.ndarray,
    actor_total_dim: int = ACTOR_FRAME_DIM,
    aux_dim: int = AUX_DIM,
    ref_ff_dim: int = 0,
) -> np.ndarray:
    """
    Slice the encoder input frame out of an env observation vector.

        env obs : [ actor stack (29*H) | ref_ff (3) | aux (4) | privileged (44) ]
        frame   : [ newest actor frame (29) | aux (4) ]  -> 33 dims

    The newest actor frame is the LAST frame of the stack, so this is correct for
    any OBS_HISTORY_LEN and supports a "keep the 3-frame stack + z" ablation without
    changes.

    `ref_ff_dim` is skipped, not consumed: the feed-forward block is actor-facing and the
    encoder's contract is frozen at 33 dims. Getting this offset wrong is not subtle -
    the frame would silently become [o_t | ref_ff] and the checkpoint's f_in guard would
    fire - but the point is that the encoder never sees it.

    Accepts a single observation or a batched leading axis, e.g. (N, 80) or (80,).
    """
    env_obs = np.asarray(env_obs)
    aux_start = actor_total_dim + int(ref_ff_dim)
    if env_obs.shape[-1] < aux_start + aux_dim:
        raise ValueError(
            f"env observation has {env_obs.shape[-1]} dims, need at least "
            f"actor_total_dim({actor_total_dim}) + ref_ff_dim({int(ref_ff_dim)}) + aux_dim({aux_dim})"
        )
    actor_stack = env_obs[..., :actor_total_dim]
    aux = env_obs[..., aux_start:aux_start + aux_dim]
    newest = actor_stack[..., -ACTOR_FRAME_DIM:] if actor_total_dim > ACTOR_FRAME_DIM else actor_stack
    return np.concatenate([newest, aux], axis=-1)


@dataclass
class NormStats:
    """
    Frozen standardization constants for the encoder's input frame and output targets.

    DEGENERATE DIMENSIONS. Encoder supervision targets are not all informative: several
    of the privileged physics parameters are constant for long stretches of an episode
    (mass, thrust scale, sag coefficient), and some are constant for a whole episode.
    A channel with near-zero standard deviation would be divided by ~0 and amplified into
    pure numerical noise, so any dimension whose std falls below `_STD_FLOOR` is pinned to
    a std of 1.0 (i.e. left in its raw units) and recorded in `degenerate_dims` so the
    training script can report it rather than silently fitting noise.
    """

    frame_mean: np.ndarray
    frame_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    target_groups: List[Sequence] = field(default_factory=list)
    clip: float = DEFAULT_CLIP
    degenerate_dims: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.frame_mean = np.asarray(self.frame_mean, dtype=np.float32)
        self.frame_std = np.asarray(self.frame_std, dtype=np.float32)
        self.target_mean = np.asarray(self.target_mean, dtype=np.float32)
        self.target_std = np.asarray(self.target_std, dtype=np.float32)

    # -- frames --------------------------------------------------------------------
    def standardize_frame(self, raw: np.ndarray) -> np.ndarray:
        """(raw - mean) / std, clipped. Never clips below _STD_FLOOR."""
        z = (np.asarray(raw, dtype=np.float32) - self.frame_mean) / self.frame_std
        if self.clip > 0:
            np.clip(z, -self.clip, self.clip, out=z)
        return z.astype(np.float32)

    # -- targets -------------------------------------------------------------------
    def standardize_targets(self, raw: np.ndarray) -> np.ndarray:
        return ((np.asarray(raw, dtype=np.float32) - self.target_mean) / self.target_std).astype(np.float32)

    def destandardize_targets(self, z: np.ndarray) -> np.ndarray:
        return (np.asarray(z, dtype=np.float32) * self.target_std + self.target_mean).astype(np.float32)

    # -- construction --------------------------------------------------------------
    @classmethod
    def from_frames_and_targets(
        cls,
        frames: np.ndarray,
        targets: np.ndarray,
        target_groups: Sequence[Sequence] = (),
        clip: float = DEFAULT_CLIP,
    ) -> "NormStats":
        """
        Compute frozen constants from the pretraining set.

        `frames` is [N, ENCODER_IN_DIM], `targets` is [N, n_targets]. Both are flattened
        over every timestep of every episode.
        """
        frames = np.asarray(frames, dtype=np.float32).reshape(-1, frames.shape[-1])
        targets = np.asarray(targets, dtype=np.float32).reshape(-1, targets.shape[-1])

        f_mean = frames.mean(axis=0)
        f_std = frames.std(axis=0)
        f_std = np.where(f_std < _STD_FLOOR, 1.0, f_std)

        t_mean = targets.mean(axis=0)
        t_std = targets.std(axis=0)
        degenerate = np.flatnonzero(t_std < _STD_FLOOR).astype(int).tolist()
        t_std = np.where(t_std < _STD_FLOOR, 1.0, t_std)

        return cls(
            frame_mean=f_mean,
            frame_std=f_std,
            target_mean=t_mean,
            target_std=t_std,
            target_groups=[list(g) for g in target_groups],
            clip=clip,
            degenerate_dims=degenerate,
        )

    # -- io ------------------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "frame_mean": [float(v) for v in self.frame_mean],
            "frame_std": [float(v) for v in self.frame_std],
            "target_mean": [float(v) for v in self.target_mean],
            "target_std": [float(v) for v in self.target_std],
            "target_groups": [list(g) for g in self.target_groups],
            "clip": float(self.clip),
            "degenerate_dims": [int(i) for i in self.degenerate_dims],
            "encoder_in_dim": int(ENCODER_IN_DIM),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "NormStats":
        return cls(
            frame_mean=np.asarray(d["frame_mean"], dtype=np.float32),
            frame_std=np.asarray(d["frame_std"], dtype=np.float32),
            target_mean=np.asarray(d["target_mean"], dtype=np.float32),
            target_std=np.asarray(d["target_std"], dtype=np.float32),
            target_groups=[list(g) for g in d.get("target_groups", [])],
            clip=float(d.get("clip", DEFAULT_CLIP)),
            degenerate_dims=[int(i) for i in d.get("degenerate_dims", [])],
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "NormStats":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))
