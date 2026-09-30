"""
Export a trained MJX/Flax policy to a Stable-Baselines3 ``.zip`` that the EXISTING deploy
chain loads unchanged: ``Simulation/evaluate.py``, ``live_flight.py``, ``deploy/
export_policy.py`` all read ``mlp_extractor.policy_net.*`` / ``action_net.*`` via
``actor_input.load_checkpoint``.

FIXES OVER THE PREVIOUS EXPORTER
  1. ``_ORIGINAL_REPO`` was referenced but never defined, so ``main()`` raised NameError
     before doing anything.
  2. The critic was built with 92 inputs while the env emits 96 (o_t 29 | z 16 | ref_ff 3 |
     aux 4 | privileged 44).  The width now comes from ``spec.CRITIC_OBS_DIM`` (= 96, the
     WRAPPED observation) rather than ``spec.TOTAL_OBS_DIM`` (= 80, the raw env obs).
  3. It "verified" nothing, and ``evaluate_mjx.py`` never loaded its own checkpoint.
     ``--verify`` now round-trips: save the zip, reload it through SB3, and compare the
     deterministic action against the Flax network over random observations.
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
# _THIS must WIN over _ROOT.  The repository root holds `train_mjx.py`, a thin shim; a
# plain `if _p not in sys.path` guard is not enough, because python already has this file's
# own directory on the path, so the root lands in FRONT and `from train_mjx import
# ActorCritic` resolves to the shim and raises ImportError.
for _p in (_ROOT, _THIS):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import jax.numpy as jnp
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from quad_mjx import spec as S
from train_mjx import ActorCritic, _flatten_params


def _infer_pi_dims(flat: dict) -> tuple:
    indices = set()
    for k in flat.keys():
        for part in k.split("/"):
            if part.startswith("actor_fc"):
                try:
                    idx = int(part[len("actor_fc"):])
                    indices.add(idx)
                except ValueError:
                    pass
    if indices:
        n_layers = max(indices) + 1
        dims = []
        for i in range(n_layers):
            b_key = next((k for k in flat if f"actor_fc{i}/bias" in k or k.endswith(f"actor_fc{i}/bias")), None)
            if b_key is not None:
                dims.append(int(flat[b_key].shape[0]))
            else:
                dims.append(32)
        return tuple(dims)
    return (32, 32)


def _torch_key_map(pi_dims, vf_dims):
    """Flax flat name -> torch state-dict key, for the MlpExtractor layout."""
    m = {"action_net": "action_net", "value_net": "value_net", "log_std": "log_std"}
    for i in range(len(pi_dims)):
        m[f"actor_fc{i}"] = f"mlp_extractor.policy_net.{2 * i}"
    for i in range(len(vf_dims)):
        m[f"critic_fc{i}"] = f"mlp_extractor.value_net.{2 * i}"
    return m


def export(flat_params: dict, output_zip: str, actor_dim: int, critic_dim: int,
           pi_dims=None, vf_dims=(512, 256, 128), action_dim: int = 4) -> str:
    """Build an SB3 ``AsymmetricActorCriticPolicy`` from flat Flax parameters and save it."""
    from asymmetric_policy import AsymmetricActorCriticPolicy
    from stable_baselines3 import PPO

    if pi_dims is None:
        pi_dims = _infer_pi_dims(flat_params)

    keymap = _torch_key_map(pi_dims, vf_dims)
    policy_sd = {}
    for flat_name, arr in flat_params.items():
        parts = flat_name.split("/")
        # Flax names every path with its collection: "params/actor_fc0/kernel".  Forget
        # this and parts[0] == "params", nothing matches keymap, and the exporter quietly
        # maps ZERO weights and then reports every parameter as missing.
        if parts and parts[0] == "params":
            parts = parts[1:]
        if len(parts) < 2:
            parts = parts or [flat_name]
        layer, kind = parts[0], parts[-1]
        if layer not in keymap:
            continue
        key = keymap[layer]
        if kind == "kernel":
            t = torch.from_numpy(np.asarray(arr, np.float32)).t()   # Flax (in,out) -> torch
            policy_sd[key + ".weight"] = t
        elif kind == "bias":
            policy_sd[key + ".bias"] = torch.from_numpy(np.asarray(arr, np.float32))
        else:                                    # log_std (a bare leaf)
            policy_sd[key] = torch.from_numpy(np.asarray(arr, np.float32))

    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(int(critic_dim),), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(int(action_dim),), dtype=np.float32)

    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = obs_space
            self.action_space = act_space

        def reset(self, *, seed=None, options=None):
            return np.zeros(critic_dim, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(critic_dim, dtype=np.float32), 0.0, False, False, {}

    model = PPO(
        policy=AsymmetricActorCriticPolicy,
        env=DummyEnv(),
        policy_kwargs={
            "actor_obs_dim": int(actor_dim),
            "activation_fn": torch.nn.Tanh,
            "net_arch": {"pi": list(pi_dims), "vf": list(vf_dims)},
        },
        verbose=0,
    )
    missing, unexpected = model.policy.load_state_dict(policy_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"exporter produced keys the policy does not have: {unexpected}")
    real_missing = [m for m in missing if "features_extractor" not in m]
    if real_missing:
        raise RuntimeError(f"policy is missing weights the exporter did not supply: {real_missing}")

    os.makedirs(os.path.dirname(os.path.abspath(output_zip)), exist_ok=True)
    model.save(output_zip)
    return output_zip


def verify(output_zip: str, flat_params: dict, actor_dim: int, critic_dim: int,
           pi_dims=None, n: int = 8, seed: int = 0) -> float:
    """Reload the zip through SB3 and diff its deterministic action against Flax."""
    from stable_baselines3 import PPO
    from asymmetric_policy import AsymmetricActorCriticPolicy

    if pi_dims is None:
        pi_dims = _infer_pi_dims(flat_params)

    model = PPO.load(output_zip, custom_objects={"actor_obs_dim": int(actor_dim)})
    net = ActorCritic(pi_dims=pi_dims)
    params = _flatten_params_to_tree(flat_params, net, actor_dim, critic_dim)

    rng = np.random.default_rng(seed)
    a_obs = rng.normal(0, 1, (n, actor_dim)).astype(np.float32)
    c_obs = rng.normal(0, 1, (n, critic_dim)).astype(np.float32)
    worst = 0.0
    n_out = 0
    for i in range(n):
        ref = np.array(net.apply(params, jnp.asarray(a_obs[i:i + 1]),
                                 jnp.asarray(c_obs[i:i + 1]))[0])[0]
        got, _ = model.predict(np.concatenate([a_obs[i], c_obs[i][actor_dim:]]),
                               deterministic=True)
        # SB3's `predict` CLIPS the deterministic action to the Box bounds, so the reference
        # has to be clipped too - otherwise this compares two different quantities and
        # "fails" on any policy that saturates.  MEASURED 2026-09-24: on the 30M checkpoint
        # (actor mean reaching +-9) this reported 9.138, and 0.159 on a healthier one, while
        # `max |SB3 - clip(Flax, -1, 1)|` was 2.4e-7 in BOTH cases.  That false failure
        # aborted the pipeline at stage 4.  Channels outside the bounds are COUNTED and
        # reported instead: saturation is a statement about the policy, not the export.
        n_out += int(np.sum(np.abs(ref) > 1.0))
        worst = max(worst, float(np.max(np.abs(np.asarray(got) - np.clip(ref, -1.0, 1.0)))))
    if n_out:
        print(f"[verify] note: {n_out} of {n * int(np.size(ref))} reference channels were "
              f"outside [-1, 1]. The env clips them and `predict` clips too, so the number "
              f"above is the value the vehicle would actually receive; a saturated actor is "
              f"a TRAINING problem, not an export one.")
    return worst


def _flatten_params_to_tree(flat: dict, net, actor_dim: int, critic_dim: int):
    """Rebuild the Flax tree from the flat dict.  Shapes come from a fresh init."""
    dummy = net.init(jax_random_key(), jnp.zeros((1, actor_dim)), jnp.zeros((1, critic_dim)))
    return _unflatten(flat, dummy)


def jax_random_key():
    import jax
    return jax.random.PRNGKey(0)


def _unflatten(flat: dict, template):
    """Replace each leaf of `template` with the value from `flat` keyed by its path."""
    import jax

    def walk(tree, prefix=""):
        if isinstance(tree, dict):
            return {k: walk(v, f"{prefix}/{k}" if prefix else str(k)) for k, v in tree.items()}
        if isinstance(tree, (jnp.ndarray, np.ndarray)):
            name = prefix.strip("/")
            if name in flat:
                return jnp.asarray(flat[name])
            raise KeyError(f"no flat parameter for {name!r}")
        return tree
    return walk(template)


def main():
    ap = argparse.ArgumentParser(description="Export an MJX policy to an SB3 .zip")
    ap.add_argument("--weights", default=os.path.join(_ROOT, "logs", "quad_mjx_policy.npz"))
    ap.add_argument("--output", default=os.path.join(_ROOT, "quad_flip_model.zip"))
    ap.add_argument("--actor-dim", type=int, default=S.ACTOR_DIM_WITH_ENCODER)
    ap.add_argument("--critic-dim", type=int, default=S.CRITIC_OBS_DIM)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    if not os.path.isfile(a.weights):
        raise SystemExit(f"no weights at {a.weights}; train first (Simulation/train_mjx.py)")

    flat = {k: np.asarray(v) for k, v in dict(np.load(a.weights)).items()}
    if "/" not in next(iter(flat)) and "arr_0" in flat:
        raise SystemExit("this npz has unnamed leaves; re-save with train_mjx.run_training")

    pi_dims = _infer_pi_dims(flat)
    out = export(flat, a.output, a.actor_dim, a.critic_dim, pi_dims=pi_dims)
    print(f"[export] {out}  actor={a.actor_dim} critic={a.critic_dim} pi_dims={pi_dims}")
    if a.verify:
        err = verify(a.output, flat, a.actor_dim, a.critic_dim, pi_dims=pi_dims)
        print(f"[verify] max |SB3 - Flax| deterministic action = {err:.3e}")
        if err > 1e-5:
            raise SystemExit("VERIFY FAILED: exported policy does not match the Flax net")
        print("[verify] PASS - evaluate.py / live_flight.py can load this checkpoint")


if __name__ == "__main__":
    main()
