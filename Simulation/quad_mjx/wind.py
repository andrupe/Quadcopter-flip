"""
JAX port of ``Simulation/utils/windModel.py`` (the PERLIN branch, which is the default).

Aperiodic turbulence from 1D value noise: a baseline velocity plus a noise excursion, a
slowly wandering heading (+-45 deg) and elevation (+-10 deg).  Ported term for term,
including the hash ``fract(sin(i*127.1 + seed*311.7) * 43758.5453)``, so a given episode
seed produces the same gusts as the numpy model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

DEG2RAD = jnp.pi / 180.0


def value_noise_1d(t, seed, octaves: int = 3, persistence: float = 0.5):
    """Value noise in ~[-1, 1].  ``seed`` may be traced."""
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0
    for i in range(octaves):
        ft = t * frequency
        i0 = jnp.floor(ft)
        i1 = i0 + 1.0
        frac = ft - i0
        frac = frac * frac * (3.0 - 2.0 * frac)
        rs = seed + i * 7919
        v0 = jnp.sin(i0 * 127.1 + rs * 311.7) * 43758.5453
        v0 = v0 - jnp.floor(v0)
        v0 = v0 * 2.0 - 1.0
        v1 = jnp.sin(i1 * 127.1 + rs * 311.7) * 43758.5453
        v1 = v1 - jnp.floor(v1)
        v1 = v1 * 2.0 - 1.0
        value = value + amplitude * (v0 + frac * (v1 - v0))
        max_amplitude = max_amplitude + amplitude
        amplitude = amplitude * persistence
        frequency = frequency * 2.0
    return value / max_amplitude


@struct.dataclass
class WindState:
    seed_vel: jax.Array
    seed_head: jax.Array
    seed_elev: jax.Array
    time_scale: jax.Array
    vel_med: jax.Array
    vel_max: jax.Array
    enabled: jax.Array


def reset(key, vel_max: float, enabled: bool = True) -> WindState:
    """``Wind.reseed()``: new seeds, a new time scale and a new baseline breeze."""
    k = jax.random.split(key, 4)
    return WindState(
        seed_vel=jax.random.randint(k[0], (), 0, 100000),
        seed_head=jax.random.randint(k[1], (), 0, 100000),
        seed_elev=jax.random.randint(k[2], (), 0, 100000),
        time_scale=0.8 + jax.random.uniform(k[3], ()) * 0.6,
        # reseed() randomises the baseline to velW_max * (0.1 + 0.4 u); the constructor's
        # 0.3 * velW_max is only the pre-reseed value.
        vel_med=vel_max * (0.1 + 0.4 * jax.random.uniform(k[3], ())),
        vel_max=jnp.asarray(vel_max),
        enabled=jnp.asarray(enabled),
    )


def random_wind(st: WindState, t):
    """Returns (velW, qW1, qW2) for time t, matching ``Wind.randomWind``."""
    ts = t * st.time_scale
    noise_vel = value_noise_1d(ts, st.seed_vel, 3)
    vel = st.vel_med + noise_vel * st.vel_max * 0.5
    vel = jnp.clip(vel, 0.0, st.vel_max)
    noise_head = value_noise_1d(ts * 0.3, st.seed_head, 2)
    q1 = noise_head * 45.0 * DEG2RAD
    noise_elev = value_noise_1d(ts * 0.25, st.seed_elev, 2)
    q2 = noise_elev * 10.0 * DEG2RAD
    zero = jnp.zeros(())
    vel = jnp.where(st.enabled, vel, zero)
    q1 = jnp.where(st.enabled, q1, zero)
    q2 = jnp.where(st.enabled, q2, zero)
    return vel, q1, q2


def wind_world(st: WindState, t):
    vel, q1, q2 = random_wind(st, t)
    return jnp.array([vel * jnp.cos(q1) * jnp.cos(q2),
                      vel * jnp.sin(q1) * jnp.cos(q2),
                      vel * jnp.sin(q2)])


def empty() -> WindState:
    z = jnp.zeros(())
    return WindState(seed_vel=jnp.array(0), seed_head=jnp.array(0), seed_elev=jnp.array(0),
                     time_scale=jnp.array(1.0), vel_med=z, vel_max=z,
                     enabled=jnp.asarray(False))
