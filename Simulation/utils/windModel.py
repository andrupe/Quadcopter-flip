# -*- coding: utf-8 -*-
"""
Wind disturbance models for quadcopter simulation.
Supports: NONE, FIXED, SINE, RANDOMSINE, PERLIN (aperiodic turbulence).

Original sine model by John Bass (john.bobzwik@gmail.com), MIT license.
Perlin noise model added for sim-to-real domain randomization.
"""

import numpy as np
from numpy import sin, cos, pi
import random as rd
import config

deg2rad = pi / 180.0


def _value_noise_1d(t: float, seed: int = 0, octaves: int = 3, persistence: float = 0.5) -> float:
    """
    Simple 1D value noise (Perlin-like) using linear interpolation of random lattice points.
    Returns a value in approximately [-1, 1] for any input t.
    
    - octaves: number of frequency layers (more = finer detail)
    - persistence: amplitude decay per octave (0.5 = each octave is half as strong)
    """
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for i in range(octaves):
        # Hash-based pseudo-random lattice values (deterministic per seed+octave)
        ft = t * frequency
        i0 = int(np.floor(ft))
        i1 = i0 + 1
        frac = ft - i0

        # Smooth interpolation (cubic Hermite / smoothstep)
        frac = frac * frac * (3.0 - 2.0 * frac)

        # Deterministic pseudo-random values at lattice points
        rng_seed = seed + i * 7919  # Different seed per octave
        v0 = np.sin(i0 * 127.1 + rng_seed * 311.7) * 43758.5453
        v0 = v0 - np.floor(v0)  # fract() -> [0, 1]
        v0 = v0 * 2.0 - 1.0     # remap to [-1, 1]

        v1 = np.sin(i1 * 127.1 + rng_seed * 311.7) * 43758.5453
        v1 = v1 - np.floor(v1)
        v1 = v1 * 2.0 - 1.0

        value += amplitude * (v0 + frac * (v1 - v0))
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= 2.0

    return value / max_amplitude  # Normalize to [-1, 1]


class Wind:

    def __init__(self, *args, rng=None):

        # Seeded generator, or None to fall back to the module-level `random`. Must be set
        # before anything below draws from it.
        self._rng = rng

        if (len(args) == 0):
            self.windType = 'NONE'
        elif not isinstance(args[0], str):
            raise Exception('Not a valid wind type.')
        else:
            self.windType = args[0].upper()

        if (self.windType == 'SINE') or (self.windType == 'RANDOMSINE'):

            if (self.windType == 'SINE'):

                self.velW_med = args[1]
                self.qW1_med  = args[2]*deg2rad
                self.qW2_med  = args[3]*deg2rad

            elif (self.windType == 'RANDOMSINE'):

                # Wind Velocity
                velW_max = float(args[1])   # m/s
                velW_min = float(args[2])   # m/s
                self.velW_max = velW_max
                # Wind Heading
                qW1_max  = args[3]   # deg
                qW1_min  = args[4]   # deg
                # Wind Elevation (positive = upwards wind in NED, positive = downwards wind in ENU)
                qW2_max  = args[5]   # deg
                qW2_min  = args[6]   # deg

                # Median values
                self.velW_med = (velW_max - velW_min)*self._u01() + velW_min
                self.qW1_med  = ((qW1_max - qW1_min)*self._u01() + qW1_min)*deg2rad
                self.qW2_med  = ((qW2_max - qW2_min)*self._u01() + qW2_min)*deg2rad

            else:
                self.velW_max = float(args[1]) if len(args) > 1 else 2.0

            # Wind Velocity - scaled proportionally to velW_max so gusts stay within limits
            scale = getattr(self, 'velW_max', 2.0) / 4.0
            self.velW_a1 = 0.5 * scale  # Wind velocity amplitude 1
            self.velW_f1 = 0.7          # Wind velocity frequency 1
            self.velW_d1 = 0            # Wind velocity delay (offset) 1
            self.velW_a2 = 0.3 * scale  # Wind velocity amplitude 2
            self.velW_f2 = 1.2          # Wind velocity frequency 2
            self.velW_d2 = 1.3          # Wind velocity delay (offset) 2
            self.velW_a3 = 0.2 * scale  # Wind velocity amplitude 3
            self.velW_f3 = 2.3          # Wind velocity frequency 3
            self.velW_d3 = 2.0          # Wind velocity delay (offset) 3

            # Wind Heading
            self.qW1_a1  = 15.0*deg2rad # Wind heading amplitude 1
            self.qW1_f1  = 0.1          # Wind heading frequency 1
            self.qW1_d1  = 0            # Wind heading delay (offset) 1
            self.qW1_a2  = 3.0*deg2rad  # Wind heading amplitude 2
            self.qW1_f2  = 0.54         # Wind heading frequency 2
            self.qW1_d2  = 0            # Wind heading delay (offset) 2

            # Wind Elevation
            self.qW2_a1  = 4.0*deg2rad  # Wind elevation amplitude 1
            self.qW2_f1  = 0.1          # Wind elevation frequency 1
            self.qW2_d1  = 0            # Wind elevation delay (offset) 1
            self.qW2_a2  = 0.8*deg2rad  # Wind elevation amplitude 2
            self.qW2_f2  = 0.54         # Wind elevation frequency 2
            self.qW2_d2  = 0            # Wind elevation delay (offset) 2

        elif (self.windType == 'PERLIN'):
            # Perlin-like aperiodic wind using value noise
            self.velW_max = float(args[1]) if len(args) > 1 else 1.0
            self.velW_med = self.velW_max * 0.3  # Light baseline breeze
            # Random seeds per episode (re-randomized in reseed())
            self._seed_vel = self._randint(0, 100000)
            self._seed_heading = self._randint(0, 100000)
            self._seed_elev = self._randint(0, 100000)
            # Noise time scale: lower = slower, more gradual gusts
            self._time_scale = 0.8 + self._u01() * 0.6  # 0.8–1.4

        elif (self.windType == 'FIXED'):

            self.velW_med = args[1]
            self.qW1_med  = args[2]*deg2rad
            self.qW2_med  = args[3]*deg2rad

        elif (self.windType == 'NONE'):

            self.velW_med = 0
            self.qW1_med  = 0
            self.qW2_med  = 0

        else:

            raise Exception('Not a valid wind type.')

    def set_rng(self, rng) -> None:
        """
        Route this model's randomness through a seeded generator.

        The module-level `random` (rd) is a GLOBAL generator shared by every Wind object
        in the process, so what wind an episode sees depends on how many other Wind
        objects were constructed before it. That silently defeats `env.reset(seed=)` and
        makes training runs irreproducible - the same seed produces a different episode
        on every run. Passing the environment's own seeded generator removes that coupling.
        """
        self._rng = rng

    def _u01(self) -> float:
        """Uniform [0, 1) from the seeded generator, falling back to the global one."""
        return float(self._rng.random()) if self._rng is not None else rd.random()

    def _randint(self, lo: int, hi: int) -> int:
        if self._rng is not None:
            return int(self._rng.integers(lo, hi))
        return int(rd.randint(lo, hi))

    def reseed(self):
        """Re-randomize wind pattern for a new episode. Call from env.reset()."""
        if self.windType == 'PERLIN':
            self._seed_vel = self._randint(0, 100000)
            self._seed_heading = self._randint(0, 100000)
            self._seed_elev = self._randint(0, 100000)
            self._time_scale = 0.8 + self._u01() * 0.6
            self.velW_med = self.velW_max * (0.1 + 0.4 * self._u01())  # Randomize baseline
        elif self.windType == 'RANDOMSINE':
            # Re-randomize sine medians for new episode
            self.velW_med = self.velW_max * self._u01()
            self.qW1_med = (360.0 * self._u01() - 180.0) * deg2rad
            self.qW2_med = (30.0 * self._u01() - 15.0) * deg2rad

    def randomWind(self, t):
        if (self.windType == 'SINE') or (self.windType == 'RANDOMSINE'):

            velW = self.velW_a1*sin(self.velW_f1*t - self.velW_d1) + self.velW_a2*sin(self.velW_f2*t - self.velW_d2) + self.velW_a3*sin(self.velW_f3*t - self.velW_d3) + self.velW_med
            qW1  = self.qW1_a1*sin(self.qW1_f1*t - self.qW1_d1) + self.qW1_a2*sin(self.qW1_f2*t - self.qW1_d2) + self.qW1_med
            qW2  = self.qW2_a1*sin(self.qW2_f1*t - self.qW2_d1) + self.qW2_a2*sin(self.qW2_f2*t - self.qW2_d2) + self.qW2_med

            # Ceiling: the live velW_max when the caller maintains one (the env shrinks it
            # with the ADR level), otherwise the model's own achievable maximum. The old
            # getattr default of a bare 2.0 silently clipped a legitimate SINE wind to
            # 2 m/s whenever the attribute happened to be absent.
            ceiling = float(getattr(
                self, "velW_max",
                max(self.velW_med, self.velW_med + self.velW_a1 + self.velW_a2 + self.velW_a3),
            ))
            velW = max(0.0, min(ceiling, float(velW)))

        elif self.windType == 'PERLIN':
            ts = t * self._time_scale
            # Aperiodic velocity: baseline + noise scaled to max
            noise_vel = _value_noise_1d(ts, seed=self._seed_vel, octaves=3)
            velW = self.velW_med + noise_vel * self.velW_max * 0.5
            velW = max(0.0, min(self.velW_max, float(velW)))

            # Aperiodic heading: slow wander ±45°
            noise_head = _value_noise_1d(ts * 0.3, seed=self._seed_heading, octaves=2)
            qW1 = noise_head * 45.0 * deg2rad

            # Aperiodic elevation: slow wander ±10°
            noise_elev = _value_noise_1d(ts * 0.25, seed=self._seed_elev, octaves=2)
            qW2 = noise_elev * 10.0 * deg2rad

        elif (self.windType == 'FIXED') or (self.windType == 'NONE'):
            velW = self.velW_med
            qW1  = self.qW1_med
            qW2  = self.qW2_med

        return velW, qW1, qW2