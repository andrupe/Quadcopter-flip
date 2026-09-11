"""
Crazyflie Lighthouse deck model: base-station geometry, occlusion, dropout and the
resulting dead-reckoning drift.

WHY THIS EXISTS
The Lighthouse deck is the only absolute position/velocity source on the airframe. It is
NOT a continuous GPS-like feed: the deck is a set of upward-facing photodiodes that only
produce a fix while enough base stations are actually illuminating them. During a flip the
deck is inverted, every station is behind the deck, and the fix disappears entirely for
the whole rotation. The flight controller must then dead-reckon on the IMU until the deck
comes back.

That matters for this project specifically. If the policy is trained against ground-truth
position and velocity, it learns a control law that silently assumes a sensor which the
real vehicle does not have during the exact manoeuvre we care about. Training against a
dead-reckoned estimate instead forces the learned controller (and the history encoder,
which is what will have to notice the vehicle is off-nominal) to cope with a state estimate
that is degrading while it acts.

WHAT IS MODELLED
  1. Base-station geometry - N stations on a ring around the flight volume, at an
     elevation above it, with a randomised count, azimuth and radius per episode.
  2. Occlusion - a station is only visible when the deck normal (body +z, the direction
     the photodiodes face) is within a half-angle cone of the direction to that station,
     and the station is within range.
  3. A minimum station count per fix. A single station gives one sweep plane, which is not
     a pose; the real deck needs at least two. So fixes need >= min_stations_for_fix
     simultaneous stations, which is what makes a flip a genuine blackout rather than a
     graceful degradation.
  4. Dropout - per-station random blackouts, for interference and occlusion of individual
     stations.
  5. Dead reckoning - while no fix is available the estimate is propagated with the last
     velocity, and a slowly accumulating velocity bias makes it diverge from truth. It
     does NOT decay back to truth on its own, which is the property that makes the problem
     hard rather than cosmetic.

The station count is randomised over 2-4 because the number actually installed is not
known. Randomising it is also the honest engineering answer: a policy that only works
against one specific station layout is not deployable, and the coverage differences between
2 and 4 stations are exactly the kind of thing that should be in the training distribution.

Standalone and dependency-free apart from numpy so it can be tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

GRAVITY: float = 9.81


@dataclass
class LighthouseConfig:
    """Randomisation envelope for one episode's Lighthouse installation."""

    # Number of base stations actually present. 2 is the minimum for a fix; 4 is the
    # upper bound of what is typically installed in a room.
    station_counts: Tuple[int, ...] = (2, 3, 4)
    # Stations are placed RELATIVE TO THE FLIGHT VOLUME, at an angle from the local
    # vertical drawn from [0, cone_frac * half_angle] and a distance in dist_range. This
    # guarantees an upright deck has coverage by construction, so every dropout in a
    # training episode is caused by the vehicle's own attitude rather than by a
    # mis-specified installation.
    cone_frac_range: Tuple[float, float] = (0.30, 0.80)
    dist_range: Tuple[float, float] = (1.5, 4.0)
    # Azimuth jitter, in radians, about the evenly spaced nominal bearings.
    azimuth_jitter: float = 0.25
    # Half-angle of the deck's acceptance cone about body +z (degrees).
    half_angle_deg_range: Tuple[float, float] = (55.0, 80.0)
    # Stations beyond this range are not usable.
    max_range: float = 6.0
    # Minimum simultaneous stations for a pose fix (1 sweep plane is not a pose).
    min_stations_for_fix: int = 2
    # Measurement noise, linearly interpolated on the domain-randomisation level.
    pos_noise_range: Tuple[float, float] = (0.003, 0.010)     # m
    vel_noise_range: Tuple[float, float] = (0.020, 0.060)     # m/s
    # Lighthouse Z is noticeably weaker than XY in practice (short baseline to the
    # stations, and it is usually the axis the barometer is fused into).
    z_noise_scale: float = 2.0
    # Per-station probability of a dropout on any given step (interference).
    dropout_prob_range: Tuple[float, float] = (0.0, 0.03)
    # Dead-reckoning error model. During a blackout the estimate is propagated open-loop,
    # and the dominant error on a quadrotor is NOT accelerometer bias - it is a small
    # tilt error theta projecting gravity into a spurious horizontal acceleration of
    # about g*theta, which then integrates into velocity and position. The two rates below
    # are random-walk rates for that tilt error (rad/sqrt(s)) and for the accelerometer
    # bias (m/s^2/sqrt(s)).
    tilt_walk_range: Tuple[float, float] = (0.010, 0.040)
    accel_walk_range: Tuple[float, float] = (0.02, 0.10)
    # A fix is only produced every `fix_decimation` steps, modelling the base-station
    # sweep period rather than a continuous output.
    fix_decimation: int = 1


class LighthouseModel:
    """
    Per-episode Lighthouse installation plus the estimator state it feeds.

    Usage per step:
        out = lh.observe(p_true, v_true, R_world, dt)
        # out["p_est"], out["v_est"] are what the flight controller actually has.

    The model OWNS the estimate. Ground truth is only ever used to generate a fix, so it
    is structurally impossible to accidentally leak truth into the actor observation at a
    step where no fix was available.
    """

    def __init__(self, config: Optional[LighthouseConfig] = None):
        self.cfg = config or LighthouseConfig()
        self.p_est = np.zeros(3, dtype=np.float64)
        self.v_est = np.zeros(3, dtype=np.float64)
        self.stations = np.zeros((0, 3), dtype=np.float64)
        self.half_angle = 0.0
        self.outage_t = 0.0
        self.fix_available = True
        self.n_visible = 0
        self.n_stations = 0
        self._tilt_err = np.zeros(2, dtype=np.float64)
        self._accel_bias = np.zeros(3, dtype=np.float64)
        self._step = 0
        self._pos_sigma = 0.0
        self._vel_sigma = 0.0
        self._dropout_prob = 0.0
        self._tilt_walk = 0.0
        self._accel_walk = 0.0

    # -- episode setup --------------------------------------------------------------
    def reset(self, p0: np.ndarray, v0: np.ndarray, rng: np.random.Generator, dr: float = 0.0) -> None:
        """
        Draw a fresh installation and initialise the estimate at truth.

        `dr` is the domain-randomisation level in [0, 1]; the broad randomisation (station
        count, geometry) is always active because it is an installation property, while
        the noise magnitudes scale with `dr`.
        """
        cfg = self.cfg
        self.n_stations = int(rng.choice(cfg.station_counts))
        self.half_angle = np.radians(float(rng.uniform(*cfg.half_angle_deg_range)))

        # Place stations around the nominal flight point, INSIDE the acceptance cone.
        #
        # Stating the geometry this way (rather than as a ring of absolute height and
        # radius) is what makes the model testable: an upright deck is guaranteed to see
        # every station, because the placement angle is drawn as a fraction of the very
        # cone the visibility test uses. An absolute-height ring does not guarantee this
        # - r*sin(elevation) can easily land the stations BELOW the flight volume, which
        # is a silent way to make the sensor permanently blind and the model vacuous.
        base = float(rng.uniform(0.0, 2.0 * np.pi))
        azimuths = base + np.linspace(0.0, 2.0 * np.pi, self.n_stations, endpoint=False)
        azimuths += rng.uniform(-cfg.azimuth_jitter, cfg.azimuth_jitter, size=self.n_stations)

        alpha = rng.uniform(0.0, cfg.cone_frac_range[1], size=self.n_stations) * self.half_angle
        alpha = np.maximum(alpha, cfg.cone_frac_range[0] * self.half_angle)
        dist = rng.uniform(*cfg.dist_range, size=self.n_stations)
        anchor = np.asarray(p0, dtype=np.float64)
        self.stations = anchor[None, :] + np.column_stack([
            dist * np.sin(alpha) * np.cos(azimuths),
            dist * np.sin(alpha) * np.sin(azimuths),
            dist * np.cos(alpha),
        ])

        self._pos_sigma = _lerp(cfg.pos_noise_range, dr)
        self._vel_sigma = _lerp(cfg.vel_noise_range, dr)
        self._dropout_prob = _lerp(cfg.dropout_prob_range, dr)
        self._tilt_walk = _lerp(cfg.tilt_walk_range, dr)
        self._accel_walk = _lerp(cfg.accel_walk_range, dr)

        self.p_est = np.asarray(p0, dtype=np.float64).copy()
        self.v_est = np.asarray(v0, dtype=np.float64).copy()
        self._tilt_err = np.zeros(2, dtype=np.float64)
        self._accel_bias = np.zeros(3, dtype=np.float64)
        self.outage_t = 0.0
        self.fix_available = True
        self.n_visible = self.n_stations
        self._step = 0

    # -- geometry -------------------------------------------------------------------
    def visible_mask(self, p: np.ndarray, R_world: np.ndarray) -> np.ndarray:
        """
        Which stations can currently illuminate the deck.

        A station is visible when the deck's acceptance cone (about body +z: the
        photodiodes face up out of the airframe) contains the direction to that station,
        and the station is in range. When the vehicle is inverted, body +z points at the
        floor, the stations are all above, and every station drops out at once.
        """
        if self.n_stations == 0:
            return np.zeros(0, dtype=bool)
        to_station = self.stations - p[None, :]
        dist = np.linalg.norm(to_station, axis=1)
        ok_range = dist <= self.cfg.max_range
        safe = np.where(dist > 1e-9, dist, 1.0)
        unit = to_station / safe[:, None]

        deck_normal = np.asarray(R_world, dtype=np.float64)[:, 2]
        cos_ang = unit @ deck_normal
        return ok_range & (cos_ang >= np.cos(self.half_angle))

    # -- per-step sensor update -----------------------------------------------------
    def observe(
        self,
        p_true: np.ndarray,
        v_true: np.ndarray,
        R_world: np.ndarray,
        dt: float,
        rng: np.random.Generator,
    ) -> dict:
        """
        Advance the sensor and the estimator by one step and return the available state.

        Returns a dict with:
            p_est, v_est   what the flight controller has (use these, never truth)
            has_fix        a genuine fix was produced this step
            n_visible      stations geometrically visible before dropout
            outage_t       seconds since the last fix (0.0 while fixed)
            drifted        m, distance between the estimate and truth
        """
        self._step += 1
        vis = self.visible_mask(p_true, R_world)
        n_vis = int(np.count_nonzero(vis))

        # Per-station dropout on top of geometry.
        if self._dropout_prob > 0.0 and n_vis > 0:
            keep = rng.random(n_vis) >= self._dropout_prob
            n_vis = int(np.count_nonzero(keep))

        has_fix = (
            n_vis >= self.cfg.min_stations_for_fix
            and (self._step % max(1, self.cfg.fix_decimation) == 0)
        )

        if has_fix:
            sigma = np.array([self._pos_sigma, self._pos_sigma,
                              self._pos_sigma * self.cfg.z_noise_scale])
            self.p_est = np.asarray(p_true, dtype=np.float64) + rng.normal(0.0, sigma)
            self.v_est = np.asarray(v_true, dtype=np.float64) + rng.normal(0.0, self._vel_sigma, size=3)
            self.outage_t = 0.0
            # A fix re-anchors attitude and velocity, so the accumulated error states are
            # mostly cancelled; retain a little, as a real EKF would not fully trust one
            # update.
            self._tilt_err *= 0.25
            self._accel_bias *= 0.25
        else:
            # DEAD RECKONING. Propagate open-loop. Deliberately no attraction back to
            # truth: a decaying error would make long blackouts harmless and hide exactly
            # the failure mode this model exists to expose.
            self.outage_t += dt
            if self._tilt_walk > 0.0:
                self._tilt_err += rng.normal(0.0, self._tilt_walk * np.sqrt(dt), size=2)
            if self._accel_walk > 0.0:
                self._accel_bias += rng.normal(0.0, self._accel_walk * np.sqrt(dt), size=3)

            # A tilt error theta mis-projects gravity, giving ~g*theta of spurious
            # horizontal acceleration. This integrates TWICE, so position error grows
            # roughly quadratically in the outage duration - which is why even a short
            # blackout is worth modelling, and why the estimate must not be silently
            # replaced by truth whenever coverage is poor.
            a_drift = np.array([
                GRAVITY * self._tilt_err[0],
                GRAVITY * self._tilt_err[1],
                self._accel_bias[2],
            ])
            self.v_est = self.v_est + a_drift * dt
            self.p_est = self.p_est + self.v_est * dt

        self.fix_available = has_fix
        self.n_visible = n_vis
        return {
            "p_est": self.p_est.copy(),
            "v_est": self.v_est.copy(),
            "has_fix": bool(has_fix),
            "n_visible": n_vis,
            "outage_t": float(self.outage_t),
            "drifted": float(np.linalg.norm(self.p_est - np.asarray(p_true, dtype=np.float64))),
        }


def _lerp(rng_range: Tuple[float, float], dr: float) -> float:
    lo, hi = rng_range
    return float(lo + float(np.clip(dr, 0.0, 1.0)) * (hi - lo))
