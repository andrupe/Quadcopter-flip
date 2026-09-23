# -*- coding: utf-8 -*-
"""
Human-generated references for the trained policy, and live handover helpers.

WHERE THIS SITS IN THE STACK
----------------------------
Training samples the reference from `trajectories.py` (differential-flatness
manoeuvres). Evaluation replays the same sampler. This module replaces the SAMPLER
with a live source, without touching the environment: `QuadFlipEnv` reads one
attribute, `self.traj`, and calls `self.traj.sample(self.t)` once per step, so anything
that duck-types that one call *is* a reference generator as far as the trained policy
is concerned. Two sources are provided:

    HumanTarget        the pilot's sticks. The commands define where the reference goes
                       (velocity + heading + climb), the same way a pilot defines the
                       trajectory on a real aircraft. Centre the sticks and the
                       reference converges to a point and stays there, so "hover" is a
                       command, not a mode.
    ShiftedTrajectory  a sampled manoeuvre from `trajectories.py` (flip, orbit, ...)
                       RELOCATED to wherever the vehicle currently is and rotated to
                       its current heading. This is what lets the policy fly a training
                       manoeuvre during a live flight: hand the pilot a flip that starts
                       from their own hover point instead of the sampler's spawn.

Both produce the same `Reference` record the trajectory layer produces (p, v, a, R,
omega, thrust_ff, spin, kind), so every consumer downstream - the actor frame's error
terms, the reward, the telemetry, the HUD - is the unmodified training code path.

CONSISTENCY RULES THAT MATTER
-----------------------------
* The reference must stay DYNAMICALLY CONSISTENT with the flatness relation the policy
  learned: attitude is `dcm_from_thrust_dir_and_yaw(a + g e_z, yaw)` and the collective
  is `m |a + g e_z|`. A reference whose attitude does not match its own acceleration
  would ask the policy to track an impossible pair (this is why `HumanTarget` filters
  the command through an acceleration-limited second-order law instead of jumping the
  reference to the stick).
* Body rate is a central difference of R over t +- dt on BOTH sides - the same stencil
  `Trajectory.sample` uses (there with a 1 ms half-width, here with the 10 ms control
  period). One-sided differences are first-order and were already the cause of a
  reference-rate defect in the waypoint family.
* `sync_to` / `sync_to_reference` exist so that every handover starts the new reference
  FROM THE VEHICLE'S OWN STATE. Engaging a hover reference at the origin while the
  vehicle is 2 m away would hand the policy a step it did not create and spend the
  first second of the handover paying for it.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from trajectories import (
    Maneuver,
    Reference,
    Trajectory,
    dcm_from_thrust_dir_and_yaw,
    omega_from_dcm,
)

GRAVITY = 9.81

# -- HumanTarget limits -----------------------------------------------------------------
# These are POLICY-side reference limits, not vehicle limits. They are chosen so the
# reference the pilot generates stays inside the envelope the policy was trained on: the
# training mixture moves references at up to ~3.7 m/s (orbit/lissajous peaks) with
# accelerations well under 9 m/s^2, and no reference in the mixture asks for a body rate
# near the 20 rad/s action clip. Keeping the live reference inside that box means the
# policy is always flying something it has seen, however hard the pilot leans on the
# sticks.
HUMAN_V_MAX: float = 3.0          # m/s cap on the reference speed (all axes)
HUMAN_ACC_MAX: float = 4.5        # m/s^2 cap on the reference acceleration (sets the tilt)
HUMAN_JERK_MAX: float = 30.0      # m/s^3 cap on how fast that acceleration may change
HUMAN_KP: float = 4.0             # 1/s: velocity error -> acceleration target
HUMAN_PULL_GAIN: float = 1.0      # 1/s: how fast the aim point closes on the vehicle
HUMAN_PULL_MIN_CMD: float = 0.05  # m/s of command above which the aim point trails
HUMAN_YAW_RATE_MAX: float = 1.5   # rad/s cap on the reference yaw rate
HUMAN_YAW_ACC_MAX: float = 6.0    # rad/s^2 cap on the reference yaw acceleration
HUMAN_DEADZONE: float = 0.08      # logical-stick deadzone, so a resting stick means "hold"
# A reference that never ends: QuadFlipEnv truncates when `t >= traj.duration`, and under
# live control the pilot decides when the flight is over, not the sampler.
HUMAN_FOREVER: float = 1.0e9


def _clip_norm(v: np.ndarray, limit: float) -> np.ndarray:
    """Scale a vector down to `limit` when longer than it (direction preserving)."""
    n = float(np.linalg.norm(v))
    if n > limit > 0.0:
        return v * (limit / n)
    return v


def yaw_of(R: np.ndarray) -> float:
    """Heading of a rotation matrix in this project's convention (x = nose)."""
    return float(math.atan2(R[1, 0], R[0, 0]))


class _ManeuverShim:
    """Everything a Reference producer needs to look like a Trajectory to the env.

    `QuadFlipEnv` reads exactly two attributes off `self.traj`: `.duration` (it truncates
    the episode when the reference ends) and `.maneuver.kind` (telemetry and the reward's
    tolerance table). Neither has to be a real `Maneuver`.
    """

    kind: str = "human"

    def __init__(self, kind: str = "human") -> None:
        self.kind = kind


class HumanTarget:
    """
    The pilot as a trajectory generator.

    Commands are expressed the way a pilot thinks about them:

        set_command(fwd, left, up, yaw_rate)   in the CURRENT reference heading frame

    and are turned into a smooth, dynamically consistent reference by a critically damped
    acceleration-limited filter. Centre everything and the reference converges to a fixed
    point - a hover - which the policy then holds.

    `sync_to` must be called on every handover (see the module docstring).
    """

    def __init__(self, p0: np.ndarray, yaw0: float = 0.0, v0: Optional[np.ndarray] = None) -> None:
        self.p_ref = np.asarray(p0, dtype=np.float64).copy()
        self.v_ref = np.zeros(3, dtype=np.float64) if v0 is None else np.asarray(v0, dtype=np.float64).copy()
        self.a_ref = np.zeros(3, dtype=np.float64)
        self.yaw_ref = float(yaw0)
        self.yaw_rate_ref = 0.0
        self._v_cmd = np.zeros(3, dtype=np.float64)      # world frame
        self._yaw_rate_cmd = 0.0
        self._t = 0.0
        self.duration = HUMAN_FOREVER
        self.maneuver = _ManeuverShim("human")
        self._R_prev = self._R_now = Maneuver.flat_attitude(self.a_ref, self.yaw_ref)
        self._R_next = self._R_now
        self._omega = np.zeros(3, dtype=np.float64)

    # -- commands ---------------------------------------------------------------------
    def set_command(self, fwd: float, left: float, up: float, yaw_rate: float = 0.0) -> None:
        """Velocity command in the current heading frame (m/s) plus a yaw-rate command."""
        cy, sy = math.cos(self.yaw_ref), math.sin(self.yaw_ref)
        # x = nose (fwd), y = left. The z axis is world up.
        vx = cy * fwd - sy * left
        vy = sy * fwd + cy * left
        self._v_cmd[:] = (vx, vy, float(up))
        v_norm = float(np.linalg.norm(self._v_cmd))
        if v_norm > HUMAN_V_MAX:
            self._v_cmd *= HUMAN_V_MAX / v_norm
        self._yaw_rate_cmd = float(np.clip(yaw_rate, -HUMAN_YAW_RATE_MAX, HUMAN_YAW_RATE_MAX))

    def clear_command(self) -> None:
        """Everything to zero: the reference converges to a hover where it is."""
        self._v_cmd[:] = 0.0
        self._yaw_rate_cmd = 0.0

    @property
    def command_world(self) -> np.ndarray:
        return self._v_cmd.copy()

    @property
    def yaw_rate_command(self) -> float:
        return self._yaw_rate_cmd

    # -- handover ----------------------------------------------------------------------
    def sync_to(self, p: np.ndarray, v: Optional[np.ndarray] = None, yaw: Optional[float] = None) -> None:
        """Re-anchor the reference on the vehicle's own state (no step in the errors)."""
        self.p_ref = np.asarray(p, dtype=np.float64).copy()
        self.v_ref = np.zeros(3, dtype=np.float64) if v is None else np.asarray(v, dtype=np.float64).copy()
        self.a_ref = np.zeros(3, dtype=np.float64)
        if yaw is not None:
            self.yaw_ref = float(yaw)
        self.yaw_rate_ref = 0.0
        self.clear_command()
        self._R_prev = self._R_now = Maneuver.flat_attitude(self.a_ref, self.yaw_ref)
        self._R_next = self._R_now
        self._omega = np.zeros(3, dtype=np.float64)

    def sync_to_reference(self, ref: Reference, settle: bool = True) -> None:
        """Re-anchor on another reference's final state (used when a manoeuvre ends)."""
        self.p_ref = np.asarray(ref.p, dtype=np.float64).copy()
        self.v_ref = np.zeros(3, dtype=np.float64) if settle else np.asarray(ref.v, dtype=np.float64).copy()
        self.a_ref = np.zeros(3, dtype=np.float64) if settle else np.asarray(ref.a, dtype=np.float64).copy()
        self.yaw_ref = yaw_of(ref.R)
        self.yaw_rate_ref = 0.0
        self.clear_command()
        R0 = Maneuver.flat_attitude(self.a_ref, self.yaw_ref)
        self._R_prev = self._R_now = R0
        self._R_next = R0
        self._omega = np.zeros(3, dtype=np.float64)

    # -- integration --------------------------------------------------------------------
    def update(self, dt: float, vehicle_p: Optional[np.ndarray] = None) -> None:
        """Advance the reference by one control period. Call once per sim step.

        `vehicle_p` (the position ESTIMATE, so the reference sees what the policy sees)
        enables the aim-point pull. While the pilot commands motion, the reference point
        trails the vehicle instead of running away from it: without the pull a constant
        velocity command leaves a persistent position error that the trained policy has
        no reason to have learned to null - its training references are dynamically
        consistent trajectories, not a point on a leash - and the vehicle ends up flying
        1.5 m behind its own target. With the pull the lag is bounded and decays, and a
        centred stick (no command) disables it entirely, so "hover" is an anchored point
        the policy holds hard rather than a point that drifts with the vehicle.
        """
        dt = float(dt)
        self._t += dt

        # Heading: first-order with an acceleration cap, so the reference yaw never steps.
        rate_err = self._yaw_rate_cmd - self.yaw_rate_ref
        self.yaw_rate_ref += float(np.clip(rate_err, -HUMAN_YAW_ACC_MAX * dt,
                                           HUMAN_YAW_ACC_MAX * dt))
        self.yaw_ref += self.yaw_rate_ref * dt

        # Velocity: a capped P law whose acceleration is SLEW LIMITED, not a PD.
        #
        # The damping term is deliberately absent. A PD on the velocity error feeds the
        # previous acceleration back into the new one, and under an acceleration CLIP that
        # combination limit-cycles: the first step demands a (clipped to ACC_MAX), the
        # second subtracts KD*ACC_MAX from the proportional term and drives the reference
        # back to where it started, forever. Measured, before this was rewritten. A capped
        # proportional law on a double integrator converges to the commanded velocity with
        # no overshoot by itself (v' = K(v_cmd - v)), so the only thing missing is a bound
        # on how fast the attitude may change - which is what the slew limit is for. The
        # acceleration cap is also what bounds the reference TILT, so it has to be a true
        # bound on `a_ref`, not on some internal demand.
        a_target = _clip_norm(HUMAN_KP * (self._v_cmd - self.v_ref), HUMAN_ACC_MAX)
        slew = a_target - self.a_ref
        step_lim = HUMAN_JERK_MAX * dt
        if float(np.linalg.norm(slew)) > step_lim > 0.0:
            slew = slew * (step_lim / float(np.linalg.norm(slew)))
        self.a_ref = _clip_norm(self.a_ref + slew, HUMAN_ACC_MAX)
        self.v_ref = _clip_norm(self.v_ref + self.a_ref * dt, HUMAN_V_MAX)
        self.p_ref = self.p_ref + self.v_ref * dt
        if vehicle_p is not None and float(np.linalg.norm(self._v_cmd)) > HUMAN_PULL_MIN_CMD:
            self.p_ref = self.p_ref + HUMAN_PULL_GAIN * (np.asarray(vehicle_p, dtype=np.float64)
                                                         - self.p_ref) * dt

        # Attitude from flatness, and the symmetric rate stencil the trajectory layer uses.
        self._R_prev = self._R_now
        self._R_now = Maneuver.flat_attitude(self.a_ref, self.yaw_ref)
        a_next = _clip_norm(HUMAN_KP * (self._v_cmd - (self.v_ref + self.a_ref * dt)), HUMAN_ACC_MAX)
        self._R_next = Maneuver.flat_attitude(a_next, self.yaw_ref + self.yaw_rate_ref * dt)
        self._omega = omega_from_dcm(self._R_prev, self._R_now, self._R_next, dt)

    # -- Reference source ---------------------------------------------------------------
    def sample(self, t: Optional[float] = None) -> Reference:
        """The current reference. `t` is accepted (and ignored) to match Trajectory."""
        return Reference(
            t=float(self._t if t is None else t),
            p=self.p_ref.copy(),
            v=self.v_ref.copy(),
            a=self.a_ref.copy(),
            R=self._R_now.copy(),
            omega=self._omega.copy(),
            thrust_ff=Maneuver.required_thrust(self.a_ref),
            spin=0.0,
            kind="human",
        )

    # The env reads `.duration` and `.maneuver.kind`; both are set in __init__.


class ShiftedTrajectory:
    """
    A sampled trajectory, moved to `p0` and rotated to `yaw0`.

    Why this exists: the sampler's manoeuvres are built for the training volume - a flip
    starts at the spawn point with the sampler's heading. During a live flight the pilot
    is hovering somewhere else, so the manoeuvre has to be RELOCATED onto them. The
    transform is a rigid motion (yaw rotation + translation), which preserves every
    dynamic property: p, v, a and R rotate, |a| and |omega| are unchanged, so the
    thrust feed-forward and the feasibility of the manoeuvre are exactly the sampler's.

    OMEGA IS DELIBERATELY NOT ROTATED - this mirrors trajectories.ShiftedManeuver, which
    has always done it correctly. The body frame turns WITH the vehicle, so for R' = Q R
    the body rate is [omega']_x = R'^T R'dot = R^T Q^T Q Rdot = R^T Rdot = [omega]_x: the
    same body-frame vector. Rotating it by Q, by analogy with v and a, describes a
    DIFFERENT manoeuvre - a 90 deg yaw relocation of a pitch flip would claim a roll rate -
    and it put an error into the policy's w_err channel that grows with the heading
    difference, i.e. worst exactly when the pilot happens to face the other way.

    At t = 0 of every sampled manoeuvre the reference is a level hover at rest
    (`p = p0`, v = 0, a = 0, thrust = m g, yaw = the sampler's). After shifting, t = 0 is
    therefore a level hover AT THE PILOT'S POSITION AND HEADING - a handover the policy
    can fly without a step, provided the "take over" happens while hovering.

    `t0` handles the clock: the env samples with its own ABSOLUTE time, so a manoeuvre
    launched 30 s into a live flight must be sampled at `t - t0`.
    """

    def __init__(self, traj: Trajectory, p0: np.ndarray, yaw0: float, t0: float = 0.0) -> None:
        self.traj = traj
        self.t0 = float(t0)
        self.duration = self.t0 + float(traj.duration)
        self.maneuver = traj.maneuver
        r0 = traj.sample(0.0)
        dyaw = float(yaw0) - yaw_of(r0.R)
        c, s = math.cos(dyaw), math.sin(dyaw)
        self._Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        self._p_shift = np.asarray(p0, dtype=np.float64) - self._Rz @ np.asarray(r0.p, dtype=np.float64)

    @property
    def kind(self) -> str:
        return str(getattr(self.maneuver, "kind", "trajectory"))

    def local_time(self, t: float) -> float:
        return float(t) - self.t0

    def sample(self, t: float) -> Reference:
        r = self.traj.sample(float(t) - self.t0)
        return Reference(
            t=r.t,
            p=self._Rz @ r.p + self._p_shift,
            v=self._Rz @ r.v,
            a=self._Rz @ r.a,
            R=self._Rz @ r.R,
            omega=r.omega,          # BODY frame - invariant, NOT rotated (see the class note)
            thrust_ff=r.thrust_ff,
            spin=r.spin,
            kind=r.kind,
        )


def hover_source(p0: np.ndarray, yaw0: float, v0: Optional[np.ndarray] = None) -> HumanTarget:
    """A HumanTarget parked in hover at a point - the "hold this" reference."""
    target = HumanTarget(p0, yaw0=yaw0, v0=v0)
    target.clear_command()
    return target
