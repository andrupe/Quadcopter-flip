# -*- coding: utf-8 -*-
"""
Radio-ready live environment: `QuadFlipEnv` with the training volume guard relaxed.

WHY A SUBCLASS RATHER THAN AN EDIT
----------------------------------
`QuadFlipEnv` is the trained task description: its observation, reward, termination and
reference machinery are what the checkpoint was fitted to, and changing any of it would
invalidate both the policy and the acrobatics benchmarks. This module therefore inherits
everything and overrides exactly two things, both of which are about flying LIVE rather
than about the task:

    _check_termination   the training flight sphere is centred on a FIXED world point
                         (0, 0, 1.2) with r = 2.0 m. A live pilot is not constrained to
                         that volume - they may hover at 3 m and then fly a flip - so the
                         sphere would declare "out_of_volume" the moment the vehicle left
                         the training box. Ground contact and divergence are still
                         terminal (the sandbox stops on a crash, as before) - UNLESS the
                         flight opted into a ground start, see below.

    ground start         a live flight may BEGIN on the floor, the way a real quad waits
                         for its pilot. `arm_ground_start()` turns floor contact into a
                         runway until the vehicle has climbed above `GROUND_RELEASE_Z`
                         once; after that the floor is a crash surface again. Nothing
                         changes unless it is armed: the default is the old behaviour
                         (the floor is terminal from the very first step).

    adopt_state          manual flight drives the SAME plant object the env owns, so
                         whether the pilot is flying manually or the policy is flying
                         does not change which body is being simulated. This method just
                         re-seats the sensor model and the telemetry buffers on the
                         current state after a handover.

`set_external_reference` is the handover primitive for the reference source: the env
reads `self.traj` once per step, so anything with `sample(t)` (see `live_target.py`) can
be dropped in. Nothing in the env's own `reset()`/`step()` is bypassed in policy mode -
the policy flies through the *identical* code path it was trained and evaluated with,
with the sampler replaced.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import mujoco
import numpy as np

from quad_flip_env import QuadFlipEnv
from trajectories import Reference


class LiveFlightEnv(QuadFlipEnv):
    """`QuadFlipEnv` for live flights: same task, no fixed flight sphere."""

    #: Above this height a ground-start flight counts as airborne (see `arm_ground_start`).
    GROUND_RELEASE_Z: float = 0.20

    def __init__(self, **kwargs) -> None:
        # Ground-start latch, armed by the live program when its start pose is on the floor.
        # Default OFF so that everything else keeps the floor as a hard boundary.
        self.ground_start: bool = False
        self.airborne: bool = True
        super().__init__(**kwargs)

    # -- ground start --------------------------------------------------------------------
    def arm_ground_start(self, on: bool = True) -> None:
        """
        Treat the floor as a RUNWAY rather than a crash surface.

        With this armed, ground contact terminates the flight only once the vehicle has
        climbed above `GROUND_RELEASE_Z` at least once since the last arm. A flight that
        begins resting on its landing legs can therefore spool up, drift, hop and take off
        - and the first genuine touchdown after liftoff ends it exactly as before.
        """
        self.ground_start = bool(on)
        self.airborne = not bool(on)

    def ground_touch_is_fatal(self) -> bool:
        """
        Is a floor contact terminal right now? This also latches "airborne"; call it once
        per step (there is no contact event to hang the latch on, and in manual flight the
        env's own `step()` is not what advances the plant).
        """
        if float(self.quad.pos[2]) > self.GROUND_RELEASE_Z:
            self.airborne = True
        return (not self.ground_start) or self.airborne

    # -- termination --------------------------------------------------------------------
    def _check_termination(self) -> bool:
        if not np.all(np.isfinite(self.quad.state)):
            self.termination_reason = "divergent_state"
            return True
        # Ground contact is terminal - unless the flight started on the floor and has not
        # left it yet (`ground_touch_is_fatal` is called unconditionally, because it is
        # also what arms the takeoff).
        landed = self.ground_touch_is_fatal()
        if self.quad.check_ground_contact() and landed:
            self.termination_reason = "ground_crash"
            return True
        return False

    # -- live handover -------------------------------------------------------------------
    def adopt_state(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        vel: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
    ) -> None:
        """
        Re-anchor the env's own bookkeeping on the current plant state.

        The plant itself is shared (manual flight and the policy both drive `self.quad`),
        so this only has to reseat what CHANGED while the env was not stepping: the
        Lighthouse installation (it is anchored on a spawn point and would otherwise
        report a dead-reckoned position far from the vehicle), the observation history
        (frames from before the handover describe another flight) and the action history.
        """
        pos = np.asarray(pos, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        if vel is not None:
            self.quad.data.qvel[0:3] = np.asarray(vel, dtype=np.float64)
        if omega is not None:
            self.quad.data.qvel[3:6] = np.asarray(omega, dtype=np.float64)
        self.quad.data.qpos[0:3] = pos
        self.quad.data.qpos[3:7] = quat / max(1e-12, float(np.linalg.norm(quat)))
        mujoco.mj_forward(self.quad.model, self.quad.data)
        self.quad._update_state_properties()

        self.lighthouse.reset(pos, self.quad.vel, self.np_random, dr=self.dr_eff)
        self.rate_pid.reset()
        # The actor frame's origin moves with the handover: this is a new flight segment
        # (the GRU is reset and the histories are rebuilt just below), and the vehicle can
        # be metres from where the previous segment started. Without this the x,y channels
        # would report the distance flown since the OLD anchor - exactly the
        # out-of-distribution, position-dependent input that anchoring exists to remove.
        self.anchor_pos = pos.copy()
        # Rebuild the observation buffers the way `reset()` does: clearing them is not
        # enough, because `_get_stacked_obs()` concatenates the history buffer and an
        # empty one cannot be stacked (and frames from before the handover describe a
        # different flight anyway). The reference must already be installed - the actor
        # frame carries the reference errors.
        self.prev_action = self.hover_trim_action.copy()
        self.prev_prev_action = self.hover_trim_action.copy()
        frame = self._compute_actor_obs()
        self._last_actor_frame = frame.copy()
        aux = self._compute_encoder_aux()
        self.obs_buffer = [frame.copy() for _ in range(self.obs_latency + 1)]
        self.aux_buffer = [aux.copy() for _ in range(self.obs_latency + 1)]
        self._aux_delayed = aux.copy()
        self.obs_history_buffer = [frame.copy() for _ in range(self.obs_history_len)]

    def set_external_reference(self, source: Any) -> None:
        """
        Install a reference source (duck-typed `Trajectory`: `.sample(t)`, `.duration`,
        `.maneuver.kind`) and immediately evaluate one sample, so the actor frame is never
        built against a reference from the previous flight.
        """
        self.traj = source
        self.ref = source.sample(self.t)

    # -- convenience ---------------------------------------------------------------------
    def reference(self) -> Reference:
        return self._reference_or_default()

    def raw_actor_obs(self) -> np.ndarray:
        """The 29-dim actor frame, exactly as the training path assembles it."""
        return self._compute_actor_obs()

    def telemetry_dict(self) -> Dict[str, Any]:
        """The subset of the step info the live HUD and the link need."""
        return {
            "t": float(self.t),
            "position": self.quad.pos.copy(),
            "velocity": self.quad.vel.copy(),
            "omega": self.quad.omega.copy(),
            "reference_position": self.ref.p.copy() if self.ref is not None else self.quad.pos.copy(),
            "reference_velocity": self.ref.v.copy() if self.ref is not None else self.quad.vel.copy(),
            "maneuver": (self.traj.maneuver.kind if self.traj is not None else "none"),
            "dr_eff": float(self.dr_eff),
        }
