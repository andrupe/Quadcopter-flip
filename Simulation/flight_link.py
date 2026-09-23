# -*- coding: utf-8 -*-
"""
The command sink: where a live-flight command would go if a Crazyflie were attached.

This module exists so the live flight program produces its output in ONE place, in the
form the vehicle consumes, whether the destination is the MuJoCo plant (today) or a
Crazyradio link (later). It deliberately contains no Crazyradio code yet, but it fixes
the interface and the units so that "later" is a ~30-line adapter rather than a refactor.

WHAT THE VEHICLE ACTUALLY RECEIVES
----------------------------------
Both control paths in the live program (the manual outer loop and the trained policy)
converge on the same pair:

    thrust   collective in newtons, produced either by the pilot's latched throttle or by
             the policy's action channel 0 mapped as 0.5*(a + 1)*maxThr
    omega    body-rate setpoint in rad/s: the pilot's stick rate commands, or the
             policy's action channels 1..3 scaled by (20, 20, 4) rad/s

On a Crazyflie that is exactly the RATE-MODE setpoint: roll/pitch/yaw rate in deg/s plus
thrust as a percentage of the configured maximum. `RadioSetpoint.from_sim` performs that
conversion ONCE, here, so the numbers the GUI, the telemetry and a future radio link see
cannot disagree about units.

THE CRTP PACKET (for the later implementation, not implemented here)
-------------------------------------------------------------------
cflib's `Crazyflie.commander.send_setpoint(roll, pitch, yawrate, thrust)` packs the same
four quantities into the commander packet:

    struct setpoint { float roll; float pitch; float yawrate; uint16_t thrust; }
        roll/pitch  deg/s   (+-2000 dps gyro range, this vehicle clips at 20 rad/s = 1146)
        yawrate     deg/s   (clipped at 4 rad/s = 229)
        thrust      uint16   in units of 65535 * 0.01 % - 0.1 .. 100.0 % (already a
                             percentage-of-max packet field, hence thrust_pct below)

The radio path also needs, and this interface already carries, the pieces that are easy
to forget until they bite:

  * a bounded send rate - CRTP setpoints must be refreshed at ~100 Hz or the vehicle's
    safety timeout drops the motors; the live program calls `send` once per control step
    and `NullLink` counts the packets so the rate is visible in the HUD;
  * a connection lifecycle (`open`/`close`) separate from the flight loop;
  * the ability to say "not implemented" loudly instead of silently flying nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

MAX_RATE_ROLL_PITCH = 20.0   # rad/s, the trained policy's authority (env.max_rate_xy)
MAX_RATE_YAW = 4.0           # rad/s, env.max_rate_z
GYRO_FULL_SCALE_DPS = 2000.0  # BMI088 range the numbers must respect


@dataclass
class RadioSetpoint:
    """One command frame, in vehicle units."""

    t: float                      # sim time the command was produced at, s
    roll_rate_dps: float          # deg/s
    pitch_rate_dps: float         # deg/s
    yaw_rate_dps: float           # deg/s
    thrust_pct: float             # percent of maxThr, 0..100 (CRTP setpoint field)
    source: str = "manual"        # "manual" | "policy"

    @classmethod
    def from_sim(
        cls,
        t: float,
        thrust_newtons: float,
        omega_rads: np.ndarray,
        max_thrust: float,
        source: str = "manual",
    ) -> "RadioSetpoint":
        omega = np.asarray(omega_rads, dtype=np.float64).reshape(3)
        return cls(
            t=float(t),
            roll_rate_dps=float(math.degrees(omega[0])),
            pitch_rate_dps=float(math.degrees(omega[1])),
            yaw_rate_dps=float(math.degrees(omega[2])),
            thrust_pct=float(np.clip(100.0 * thrust_newtons / max(1e-9, max_thrust), 0.0, 100.0)),
            source=source,
        )

    def clamped(self) -> "RadioSetpoint":
        """Clip to the vehicle's own limits (what the firmware would otherwise do)."""
        lim = math.degrees(MAX_RATE_ROLL_PITCH)
        yaw_lim = math.degrees(MAX_RATE_YAW)
        return RadioSetpoint(
            t=self.t,
            roll_rate_dps=float(np.clip(self.roll_rate_dps, -lim, lim)),
            pitch_rate_dps=float(np.clip(self.pitch_rate_dps, -lim, lim)),
            yaw_rate_dps=float(np.clip(self.yaw_rate_dps, -yaw_lim, yaw_lim)),
            thrust_pct=float(np.clip(self.thrust_pct, 0.0, 100.0)),
            source=self.source,
        )

    def summary(self) -> str:
        return (f"{self.source:6s} rp {self.roll_rate_dps:+7.0f} {self.pitch_rate_dps:+7.0f} "
                f"yaw {self.yaw_rate_dps:+6.0f} dps  thr {self.thrust_pct:5.1f}%")


class CommandLink:
    """A destination for command frames. The flight loop only ever calls `send`."""

    name = "link"

    def open(self) -> None:
        raise NotImplementedError

    def send(self, setpoint: RadioSetpoint) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name

    # Counters shared by the implementations (the HUD shows the send rate, because a
    # radio link that stops streaming is a crash, not a slowdown).
    sent: int = 0
    last: Optional[RadioSetpoint] = None


class NullLink(CommandLink):
    """The sim-only sink: validates and counts frames, transmits nothing."""

    name = "sim-only (no radio)"

    def __init__(self) -> None:
        self.sent = 0
        self.last = None

    def open(self) -> None:
        pass

    def send(self, setpoint: RadioSetpoint) -> None:
        self.sent += 1
        self.last = setpoint

    def close(self) -> None:
        pass

    def describe(self) -> str:
        return f"{self.name}, {self.sent} frames sent"


class CrazyradioLink(CommandLink):
    """
    NOT IMPLEMENTED - placeholder with the exact contract to fill in.

    To finish this (later, deliberately not now):

        from cflib.crazyflie import Crazyflie
        from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

        self.cf = SyncCrazyflie(self.uri)          # uri e.g. "radio://0/80/2M/E7E7E7E701"
        self.cf.open_link()
        self.commander = self.cf.cf.commander      # send_setpoint(roll, pitch, yawrate, thrust)

        def send(self, sp):
            sp = sp.clamped()
            self.commander.send_setpoint(sp.roll_rate_dps, sp.pitch_rate_dps,
                                         sp.yaw_rate_dps, sp.thrust_pct)

    Two safety items belong with that code, not with the flight loop:

      * a watchdog: if no frame has been sent for ~50 ms, send hover thrust (or arm a
        landing), because the firmware's own timeout drops the motors mid-air;
      * a lean-angle / thrust guard, since a rate-mode Crazyflie will happily fly into
        the ground at 20 rad/s if the pilot holds a stick.

    The live program only ever talks to `CommandLink`, so the day this class is
    implemented, nothing above it changes.
    """

    name = "crazyradio"

    def __init__(self, uri: str = "radio://0/80/2M/E7E7E7E7E7") -> None:
        self.uri = uri
        self.sent = 0
        self.last = None

    def open(self) -> None:
        raise NotImplementedError(
            "CrazyradioLink is a placeholder: the live program is sim-only. "
            "Fill in open()/send()/close() with cflib (see the class docstring)."
        )

    def send(self, setpoint: RadioSetpoint) -> None:  # pragma: no cover - not yet wired
        raise NotImplementedError

    def close(self) -> None:
        pass


def make_link(radio: bool = False, uri: str = "radio://0/80/2M/E7E7E7E7E7") -> CommandLink:
    """Factory used by the live program: sim-only by default, radio when asked for."""
    return CrazyradioLink(uri) if radio else NullLink()
