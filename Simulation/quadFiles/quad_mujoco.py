# -*- coding: utf-8 -*-
"""
High-Performance MuJoCo Physics Backend for Quadcopter Simulation.
Encapsulates mujoco.MjModel and mujoco.MjData while maintaining full
state and API compatibility with downstream controllers and RL environments.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# pyrefly: ignore [missing-import]
import mujoco

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.dirname(_THIS_DIR)
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

import utils
import config
from quadFiles.initQuad import makeMixerFM


class QuadcopterMuJoCo:
    """
    High-Performance MuJoCo Physics Backend for Quadcopter Simulation.
    Encapsulates mujoco.MjModel and mujoco.MjData while maintaining
    state and API compatibility with downstream flight controllers.
    """

    def __init__(self, Ti: float = 0.0, xml_path: Optional[str] = None):
        if xml_path is None:
            xml_path = os.path.join(_SIM_DIR, "assets", "scene.xml")

        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"MuJoCo XML scene file not found at: {xml_path}")

        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Standardize coordinate frame to ENU for MuJoCo (+Z is Up)
        config.orient = "ENU"

        # Physical parameters matching Bitcraze Crazyflie 2.X
        mB = 0.028       # Mass (kg)
        g = 9.81         # Gravity (m/s^2)
        dxm = 0.0325     # Arm length x (m)
        dym = 0.0325     # Arm length y (m)
        dzm = 0.01       # Motor height (m)
        kTh = 2.2e-8     # Thrust coefficient (N / (rad/s)^2)
        kTo = 7.94e-10   # Torque coefficient (Nm / (rad/s)^2)
        minW = 0.0       # Min motor speed (rad/s)
        maxW = 2600.0    # Max motor speed (rad/s)
        w_hover = 1767.0 # Hover motor speed (rad/s)
        thr_hover = mB * g / 4.0 # Hover thrust per motor (~0.06867 N)

        IB = np.array([
            [1.43e-5, 0.0,     0.0    ],
            [0.0,     1.43e-5, 0.0    ],
            [0.0,     0.0,     2.89e-5],
        ])

        self.params: Dict[str, Any] = {
            "mB": mB,
            "g": g,
            "dxm": dxm,
            "dym": dym,
            "dzm": dzm,
            "IB": IB,
            "invI": np.linalg.inv(IB),
            "IRzz": 1.0e-6,
            "Cd": 0.01,
            "kTh": kTh,
            "kTo": kTo,
            "minThr": 0.0,
            "maxThr": 0.60,
            "minWmotor": minW,
            "maxWmotor": maxW,
            "w_hover": w_hover,
            "thr_hover": thr_hover,
            "FF": 0.0,
            "tau": 0.005,
            "kp": 1.0,
            "damp": 1.0,
            "motorc1": 26.0,
            "motorc0": 0.0,
            "motordeadband": 1,
        }

        self.params["mixerFM"] = makeMixerFM(self.params)
        self.params["mixerFMinv"] = np.linalg.inv(self.params["mixerFM"])

        # Cache mocap body ID if target_marker exists
        try:
            self.target_mocap_id = self.model.body("target_marker").mocapid[0]
        except Exception:
            self.target_mocap_id = -1

        # Cache floor geom ID if exists
        try:
            self.floor_geom_id = self.model.geom("floor").id
        except Exception:
            self.floor_geom_id = -1

        self.t: float = float(Ti)
        self.wMotor: np.ndarray = np.ones(4) * w_hover
        self.thr: np.ndarray = np.ones(4) * thr_hover
        self.tor: np.ndarray = np.ones(4) * (kTo * (w_hover**2))
        self.vel_dot: np.ndarray = np.zeros(3)
        self.omega_dot: np.ndarray = np.zeros(3)
        self.acc: np.ndarray = np.zeros(3)

        self.reset()

    def reset(
        self,
        pos: Optional[Union[np.ndarray, list]] = None,
        quat: Optional[Union[np.ndarray, list]] = None,
    ) -> "QuadcopterMuJoCo":
        """
        Reset dynamic simulation state in MuJoCo C structure.
        """
        if pos is None:
            pos = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if quat is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0:3] = np.asarray(pos, dtype=np.float64)
        self.data.qpos[3:7] = np.asarray(quat, dtype=np.float64)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.params["thr_hover"]

        # Run forward kinematics to populate state views and sensors
        mujoco.mj_forward(self.model, self.data)

        self.wMotor = np.ones(4) * self.params["w_hover"]
        self.thr = np.ones(4) * self.params["thr_hover"]
        self.tor = np.ones(4) * (self.params["kTo"] * (self.params["w_hover"]**2))
        self.vel_dot = np.zeros(3)
        self.omega_dot = np.zeros(3)
        self.acc = np.zeros(3)

        self._update_state_properties()
        return self

    def set_target_marker(self, pos: Union[np.ndarray, list]):
        """
        Dynamically update mocap target marker position in 3D viewer.
        """
        if self.target_mocap_id >= 0:
            self.data.mocap_pos[self.target_mocap_id] = np.asarray(pos, dtype=np.float64)

    def _update_state_properties(self):
        """
        Extract state vectors directly from zero-copy C memory views.
        """
        self.pos = self.data.qpos[0:3].copy()
        self.quat = self.data.qpos[3:7].copy()
        self.vel = self.data.qvel[0:3].copy()
        self.omega = self.data.qvel[3:6].copy()

        # Euler angles (Roll, Pitch, Yaw)
        ypr = utils.quatToYPR_ZYX(self.quat)
        self.euler = ypr[::-1]  # [phi, theta, psi]
        self.psi = ypr[0]
        self.theta = ypr[1]
        self.phi = ypr[2]

        # Rotation matrix (DCM)
        self.dcm = utils.quat2Dcm(self.quat)

        # 21-element compatibility state vector:
        # [x, y, z, q0, q1, q2, q3, vx, vy, vz, p, q, r, wM1, 0, wM2, 0, wM3, 0, wM4, 0]
        self.state = np.array([
            self.pos[0], self.pos[1], self.pos[2],
            self.quat[0], self.quat[1], self.quat[2], self.quat[3],
            self.vel[0], self.vel[1], self.vel[2],
            self.omega[0], self.omega[1], self.omega[2],
            self.wMotor[0], 0.0,
            self.wMotor[1], 0.0,
            self.wMotor[2], 0.0,
            self.wMotor[3], 0.0,
        ], dtype=np.float64)

    def extended_state(self):
        """Compatibility method for legacy ODE interface."""
        self._update_state_properties()

    def update(
        self,
        t: float,
        dt: float,
        motor_cmd: np.ndarray,
        wind: Any = None,
    ):
        """
        Step simulation by mapping motor angular speeds (rad/s) or
        thrusts to MuJoCo actuator controls.
        """
        prev_vel = self.vel.copy()
        prev_omega = self.omega.copy()

        w_motor = np.clip(
            np.asarray(motor_cmd, dtype=np.float64),
            self.params["minWmotor"],
            self.params["maxWmotor"],
        )

        # Aerodynamic rotor thrust: F = kTh * w^2
        thrusts = self.params["kTh"] * (w_motor ** 2)

        # Direct assignment to MuJoCo control array
        self.data.ctrl[:] = thrusts

        # Dynamic wind injection
        if wind is not None and hasattr(wind, "randomWind"):
            velW, qW1, qW2 = wind.randomWind(t)
            wx = velW * np.cos(qW1) * np.cos(qW2)
            wy = velW * np.sin(qW1) * np.cos(qW2)
            wz = velW * np.sin(qW2)
            self.model.opt.wind[:] = [wx, wy, wz]

        # Step physics solver using exact sub-stepping
        sim_dt = self.model.opt.timestep
        n_substeps = max(1, int(round(dt / sim_dt)))
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.t = t + dt
        self.wMotor = w_motor
        self.thr = thrusts
        self.tor = self.params["kTo"] * (w_motor ** 2)

        self.vel_dot = (self.data.qvel[0:3] - prev_vel) / dt
        self.omega_dot = (self.data.qvel[3:6] - prev_omega) / dt
        self.acc = self.vel_dot.copy()

        self._update_state_properties()

    def check_ground_contact(self) -> bool:
        """
        Detect whether the quadcopter chassis or landing legs are touching the ground plane.
        """
        if self.pos[2] <= 0.03:
            return True

        if self.data.ncon > 0 and self.floor_geom_id >= 0:
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if contact.geom1 == self.floor_geom_id or contact.geom2 == self.floor_geom_id:
                    return True

        return False
