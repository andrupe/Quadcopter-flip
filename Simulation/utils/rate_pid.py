# -*- coding: utf-8 -*-
"""
High-Frequency Discrete-Time Rate PID Controller for Quadcopter Inner-Loop Stabilization.
Designed for Bitcraze Crazyflie 2.X dynamics running at 500 Hz - 1000 Hz.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


class RatePIDController:
    """
    Discrete-time 3-axis angular rate PID controller.
    
    Tracks desired body angular velocities [p_des, q_des, r_des] in rad/s
    and outputs stabilizing body torques [tau_x, tau_y, tau_z] in N*m.
    
    Features:
    - Derivative on measurement to prevent setpoint derivative kick during rapid flips.
    - Anti-windup integration clamping.
    - Asymmetric physical saturation bounds aligned with Crazyflie 2.X rotor authority.
    """

    def __init__(
        self,
        kp: Optional[Union[np.ndarray, list, float]] = None,
        ki: Optional[Union[np.ndarray, list, float]] = None,
        kd: Optional[Union[np.ndarray, list, float]] = None,
        max_torque_xy: float = 0.010,   # N*m max roll/pitch torque authority
        max_torque_z: float = 0.003,    # N*m max yaw torque authority
        max_integral_xy: float = 0.003, # N*m anti-windup ceiling for roll/pitch
        max_integral_z: float = 0.001,  # N*m anti-windup ceiling for yaw
    ):
        # Tuned gains calibrated for Bitcraze Crazyflie 2.X dynamics
        # Delivers critically damped rate tracking (eliminates overshoot & chattering)
        # with anti-windup integration to eliminate hardware asymmetries and trim bias.
        if kp is None:
            self.kp = np.array([0.0015208, 0.0015208, 0.0016214], dtype=np.float64)
        else:
            self.kp = np.asarray(kp, dtype=np.float64)

        if ki is None:
            self.ki = np.array([0.0001913, 0.0001913, 0.0022230], dtype=np.float64)
        else:
            self.ki = np.asarray(ki, dtype=np.float64)

        if kd is None:
            self.kd = np.array([0.0000257, 0.0000257, 0.0000916], dtype=np.float64)
        else:
            self.kd = np.asarray(kd, dtype=np.float64)

        self.max_torque = np.array([max_torque_xy, max_torque_xy, max_torque_z], dtype=np.float64)
        self.max_integral = np.array([max_integral_xy, max_integral_xy, max_integral_z], dtype=np.float64)

        self.integral: np.ndarray = np.zeros(3, dtype=np.float64)
        self.prev_omega: Optional[np.ndarray] = None
        self.last_torque: np.ndarray = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        """Reset integrator and previous measurement states."""
        self.integral = np.zeros(3, dtype=np.float64)
        self.prev_omega = None
        self.last_torque = np.zeros(3, dtype=np.float64)

    def set_gains(
        self,
        kp: Optional[Union[np.ndarray, list, float]] = None,
        ki: Optional[Union[np.ndarray, list, float]] = None,
        kd: Optional[Union[np.ndarray, list, float]] = None,
    ) -> None:
        """Update PID gains dynamically."""
        if kp is not None:
            self.kp = np.asarray(kp, dtype=np.float64) if not np.isscalar(kp) else np.full(3, kp, dtype=np.float64)
        if ki is not None:
            self.ki = np.asarray(ki, dtype=np.float64) if not np.isscalar(ki) else np.full(3, ki, dtype=np.float64)
        if kd is not None:
            self.kd = np.asarray(kd, dtype=np.float64) if not np.isscalar(kd) else np.full(3, kd, dtype=np.float64)

    def get_gains(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return current PID gain arrays (kp, ki, kd)."""
        return self.kp.copy(), self.ki.copy(), self.kd.copy()

    def to_dict(self) -> Dict[str, list]:
        """Serialize controller gains to dictionary."""
        return {
            "kp": self.kp.tolist(),
            "ki": self.ki.tolist(),
            "kd": self.kd.tolist(),
            "max_torque": self.max_torque.tolist(),
            "max_integral": self.max_integral.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RatePIDController:
        """Instantiate RatePIDController from serialized dictionary."""
        return cls(
            kp=data.get("kp"),
            ki=data.get("ki"),
            kd=data.get("kd"),
            max_torque_xy=data.get("max_torque", [0.01, 0.01, 0.003])[0],
            max_torque_z=data.get("max_torque", [0.01, 0.01, 0.003])[2],
            max_integral_xy=data.get("max_integral", [0.003, 0.003, 0.001])[0],
            max_integral_z=data.get("max_integral", [0.003, 0.003, 0.001])[2],
        )

    def update(
        self,
        omega_des: np.ndarray,
        omega_meas: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Compute control torques for a single time step.

        Parameters
        ----------
        omega_des : np.ndarray
            Desired body angular velocity [p_des, q_des, r_des] in rad/s.
        omega_meas : np.ndarray
            Current measured angular velocity [p_meas, q_meas, r_meas] in rad/s.
        dt : float
            Inner-loop timestep in seconds (typically 0.001s to 0.002s).

        Returns
        -------
        torques : np.ndarray
            Commanded body torques [tau_x, tau_y, tau_z] in N*m.

        IMPLEMENTATION NOTE. This runs at the physics substep rate and is called 10x per
        environment step, so at 3 elements per axis the numpy call overhead (sub, mul,
        clip, copy - each a dispatch on a 3-vector) dominates the arithmetic by roughly
        an order of magnitude. The loop below keeps every operation but does it on
        Python floats: same expressions, same order ((ki*error)*dt, (p+i)+d), same
        clamping semantics as np.clip, same NaN propagation. `scratch/check_fast_math.py`
        checks it against the vector formulation.
        """
        if not isinstance(omega_des, np.ndarray):
            omega_des = np.asarray(omega_des, dtype=np.float64)
        if not isinstance(omega_meas, np.ndarray):
            omega_meas = np.asarray(omega_meas, dtype=np.float64)
        dt = max(1e-6, float(dt))

        kp, ki, kd = self.kp, self.ki, self.kd
        integral = self.integral
        max_integral = self.max_integral
        max_torque = self.max_torque
        prev_omega = self.prev_omega
        torque = np.empty(3, dtype=np.float64)

        for j in range(3):
            meas = float(omega_meas[j])
            error = float(omega_des[j]) - meas

            # 1-2. Error and proportional term
            p_term = float(kp[j]) * error

            # 3. Integral term with anti-windup clamping. `(ki * error) * dt`, in that
            # order, is exactly what the vector expression evaluated.
            i_term = float(integral[j]) + float(ki[j]) * error * dt
            i_lim = float(max_integral[j])
            if i_term > i_lim:
                i_term = i_lim
            elif i_term < -i_lim:
                i_term = -i_lim
            integral[j] = i_term

            # 4. Derivative term on measurement (prevents derivative kick on step changes)
            if prev_omega is not None:
                d_term = -float(kd[j]) * ((meas - float(prev_omega[j])) / dt)
            else:
                d_term = 0.0

            # 5. Total torque with actuator authority clamping
            tau = (p_term + i_term) + d_term
            t_lim = float(max_torque[j])
            if tau > t_lim:
                tau = t_lim
            elif tau < -t_lim:
                tau = -t_lim
            torque[j] = tau

        self.prev_omega = omega_meas.copy()
        self.last_torque = torque.copy()

        return torque
