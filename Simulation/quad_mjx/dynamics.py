"""
JAX/MJX plant: the SAME closed loop as ``Simulation/quadFiles/quad_mujoco.py``.

WHAT IS DELIBERATELY IDENTICAL TO THE BASELINE (no domain shift)
  * The inner-loop rate PID is kept.  Action -> (throttle, omega_des) -> PID -> moments
    -> mixer -> motor speeds -> motor thrusts -> MuJoCo.  We do NOT bypass it and command
    body moments directly.
  * The PID reads the same signal the baseline feeds it: MuJoCo's ground-truth ``qvel[3:6]``
    plus the gyro bias.  (The baseline documents this as unrealistically well-informed and
    keeps it on purpose; changing it here would invalidate the tuned gains.)
  * Gains, torque limits, integral limits, the mixer, kTh/kTo, motor tau, the 1st-order
    motor lag, the directional tau_up/tau_down, dynamic battery sag and motor
    efficiencies all match.

WHAT WAS BROKEN IN THE PREVIOUS ATTEMPT (fixed here)
  1. ``substeps`` was 5 with a 1 ms model timestep, so one control step advanced 5 ms of
     physics while ``t`` advanced 10 ms -- the whole simulation ran at HALF SPEED and the
     reference was effectively twice as fast as the vehicle could fly.  SUBSTEPS now
     derives from ``SIM_DT / PHYSICS_DT``.
  2. The PID state (integrator, previous rate) was rebuilt every control step, so the
     loop had NO integral action across steps and a D-term with no memory.
  3. Anti-windup clamped the RAW integral and then multiplied by ``ki``, making the
     effective integral authority ~1/ki (~5000x) weaker than the baseline's.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from . import spec


# ======================================================================================
# STATIC PLANT GEOMETRY
# ======================================================================================
MIXER_FM = jnp.array([
    [spec.K_TH, spec.K_TH, spec.K_TH, spec.K_TH],
    [spec.DYM * spec.K_TH, -spec.DYM * spec.K_TH, -spec.DYM * spec.K_TH, spec.DYM * spec.K_TH],
    [-spec.DXM * spec.K_TH, -spec.DXM * spec.K_TH, spec.DXM * spec.K_TH, spec.DXM * spec.K_TH],
    [-spec.K_TO, spec.K_TO, -spec.K_TO, spec.K_TO],
])
# Mixer inverse: [thrust, mx, my, mz] -> [w1^2, w2^2, w3^2, w4^2], built from the NOMINAL
# kTh/kTo exactly as initQuad.makeMixerFM does.  thrust_scale is applied to the resulting
# thrusts, NOT to the mixer (same as quad_mujoco.update).
MIXER_FINV = jnp.linalg.inv(MIXER_FM)

# Rate PID gains (utils/rate_pid.py defaults).
KP = jnp.array([0.0015208, 0.0015208, 0.0016214])
KI = jnp.array([0.0001913, 0.0001913, 0.0022230])
KD = jnp.array([0.0000257, 0.0000257, 0.0000916])
MAX_TORQUE = jnp.array([0.010, 0.010, 0.003])
MAX_INTEGRAL = jnp.array([0.003, 0.003, 0.001])

W_HOVER = float(jnp.sqrt((spec.MASS * spec.GRAVITY / 4.0) / spec.K_TH))   # ~1918 rad/s


@struct.dataclass
class PlantIndex:
    """Static (host-computed) model indices; identical for every env."""
    body_id: int
    rotor_site_ids: jax.Array       # (4,)
    base_site_pos: jax.Array        # (4,3)
    accel_adr: int
    gyro_adr: int


@struct.dataclass
class DRParams:
    """Per-episode domain randomisation, all nominal at dr = 0."""
    mass: jax.Array
    com_offset: jax.Array           # (3,)
    inertia: jax.Array              # (3,) diagonal
    site_offsets: jax.Array         # (4,3)
    motor_eff: jax.Array            # (4,)
    tau_up: jax.Array
    tau_down: jax.Array
    sag_coef: jax.Array
    thrust_scale: jax.Array
    gyro_bias: jax.Array            # (3,)
    thrust_nl: jax.Array            # thrust curve nonlinearity
    rate_pid_noise: jax.Array       # inner loop rate PID noise


@struct.dataclass
class PlantState:
    """Persistent across the WHOLE episode -- the integrator must not be re-created."""
    w_motor: jax.Array              # (4,) rad/s
    pid_integral: jax.Array         # (3,) torque units (ki * integral(error dt))
    pid_prev_omega: jax.Array       # (3,)
    pid_started: jax.Array          # bool: False only for the very first substep
    dynamic_sag: jax.Array
    v_pol: jax.Array                # polarization / dynamic sag relaxation state
    soc: jax.Array                  # state of charge [0, 1]


def nominal_dr() -> DRParams:
    return DRParams(
        mass=jnp.array(spec.MASS),
        com_offset=jnp.zeros(3),
        inertia=jnp.array([1.685e-5, 1.685e-5, 3.359e-5]),
        site_offsets=jnp.zeros((4, 3)),
        motor_eff=jnp.ones(4),
        tau_up=jnp.array(spec.MOTOR_TAU),
        tau_down=jnp.array(spec.MOTOR_TAU),
        sag_coef=jnp.array(0.0),
        thrust_scale=jnp.array(1.0),
        gyro_bias=jnp.zeros(3),
        thrust_nl=jnp.array(0.0),
        rate_pid_noise=jnp.array(0.0),
    )


def sample_dr(key, dr) -> DRParams:
    """
    Draw the episode's distortions.  Mirrors ``QuadFlipEnv.reset``'s DR block exactly:
    every range is ``nominal + dr * (bound - nominal)`` and payload mass ADDS to the
    airframe mass (33 g -> 38 g), scaling the inertia by the same ratio.
    """
    k = jax.random.split(key, 8)
    d = jnp.clip(dr, 0.0, 1.0)

    com_dx = jax.random.uniform(k[0], (), minval=-1.0, maxval=1.0) * d * spec.COM_OFFSET_MAX_XY
    com_dy = jax.random.uniform(k[1], (), minval=-1.0, maxval=1.0) * d * spec.COM_OFFSET_MAX_XY
    com_dz = jax.random.uniform(k[2], (), minval=-1.0, maxval=1.0) * d * spec.COM_OFFSET_MAX_Z
    com_offset = jnp.array([com_dx, com_dy, com_dz])

    m_payload = jax.random.uniform(k[3], (), minval=0.0, maxval=d * spec.PAYLOAD_MASS_MAX)
    mass = spec.MASS + m_payload
    inertia = jnp.array([1.685e-5, 1.685e-5, 3.359e-5]) * (mass / spec.MASS)

    arm = jax.random.uniform(k[4], (4, 3), minval=-1.0, maxval=1.0) * d * spec.ARM_LENGTH_JITTER_MAX
    site_offsets = arm.at[:, 2].set(0.0)

    motor_eff = 1.0 - jax.random.uniform(
        k[5], (4,), minval=0.0, maxval=d * spec.MOTOR_MISMATCH_MAX)

    sag_coef = jax.random.uniform(k[6], (), minval=0.0, maxval=d * spec.DYNAMIC_SAG_COEF_MAX)

    tau_lo = spec.MOTOR_TAU - d * (spec.MOTOR_TAU - spec.MOTOR_TAU_RANGE[0])
    tau_hi = spec.MOTOR_TAU + d * (spec.MOTOR_TAU_RANGE[1] - spec.MOTOR_TAU)
    batt_lo, batt_hi = spec.BATTERY_THRUST_SCALE_RANGE
    k7 = jax.random.split(k[7], 6)
    tau_up = jax.random.uniform(k7[0], (), minval=tau_lo, maxval=tau_hi)
    tau_down = tau_up * (1.0 + jax.random.uniform(
        k7[1], (), minval=0.0, maxval=d * spec.MOTOR_TAU_DOWN_FACTOR))

    thrust_scale = jax.random.uniform(
        k7[2], (), minval=1.0 - d * (1.0 - batt_lo), maxval=1.0 + d * (batt_hi - 1.0))

    gyro_bias = jax.random.uniform(
        k7[3], (3,), minval=-d * spec.GYRO_BIAS_MAX, maxval=d * spec.GYRO_BIAS_MAX)

    thrust_nl = jax.random.uniform(
        k7[4], (), minval=-d * spec.THRUST_NL_MAX, maxval=d * spec.THRUST_NL_MAX)
    rate_pid_noise = d * spec.RATE_PID_NOISE_MAX

    return DRParams(
        mass=mass,
        com_offset=com_offset,
        inertia=inertia,
        site_offsets=site_offsets,
        motor_eff=motor_eff,
        tau_up=tau_up,
        tau_down=tau_down,
        sag_coef=sag_coef,
        thrust_scale=thrust_scale,
        gyro_bias=gyro_bias,
        thrust_nl=thrust_nl,
        rate_pid_noise=rate_pid_noise,
    )


def apply_model_dr(model, idx: PlantIndex, dr: DRParams):
    """
    Write the DR into the MJX model pytree, the way ``apply_hardware_distortions`` writes
    it into the MuJoCo model.  Under vmap each env carries its own model.
    """
    body_ipos = model.body_ipos.at[idx.body_id].set(dr.com_offset)
    body_mass = model.body_mass.at[idx.body_id].set(dr.mass)
    body_inertia = model.body_inertia.at[idx.body_id].set(dr.inertia)
    site_pos = model.site_pos.at[idx.rotor_site_ids].set(idx.base_site_pos + dr.site_offsets)
    return model.replace(
        body_ipos=body_ipos,
        body_mass=body_mass,
        body_inertia=body_inertia,
        site_pos=site_pos,
    )


def kth_effective(dr: DRParams):
    return spec.K_TH * dr.thrust_scale


def initial_plant_state() -> PlantState:
    # explicit dtypes: see the note in `env._reset_with_spec`.  A weak-typed leaf here would
    # make the state out of `reset` disagree with the state out of `step` and cost a second
    # XLA compilation of the whole rollout.
    return PlantState(
        w_motor=jnp.ones(4) * W_HOVER,
        pid_integral=jnp.zeros(3),
        pid_prev_omega=jnp.zeros(3),
        pid_started=jnp.asarray(False, jnp.bool_),
        dynamic_sag=spec.float_scalar(1.0),
        v_pol=spec.float_scalar(0.0),
        soc=spec.float_scalar(1.0),
    )


def mixer_fm(throttle, moments):
    """[thrust, mx, my, mz] -> per-rotor target angular speed (rad/s), clipped."""
    wrench = jnp.array([throttle, moments[0], moments[1], moments[2]])
    w_sq = MIXER_FINV @ wrench
    w_sq = jnp.clip(w_sq, spec.MIN_W ** 2, spec.MAX_W ** 2)
    return jnp.sqrt(w_sq)


def rate_pid_step(integral, prev_omega, started, omega_des, omega_meas, sub_dt):
    """
    One PID update, term for term the same as ``RatePIDController.update``:

        i <- clamp(i + ki * error * dt, +-max_integral)      (torque units!)
        d  = -kd * (meas - prev) / dt                        (derivative on measurement)
        tau <- clamp(kp*error + i + d, +-max_torque)

    ``started`` reproduces ``prev_omega is None``: the D term is exactly 0 on the very
    first call after a reset, which is what the baseline does.
    """
    error = omega_des - omega_meas
    new_integral = jnp.clip(integral + KI * error * sub_dt, -MAX_INTEGRAL, MAX_INTEGRAL)
    d_meas = jnp.where(started, (omega_meas - prev_omega) / sub_dt, 0.0)
    torque = KP * error + new_integral - KD * d_meas
    torque = jnp.clip(torque, -MAX_TORQUE, MAX_TORQUE)
    return torque, new_integral, omega_meas


def step_plant(model, data, plant: PlantState, dr: DRParams, throttle, omega_des,
               accel_adr: int = 0):
    """
    Advance one CONTROL step (SUBSTEPS x PHYSICS_DT) with the closed inner loop.

    Returns (data, plant_state, telemetry).
    """
    sub_dt = spec.SIM_DT / spec.SUBSTEPS
    k_th_eff = kth_effective(dr)

    def substep(carry, _):
        data, w_curr, integral, prev_omega, started, sag, v_pol, soc = carry

        # 1. inner loop. The PID sees ground truth + the gyro bias.
        omega_meas = data.qvel[3:6] + dr.gyro_bias
        moments, integral, prev_omega = rate_pid_step(
            integral, prev_omega, started, omega_des, omega_meas, sub_dt)

        # 2. mixer -> per-rotor target speed + 16-bit integer PWM command quantization
        w_target = mixer_fm(throttle, moments)
        w_ratio = w_target / spec.MAX_W
        # Match Crazyflie powerDistributionCap: when any motor exceeds 100%, subtract
        # the overshoot from ALL motors to preserve angular torque authority over collective thrust.
        w_max = jnp.max(w_ratio)
        diff = jnp.maximum(0.0, w_max - 1.0)
        w_capped = jnp.maximum(0.0, w_ratio - diff)
        w_pwm = jnp.round(jnp.clip(w_capped, 0.0, 1.0) * 65535.0) / 65535.0
        w_target = w_pwm * spec.MAX_W

        # 3. directional 1st-order motor lag (tau_down slower than tau_up)
        tau = jnp.where(w_target >= w_curr, dr.tau_up, dr.tau_down)
        alpha = sub_dt / (tau + sub_dt)
        w_next = jnp.clip(w_curr + alpha * (w_target - w_curr), spec.MIN_W, spec.MAX_W)

        # 4. dynamic battery sag: SoC, internal resistance, polarization state
        burst = jnp.mean((w_next / spec.MAX_W) ** 2)
        tau_rec = 0.40
        alpha_pol = sub_dt / (tau_rec + sub_dt)
        v_pol = jnp.where(dr.sag_coef > 0.0, v_pol + alpha_pol * (burst - v_pol), 0.0)
        soc = jnp.where(dr.sag_coef > 0.0, jnp.maximum(0.2, soc - 0.001 * burst * sub_dt), 1.0)
        v_drop = dr.sag_coef * (0.6 * burst + 0.4 * v_pol)
        sag = jnp.where(dr.sag_coef > 0.0, jnp.maximum(0.0, soc * (1.0 - v_drop)), 1.0)

        # 5. rotor thrusts (kTh_effective includes battery thrust scale, sag, nonlinearity)
        nl = 1.0 + dr.thrust_nl * (w_next / spec.MAX_W - 0.5)
        thrusts = (k_th_eff * sag) * dr.motor_eff * nl * (w_next ** 2)
        data = data.replace(ctrl=thrusts)

        # 6. physics
        data = _mj_step(model, data)
        return (data, w_next, integral, prev_omega, jnp.asarray(True, jnp.bool_), sag, v_pol, soc), \
            data.qvel[3:6]

    (data, w_motor, integral, prev_omega, started, sag, v_pol, soc), omega_hist = jax.lax.scan(
        substep,
        (data, plant.w_motor, plant.pid_integral, plant.pid_prev_omega,
         plant.pid_started, plant.dynamic_sag, plant.v_pol, plant.soc),
        None,
        length=spec.SUBSTEPS,
    )

    omega_filtered = jnp.mean(omega_hist, axis=0)

    # v_batt_norm = sqrt(kTh_effective / kTh * sag) = sqrt(thrust_scale * sag)
    v_batt_norm = jnp.sqrt(jnp.maximum(dr.thrust_scale * sag, 0.0))

    # Accelerometer (body frame specific force) at the FINAL state -- the baseline reads
    # sensordata after the substep loop in _update_state_properties().
    specific_force_b = jax.lax.dynamic_slice(data.sensordata, (accel_adr,), (3,))

    new_plant = PlantState(
        w_motor=w_motor,
        pid_integral=integral,
        pid_prev_omega=prev_omega,
        pid_started=started,
        dynamic_sag=sag,
        v_pol=v_pol,
        soc=soc,
    )
    telemetry = {
        "omega_filtered": omega_filtered,
        "specific_force_b": specific_force_b,
        "v_batt_norm": v_batt_norm,
        "motor_speed_norm": w_motor / spec.MAX_W,
        "dynamic_sag": sag,
    }
    return data, new_plant, telemetry


def _mj_step(model, data):
    from mujoco import mjx
    return mjx.step(model, data)
