"""
Vectorised MJX environment: a faithful PORT of ``Simulation/quad_flip_env.py``.

Everything the baseline does to the observation, the reward and the episode lifecycle is
reproduced here, including the parts the previous MJX attempt dropped:

  * the actor frame is built from the LIGHTHOUSE ESTIMATE (never truth), and the reference
    errors are computed against that same estimate;
  * the 4-dim encoder aux (real accelerometer specific force + battery sag) and the 3-dim
    reference feed-forward (a + g e_z, world frame), in the frozen `[o_t | z | ref_ff]`
    order with `aux` sitting AFTER ref_ff so the actor stays a clean prefix;
  * observation latency (the current frame is recorded and an older one is served), applied
    to `o_t` and to `aux` but NOT to `ref_ff` or the privileged block;
  * the cauchy tracking kernels with the PER-FAMILY tolerances, the bounded action-smoothness
    bonus, and FLIP_PROGRESS riding `w_att`;
  * the flight-envelope termination (flip ceilings, the reference tunnel, the take-off
    grace window and the outer sphere) with its curriculum scale;
  * domain randomisation sampled per episode and written into the model + actuator maths;
  * Perlin wind, gyro bias, IMU/attitude/accel noise;
  * the evaluate.py lighthouse failure injection (loss / outage / runaway / teleport).

The reward is scored against TRUTH; the observation is not.  That asymmetry is deliberate
in the baseline and is preserved here.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from mujoco import mjx

from . import spec as S, dynamics as D, lighthouse as LH, wind as WIND, encoder as ENC
from .trajectories import IDX_FLIP, IDX_HOVER, Reference, TrajSpec, sample as traj_sample
from . import sampler as SAMP

# termination codes (names printed by the trainer / evaluate)
TERM_NONE, TERM_DIVERGED, TERM_GROUND, TERM_FLIP_CEIL, TERM_FLIP_XY, TERM_TUNNEL, TERM_OUT = range(7)
TERM_NAMES = ("none", "divergent_state", "ground_crash", "flip_ballooned_ceiling",
              "flip_drifted_xy", "tunnel_breach", "out_of_volume")

LAT_MAX = S.OBS_LATENCY_MAX_STEPS


@struct.dataclass
class FailureState:
    """evaluate.py's LighthouseFailure, as env state."""
    mode: jax.Array            # int32
    direction: jax.Array       # (3,)
    offset: jax.Array          # (3,)
    lie_v: jax.Array           # (3,)
    max_lie: jax.Array
    next_event: jax.Array
    blinding: jax.Array        # bool
    active: jax.Array          # bool
    latched: jax.Array         # bool: reception latch under motor power


@struct.dataclass
class EnvState:
    model: mjx.Model                 # per-episode, with the DR written in
    data: mjx.Data
    plant: D.PlantState
    lh: LH.LighthouseState
    traj: TrajSpec
    t: jax.Array
    step_count: jax.Array
    prev_action: jax.Array           # (4,) POST-EMA applied action
    anchor_pos: jax.Array            # (3,)
    gru_h: jax.Array
    gru_z: jax.Array
    dr: D.DRParams
    dr_level: jax.Array
    wind: WIND.WindState
    envelope_scale: jax.Array
    obs_latency: jax.Array
    obs_buf: jax.Array               # (LAT_MAX+1, 29)
    aux_buf: jax.Array               # (LAT_MAX+1, 4)
    # flip-progress bookkeeping
    flip_spin_veh: jax.Array
    flip_axis: jax.Array             # (3,)
    flip_axis_valid: jax.Array
    flip_last_kind: jax.Array
    flip_last_spin: jax.Array
    flip_progress_err: jax.Array
    has_lifted_off: jax.Array
    failure: FailureState
    rng: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    termination: jax.Array


@struct.dataclass
class Observation:
    actor_obs: jax.Array             # (48) [o_t 29 | z 16 | ref_ff 3]
    critic_obs: jax.Array            # (96) [o_t 29 | z 16 | ref_ff 3 | aux 4 | priv 44]
    frame: jax.Array                 # (33) encoder frame [o_t 29 | aux 4]
    actor_frame: jax.Array           # (29) delayed o_t
    reward_terms: jax.Array          # (4,) [r_pos, r_vel, r_att, r_rate]


@struct.dataclass
class EnvConfig:
    episode_seconds: float = S.EPISODE_SECONDS
    obs_noise: bool = True
    random_wind: bool = True
    random_initial_state: bool = True
    random_initial_pos: bool = True
    random_initial_att: bool = True
    random_initial_vel: bool = True
    random_battery: bool = True
    use_encoder: bool = True
    # failure injection (evaluation only)
    lhf_mode: int = S.LHF_NONE
    lhf_start_s: float = 0.0
    lhf_severity: float = 1.0
    lhf_outage_s: float = 1.5
    lhf_period_s: float = 4.0
    lhf_teleport_m: float = 6.0
    lhf_z_too: bool = False
    # TIP 4: skip MJX's contact pipeline (collision + contact constraint rows + the
    # Newton solver).  A ground touch is a TERMINATION, so landing dynamics are never
    # trained through and the constraint solve buys nothing; the ground test then falls
    # back to the geometric `qpos[2] <= GROUND_TERMINATE_Z`, which is what it already
    # dominates.  OFF by default so every parity/diagnostic path stays bit-identical --
    # the trainer opts in (see `train_mjx.DISABLE_CONTACTS`).
    disable_contacts: bool = False


class QuadFlipMJXEnv:
    """Batched MJX environment. `reset`/`step` are pure and vmappable."""

    def __init__(self, xml_path: Optional[str] = None, config: Optional[EnvConfig] = None,
                 encoder_path: Optional[str] = None, use_encoder: bool = True):
        import mujoco

        if xml_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            xml_path = os.path.join(os.path.dirname(here), "assets", "scene.xml")
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"MuJoCo scene XML not found: {xml_path}")

        self.cfg = config or EnvConfig()

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        if self.cfg.disable_contacts:
            # MJX honours DisableBit.CONTACT (`collision_driver` early-returns and
            # `constraint.make_efc_type` emits no contact rows), but the flag has to be set
            # on the MjModel BEFORE `put_model`: make_condim / make_efc_type are evaluated
            # at conversion time, so flipping it on the mjx.Model afterwards leaves the
            # static efc layout describing contacts that are never built.
            self.mj_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.base_data = mjx.make_data(self.mjx_model)

        bid = int(np.argmax(self.mj_model.body_mass))
        sites = jnp.array([mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, n)
                           for n in ("rotor_fl", "rotor_fr", "rotor_rr", "rotor_rl")])
        self.idx = D.PlantIndex(body_id=bid, rotor_site_ids=sites,
                                base_site_pos=jnp.array(self.mj_model.site_pos[sites]),
                                accel_adr=0, gyro_adr=3)
        # the accelerometer is the FIRST sensor in scene/quadcopter.xml; resolve it properly
        try:
            self.accel_adr = int(self.mj_model.sensor_adr[
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer")])
        except Exception:
            self.accel_adr = 0

        self.traj_cfg = SAMP.traj_cfg(self.cfg.episode_seconds)
        self.lh_cfg = LH.LighthouseConfig().replace(
            max_fix_range=S.LIGHTHOUSE_FIX_RANGE_MULT * S.FLIGHT_RADIUS)
        self.wind_max = S.MAX_WIND_SPEED
        self.weights = SAMP.default_weights()
        self.max_steps = int(round(self.cfg.episode_seconds / S.SIM_DT))

        self.use_encoder = bool(use_encoder and self.cfg.use_encoder)
        self.gru = None
        if self.use_encoder:
            path = encoder_path
            if path is None:
                cand = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))), "logs", "encoder_gru.pt")
                path = cand if os.path.isfile(cand) else None
            if path is not None and os.path.isfile(path):
                self.gru = ENC.load(path)
                self.encoder_path = path

        self.actor_dim = (S.ACTOR_TOTAL_DIM + S.Z_DIM + S.REF_FF_DIM
                          if self.gru is not None else S.ACTOR_DIM_NO_ENCODER)
        self.action_dim = S.ACTION_DIM
        self.critic_dim = S.CRITIC_OBS_DIM

    # -- config hooks ---------------------------------------------------------------
    def set_weights(self, weights):
        self.weights = jnp.asarray(weights)

    def set_maneuver_weight(self, name, weight):
        if name not in S.KIND_INDEX:
            raise ValueError(f"unknown manoeuvre {name!r}; expected one of {S.KIND_NAMES}")
        w = np.array(self.weights)
        w[S.KIND_INDEX[name]] = float(weight)
        self.weights = jnp.asarray(w)

    def get_maneuver_weights(self):
        return np.array(self.weights)

    # ==================================================================================
    # reset
    # ==================================================================================
    def reset(self, key, kind_pin: int = -1, dr_level=0.0, envelope_scale=None):
        """
        Draw a fresh reference, then reset.  The REFERENCE DRAW is the expensive part of a
        reset (a rejection loop over up to 40 candidates, each screened over 96/256 points),
        so training never calls this per env per step: it pre-draws a small pool with
        `sample()` and resets from it via `reset_from_spec`.  Keep this as the reference
        implementation (and as the eval path, where `kind_pin` pins a family).
        """
        # split(key, 7)[1] is the k_traj slot `_reset_with_spec` deliberately skips over,
        # so a reset still consumes the SAME seven subkeys -- and everything downstream stays
        # bit-identical to the pre-pool env for a given seed.
        k_traj = jax.random.split(key, 7)[1]
        spec_ = SAMP.sample(k_traj, self.traj_cfg, self.weights,
                            mass=S.MASS, kind_pin=kind_pin)
        return self._reset_with_spec(key, spec_, dr_level, envelope_scale)

    def reset_from_spec(self, key, spec_, dr_level=0.0, envelope_scale=None):
        """Reset onto an ALREADY-DRAWN `TrajSpec` (the training pool path).  No sampling."""
        return self._reset_with_spec(key, spec_, dr_level, envelope_scale)

    def _reset_with_spec(self, key, spec_, dr_level=0.0, envelope_scale=None):
        cfg = self.cfg
        # k_traj is unused here (the reference is passed in, not drawn); the split is kept
        # at 7 so the other six keys land in exactly the slots they always did.
        k_dr, _k_traj, k_spawn, k_lh, k_wind, k_fail, k_rest = jax.random.split(key, 7)
        dr = jnp.clip(jnp.asarray(dr_level, jnp.float32), 0.0, 1.0)

        # --- domain randomisation -------------------------------------------------
        drr = D.sample_dr(k_dr, dr)
        model = D.apply_model_dr(self.mjx_model, self.idx, drr)

        # --- reference (already drawn by the caller) ------------------------------
        ref0 = traj_sample(spec_, jnp.asarray(0.0))

        # --- spawn: ON the reference start, plus the initial kick -----------------
        kx, ky, kz, katt, kvel, kr = jax.random.split(k_spawn, 6)
        if cfg.random_initial_pos:
            jxy = 0.05 + dr * 0.15
            jz = 0.03 + dr * 0.09
            pos_jit = jnp.array([
                jax.random.uniform(kx, (), minval=-jxy, maxval=jxy),
                jax.random.uniform(ky, (), minval=-jxy, maxval=jxy),
                jax.random.uniform(kz, (), minval=-jz, maxval=jz)])
        else:
            pos_jit = jnp.zeros(3)
        spawn_pos = ref0.p + pos_jit

        # --- spawn attitude = reference attitude + jitter (body frame) -------------
        base_quat = _dcm_to_quat(ref0.R)
        if cfg.random_initial_att:
            arp = 0.05 + dr * 0.12
            ay = 0.035 + dr * 0.10
            jit = jnp.array([
                jax.random.uniform(katt, (), minval=-arp, maxval=arp),
                jax.random.uniform(katt, (), minval=-arp, maxval=arp),
                jax.random.uniform(katt, (), minval=-ay, maxval=ay)])
            dq = _normalize_quat(jnp.array([1.0, 0.5 * jit[0], 0.5 * jit[1], 0.5 * jit[2]]))
            spawn_quat = _quat_mult(base_quat, dq)
        else:
            spawn_quat = base_quat

        # --- initial kick ---------------------------------------------------------
        if cfg.random_initial_vel:
            vlin = S.INIT_VEL_RANGE[0] + dr * (S.INIT_VEL_RANGE[1] - S.INIT_VEL_RANGE[0])
            vang = S.INIT_RATE_RANGE[0] + dr * (S.INIT_RATE_RANGE[1] - S.INIT_RATE_RANGE[0])
            kv, kw = jax.random.split(kvel)
            v0 = ref0.v + jax.random.uniform(kv, (3,), minval=-vlin, maxval=vlin)
            w0 = ref0.omega + jax.random.uniform(kw, (3,), minval=-vang, maxval=vang)
        else:
            v0 = ref0.v
            w0 = ref0.omega

        qpos = jnp.concatenate([spawn_pos, _normalize_quat(spawn_quat)])
        qvel = jnp.concatenate([v0, w0])
        data = self.base_data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(4))
        data = mjx.forward(model, data)

        # --- lighthouse ----------------------------------------------------------
        lh = LH.empty_state(self.lh_cfg)
        lh = LH.reset(self.lh_cfg, lh, k_lh, spawn_pos, v0, dr)

        # --- wind ----------------------------------------------------------------
        wind = WIND.reset(k_wind, self.wind_max * jnp.maximum(0.1, dr),
                          enabled=cfg.random_wind)

        # --- observation latency -------------------------------------------------
        max_lat = jnp.round(dr * S.OBS_LATENCY_MAX_STEPS).astype(jnp.int32)
        lat = jax.random.randint(k_rest, (), 0, jnp.maximum(max_lat, 0) + 1)

        # --- failure injection ---------------------------------------------------
        fail = self._failure_init(k_fail, dr, drr)

        plant = D.initial_plant_state()
        # TYPES MATTER HERE.  A `jnp.array(0.0)` with no dtype is WEAK-typed, and the env's
        # arithmetic promotes those leaves to STRONG float32 within one step.  The state
        # `reset` produces must therefore match the state `step` produces leaf-for-leaf, or
        # XLA compiles `iterate` TWICE: once for the state that came from the batched reset
        # and once for the state that came back out of the rollout.  MEASURED: that cost a
        # second ~30 s compilation at the production shape (300 EnvState leaves, 6 of them
        # weak-vs-strong).  Every scalar below is therefore given an explicit dtype.
        hover_a0 = 2.0 * drr.mass * S.GRAVITY / (S.MAX_THRUST * drr.thrust_scale) - 1.0
        init_action = jnp.array([hover_a0, 0.0, 0.0, 0.0])
        state = EnvState(
            model=model, data=data, plant=plant, lh=lh, traj=spec_,
            t=S.float_scalar(0.0), step_count=jnp.array(0, jnp.int32),
            prev_action=init_action, anchor_pos=spawn_pos,
            gru_h=jnp.zeros(ENC.head_hidden(self.gru) if self.gru is not None else 1),
            gru_z=jnp.zeros(S.Z_DIM),
            dr=drr, dr_level=dr, wind=wind,
            envelope_scale=S.float_scalar(S.ENVELOPE_SCALE_START if envelope_scale is None
                                          else envelope_scale),
            obs_latency=lat,
            obs_buf=jnp.zeros((LAT_MAX + 1, S.ACTOR_SINGLE_OBS_DIM)),
            aux_buf=jnp.zeros((LAT_MAX + 1, S.ENCODER_AUX_DIM)),
            flip_spin_veh=S.float_scalar(0.0), flip_axis=jnp.zeros(3),
            flip_axis_valid=jnp.asarray(False, jnp.bool_),
            flip_last_kind=jnp.array(-1, jnp.int32),
            flip_last_spin=S.float_scalar(0.0),
            flip_progress_err=S.float_scalar(0.0),
            has_lifted_off=jnp.asarray(False, jnp.bool_), failure=fail, rng=k_rest,
            terminated=jnp.asarray(False, jnp.bool_),
            truncated=jnp.asarray(False, jnp.bool_),
            termination=jnp.array(TERM_NONE, jnp.int32),
        )

        # first observation, then pre-fill the latency buffers with it (the baseline
        # services the oldest entry, and before `latency` steps that is the first frame)
        frame, aux, rng2 = self._actor_frame_and_aux(state)
        state = state.replace(
            rng=rng2,
            obs_buf=jnp.tile(frame, (LAT_MAX + 1, 1)),
            aux_buf=jnp.tile(aux, (LAT_MAX + 1, 1)))
        state, obs = self._observe(state)
        return state, obs

    # ==================================================================================
    # step
    # ==================================================================================
    def step(self, state: EnvState, action):
        action = jnp.clip(jnp.asarray(action, jnp.float32), -1.0, 1.0)
        # T1-A: the same per-step slew limit the baseline applies BEFORE its EMA, measured
        # against the previous APPLIED action (what actually reached the mixer).
        if S.ACTION_MAX_DELTA > 0.0:
            slew = action - state.prev_action
            action = state.prev_action + jnp.clip(slew, -S.ACTION_MAX_DELTA, S.ACTION_MAX_DELTA)
        # EMA uses the PREVIOUS APPLIED action, exactly as the baseline does
        applied = S.ACTION_EMA_ALPHA * action + (1.0 - S.ACTION_EMA_ALPHA) * state.prev_action

        throttle = 0.5 * (applied[0] + 1.0) * S.MAX_THRUST
        omega_des = jnp.array([applied[1] * S.MAX_RATE_XY,
                               applied[2] * S.MAX_RATE_XY,
                               applied[3] * S.MAX_RATE_Z])

        # wind into the model (quad_mujoco writes model.opt.wind)
        wind_world = WIND.wind_world(state.wind, state.t)
        model = state.model.replace(opt=state.model.opt.replace(wind=wind_world))

        data, plant, tel = D.step_plant(model, state.data, state.plant, state.dr,
                                        throttle, omega_des, self.accel_adr)
        t = state.t + S.SIM_DT
        rng, k_obs = jax.random.split(state.rng)

        # advance the reference, then the sensor (order matters: the frame must describe
        # the same instant as the reference it is compared against).
        #
        # The failure injector starves fixes through the model's OWN path, so the
        # blinded flag it computed on the PREVIOUS step has to be passed INTO observe.
        # (The numpy version wraps observe and sets the flag after it returns, which is
        # the same one-step delay.)
        ref = traj_sample(state.traj, t)
        need = jnp.where(state.failure.blinding, 1_000_000_000,
                         self.lh_cfg.min_stations_for_fix)
        lh, lh_out = LH.observe(self.lh_cfg, state.lh, data.qpos[0:3], data.qvel[0:3],
                                _quat_to_dcm(data.qpos[3:7]), S.SIM_DT, rng,
                                min_stations=need)

        mid = state.replace(data=data, plant=plant, lh=lh, t=t, rng=rng,
                            step_count=state.step_count + 1, prev_action=applied)
        mid, tel = self._inject_failure(mid, lh_out, tel)

        # observations (delayed o_t / aux, fresh ref_ff and privileged block)
        frame, aux, rng_next = self._actor_frame_and_aux(mid, tel, ref, k_obs)
        obs_buf = jnp.roll(mid.obs_buf, -1, axis=0).at[-1].set(frame)
        aux_buf = jnp.roll(mid.aux_buf, -1, axis=0).at[-1].set(aux)
        mid = mid.replace(obs_buf=obs_buf, aux_buf=aux_buf, rng=rng_next)

        # flip progress runs BEFORE the reward so the reward sees this step's rotation
        mid = self._update_flip_progress(mid, ref, tel)
        reward, terms = self._reward(mid, applied, ref, tel)

        term, code = self._check_termination(mid, ref)
        trunc = (mid.step_count >= self.max_steps) | (t >= mid.traj.total)
        # The penalty is applied ONCE, on the terminating step, after the order the
        # baseline uses (reward -> check_termination -> penalty).
        reward = jnp.where(term, reward - S.TERMINATION_PENALTY, reward)
        has_lifted = mid.has_lifted_off | (data.qpos[2] > 0.08)
        mid = mid.replace(terminated=term, truncated=trunc, termination=code,
                          has_lifted_off=has_lifted)
        mid, obs = self._observe(mid, tel, ref, wind_world)
        # `Observation.reward_terms` was declared and then left at ZEROS.  Fill it in: the
        # actor / critic / encoder vectors are assembled EXPLICITLY in `_observe` (see the
        # concatenations there), not sliced from this field, so nothing the policy consumes -
        # and therefore neither the observation contract nor the parity check - changes.
        # A diagnostic needs it to decompose the reward, and an all-zero field is a trap.
        obs = obs.replace(reward_terms=terms)
        return mid, obs, reward, term, trunc

    # ==================================================================================
    # observation assembly
    # ==================================================================================
    def _delayed(self, state: EnvState):
        return state.obs_buf[0], state.aux_buf[0]

    def _actor_frame_and_aux(self, state: EnvState, tel=None, ref=None, key=None):
        """
        The 29-dim actor frame, built from the LIGHTHOUSE ESTIMATE, and the 4-dim encoder
        aux.  `tel` (plant telemetry) is required outside reset.
        """
        if tel is None:
            tel = {"omega_filtered": state.data.qvel[3:6], "specific_force_b": jnp.zeros(3),
                   "v_batt_norm": jnp.asarray(1.0)}
        if key is None:
            key = state.rng
        if ref is None:
            ref = traj_sample(state.traj, state.t)

        est = state.lh.p_est
        est_v = state.lh.v_est
        quat = state.data.qpos[3:7]
        quat = jnp.where(quat[0] < 0.0, -quat, quat)
        measured_omega = tel["omega_filtered"] + state.dr.gyro_bias
        aux = jnp.concatenate([tel["specific_force_b"], jnp.array([tel["v_batt_norm"]])])

        if self.cfg.obs_noise:
            k_w, k_a, k_q, key_next = jax.random.split(key, 4)
            dr = state.dr_level
            som = (S.OBS_NOISE_OMEGA_RANGE[0]
                   + dr * (S.OBS_NOISE_OMEGA_RANGE[1] - S.OBS_NOISE_OMEGA_RANGE[0]))
            sa = jnp.radians(S.OBS_NOISE_ATT_DEG_RANGE[0]
                             + dr * (S.OBS_NOISE_ATT_DEG_RANGE[1] - S.OBS_NOISE_ATT_DEG_RANGE[0]))
            sacc = (S.OBS_NOISE_ACCEL_RANGE[0]
                    + dr * (S.OBS_NOISE_ACCEL_RANGE[1] - S.OBS_NOISE_ACCEL_RANGE[0]))
            jit = jax.random.normal(k_q, (3,)) * sa
            dq = _normalize_quat(jnp.array([1.0, 0.5 * jit[0], 0.5 * jit[1], 0.5 * jit[2]]))
            quat = _quat_mult(quat, dq)
            quat = jnp.where(quat[0] < 0.0, -quat, quat)
            measured_omega = measured_omega + jax.random.normal(k_w, (3,)) * som
            aux = aux.at[0:3].set(aux[0:3] + jax.random.normal(k_a, (3,)) * sacc)
        else:
            key_next = key

        att_err = attitude_error_rotvec(ref.R, _quat_to_dcm(quat))

        frame = jnp.zeros(S.ACTOR_SINGLE_OBS_DIM)
        if S.ANCHOR_ACTOR_XY:
            xy = est[:2] - state.anchor_pos[:2]
            frame = frame.at[S.O_POS:S.O_POS + 2].set(xy).at[S.O_POS + 2].set(est[2])
        else:
            frame = frame.at[S.O_POS:S.O_POS + 3].set(est)
        frame = (frame.at[S.O_QUAT:S.O_QUAT + 4].set(quat)
                 .at[S.O_OMEGA:S.O_OMEGA + 3].set(measured_omega)
                 .at[S.O_VELXY:S.O_VELXY + 2].set(est_v[:2])
                 .at[S.O_VELZ].set(est_v[2])
                 .at[S.O_PREV_ACTION:S.O_PREV_ACTION + 4].set(state.prev_action)
                 .at[S.O_P_ERR:S.O_P_ERR + 3].set(ref.p - est)
                 .at[S.O_V_ERR:S.O_V_ERR + 3].set(ref.v - est_v)
                 .at[S.O_ATT_ERR:S.O_ATT_ERR + 3].set(att_err)
                 .at[S.O_W_ERR:S.O_W_ERR + 3].set(ref.omega - measured_omega))
        return frame, aux, key_next

    def _privileged(self, state: EnvState, tel, ref, wind_world):
        """The 44-dim privileged truth block (order identical to the baseline)."""
        R = _quat_to_dcm(state.data.qpos[3:7])
        pos, vel = state.data.qpos[0:3], state.data.qvel[0:3]
        rel_pos_body = R.T @ (ref.p - pos)
        vel_body = R.T @ vel
        dr = state.dr
        return jnp.concatenate([
            pos, vel, state.data.qpos[3:7], state.data.qvel[3:6],
            rel_pos_body, vel_body,
            tel["motor_speed_norm"],
            wind_world,
            dr.com_offset,
            jnp.array([dr.mass - S.MASS]),
            jnp.array([dr.mass]),
            dr.motor_eff,
            jnp.array([dr.tau_up]), jnp.array([dr.tau_down]),
            jnp.array([tel["dynamic_sag"]]), jnp.array([dr.thrust_scale]),
            dr.gyro_bias,
            jnp.array([state.obs_latency]), jnp.array([state.dr_level]),
        ])

    def _observe(self, state: EnvState, tel=None, ref=None, wind_world=None):
        frame_d, aux_d = self._delayed(state)
        if ref is None:
            ref = traj_sample(state.traj, state.t)
        if wind_world is None:
            wind_world = WIND.wind_world(state.wind, state.t)
        if tel is None:
            tel = {"omega_filtered": state.data.qvel[3:6], "dynamic_sag": jnp.asarray(1.0),
                   "v_batt_norm": jnp.asarray(1.0), "motor_speed_norm": jnp.zeros(4),
                   "specific_force_b": jnp.zeros(3)}

        ref_ff = ref.a + S.GRAVITY_VEC
        if self.gru is not None:
            z, h = ENC.step(self.gru, state.gru_h, jnp.concatenate([frame_d, aux_d]))
            state = state.replace(gru_h=h, gru_z=z)
        else:
            z = jnp.zeros(0)

        actor = jnp.concatenate([frame_d, z, ref_ff]) if self.gru is not None \
            else jnp.concatenate([frame_d, ref_ff])
        priv = self._privileged(state, tel, ref, wind_world)
        critic = jnp.concatenate([frame_d, z, ref_ff, aux_d, priv])
        return state, Observation(actor_obs=actor, critic_obs=critic,
                                  frame=jnp.concatenate([frame_d, aux_d]),
                                  actor_frame=frame_d, reward_terms=jnp.zeros(4))

    # ==================================================================================
    # reward
    # ==================================================================================
    def _reward(self, state: EnvState, applied, ref, tel):
        pos = state.data.qpos[0:3]
        vel = state.data.qvel[0:3]
        R = _quat_to_dcm(state.data.qpos[3:7])
        omega = tel["omega_filtered"]

        tol = S.TRACK_TOL[ref.kind]
        p_err = jnp.linalg.norm(ref.p - pos)
        v_err = jnp.linalg.norm(ref.v - vel)
        att_err = jnp.linalg.norm(attitude_error_rotvec(ref.R, R))
        w_err = jnp.linalg.norm(ref.omega - omega)

        if S.FLIP_PROGRESS:
            att_err = jnp.where(ref.kind == IDX_FLIP,
                                jnp.maximum(att_err, state.flip_progress_err), att_err)
            # T1-C: Flip altitude objective. Penalise falling below the reference trajectory with a 0.15m buffer.
            # Tamed from 3.0 to 1.5 so policy does not prematurely abort flips to avoid vertical dipping.
            alt_drop = jnp.maximum(0.0, (ref.p[2] - 0.15) - pos[2])
            p_err = jnp.where(ref.kind == IDX_FLIP,
                              jnp.maximum(p_err, 1.5 * alt_drop), p_err)

        r_pos = S.tracking_kernel(p_err, tol[0])
        r_vel = S.tracking_kernel(v_err, tol[1])
        r_att = S.tracking_kernel(att_err, tol[2])
        r_rate = S.tracking_kernel(w_err, tol[3])
        # Relax action smoothness tolerance during flip (1.0 vs 0.33) so bang-bang torque is not penalized
        delta_act = jnp.linalg.norm(applied - state.prev_action)
        act_tol = jnp.where(ref.kind == IDX_FLIP, 1.0, S.TOL_ACTION_SMOOTH)
        r_action = jnp.exp(-((delta_act / act_tol) ** 2))

        reward = (S.TRACK_W_POS * r_pos + S.TRACK_W_VEL * r_vel + S.TRACK_W_ATT * r_att
                  + S.TRACK_W_RATE * r_rate + S.W_ACTION_SMOOTH * r_action)
        return reward, jnp.array([r_pos, r_vel, r_att, r_rate])

    # ==================================================================================
    # flip progress (FLIP_PROGRESS)
    # ==================================================================================
    def _update_flip_progress(self, state: EnvState, ref, tel):
        omega = tel["omega_filtered"]
        is_flip = ref.kind == IDX_FLIP
        spin_ref = ref.spin

        # a new flip segment begins on a kind change, or when the reference spin reverses
        restart = is_flip & ((state.flip_last_kind != IDX_FLIP)
                             | (spin_ref + 1e-9 < state.flip_last_spin))
        veh = jnp.where(restart, 0.0, state.flip_spin_veh)
        axis_valid = jnp.where(restart, jnp.asarray(False), state.flip_axis_valid)
        axis = jnp.where(restart, jnp.zeros(3), state.flip_axis)

        rate = jnp.linalg.norm(ref.omega)
        latch = is_flip & (~axis_valid) & (rate >= S.FLIP_SPIN_AXIS_MIN_RATE)
        axis = jnp.where(latch, ref.omega / jnp.maximum(rate, 1e-9), axis)
        axis_valid = axis_valid | latch

        veh = jnp.where(is_flip & axis_valid,
                        veh + jnp.dot(omega, axis) * S.SIM_DT, veh)
        err = jnp.where(is_flip, jnp.abs(spin_ref - veh), 0.0)

        return state.replace(
            flip_spin_veh=veh, flip_axis=axis, flip_axis_valid=axis_valid,
            flip_last_kind=jnp.where(is_flip, IDX_FLIP, -1),
            flip_last_spin=spin_ref, flip_progress_err=err)

    # ==================================================================================
    # termination
    # ==================================================================================
    def _check_termination(self, state: EnvState, ref):
        pos = state.data.qpos[0:3]
        finite = jnp.all(jnp.isfinite(state.data.qpos)) & jnp.all(jnp.isfinite(state.data.qvel))

        touch = pos[2] <= S.GROUND_TERMINATE_Z
        if not self.cfg.disable_contacts:
            # MJX's `data.ncon` is the ALLOCATED contact capacity (nconmax = 12 here), NOT
            # the live count -- using it made every step look like a ground crash. Unused
            # contact slots read dist == 1.0 and a live floor contact reads < 0.05, so the
            # sentinel is unambiguous.
            #
            # SKIPPED when contacts are disabled (TIP 4): `dist` is only ever written by the
            # collision phase, so with mjDSBL_CONTACT set it holds stale values and this
            # test would fire on garbage.  The z test above is then the whole ground check --
            # which is what it already dominates: the legs reach 0.013 m below the body
            # origin against a 0.03 m bar, and every geom that could touch first at a tilt
            # (arms, motor cans, LEDs) has contype=0 and generates no contact at all.
            cdist = getattr(getattr(state.data, "_impl", state.data), "contact", None)
            if cdist is not None and hasattr(cdist, "dist"):
                touch = touch | (jnp.min(cdist.dist) < 0.1)

        is_takeoff = (state.anchor_pos[2] < 0.20) | (ref.kind == S.KIND_INDEX["takeoff"])
        upright = _quat_to_dcm(state.data.qpos[3:7])[2, 2] > S.TAKEOFF_UPRIGHT_DCM22
        grace = (state.t < S.TAKEOFF_GRACE_SECONDS) & (~state.has_lifted_off)
        ground = touch & ~(is_takeoff & upright & grace)

        scale = state.envelope_scale
        eff = scale / S.ENVELOPE_EFF_DIVISOR
        p_rel = pos - ref.p
        d_xy = jnp.linalg.norm(p_rel[:2])
        dz = pos[2] - ref.p[2]

        flip_ceil = (ref.kind == IDX_FLIP) & (dz > S.FLIP_MAX_CEILING_EXCURSION * eff)
        flip_xy = (ref.kind == IDX_FLIP) & (d_xy > S.FLIP_MAX_XY_DRIFT * eff)
        # tunnel: max(0.50, 3.5 * tol_pos) * eff, for everything that is not a flip
        tol_pos = S.TRACK_TOL[ref.kind][0]
        tunnel = ((ref.kind != IDX_FLIP)
                  & (jnp.linalg.norm(p_rel) > jnp.maximum(S.TUNNEL_MIN_ERROR,
                                                          S.TUNNEL_TOL_MULT * tol_pos) * eff))
        inner_active = scale < S.ENVELOPE_FREE_THRESHOLD
        inner = inner_active & (flip_ceil | flip_xy | tunnel)

        centre = jnp.array([state.anchor_pos[0], state.anchor_pos[1], S.VOLUME_CENTER_Z])
        max_outer = jnp.where(scale >= S.ENVELOPE_FREE_THRESHOLD,
                              S.OUTER_RADIUS_FREE, S.OUTER_RADIUS_TIGHT)
        outer = jnp.dot(pos - centre, pos - centre) > max_outer ** 2

        code = jnp.where(~finite, TERM_DIVERGED,
                         jnp.where(ground, TERM_GROUND,
                                   jnp.where(flip_ceil, TERM_FLIP_CEIL,
                                             jnp.where(flip_xy, TERM_FLIP_XY,
                                                       jnp.where(tunnel & inner_active, TERM_TUNNEL,
                                                                 jnp.where(outer, TERM_OUT,
                                                                           TERM_NONE))))))
        return (code != TERM_NONE), code.astype(jnp.int32)

    # ==================================================================================
    # lighthouse failure injection
    # ==================================================================================
    def _failure_init(self, key, dr, drr):
        k1, k2 = jax.random.split(key)
        th = jax.random.uniform(k1, (), minval=0.0, maxval=2.0 * jnp.pi)
        d = jnp.array([jnp.cos(th), jnp.sin(th), 0.25 if self.cfg.lhf_z_too else 0.0])
        mode = jnp.array(self.cfg.lhf_mode, jnp.int32)
        return FailureState(
            mode=mode,
            direction=d / jnp.maximum(jnp.linalg.norm(d), 1e-9),
            offset=jnp.zeros(3), lie_v=jnp.zeros(3),
            max_lie=S.float_scalar(0.0), next_event=S.float_scalar(0.0),
            blinding=jnp.asarray(False, jnp.bool_), active=jnp.asarray(False, jnp.bool_),
            latched=jnp.asarray(False, jnp.bool_))

    def _inject_failure(self, state: EnvState, lh_out, tel):
        """
        Port of ``LighthouseFailure._corrupt``.

        It runs AFTER observe(), exactly as the numpy injector does by wrapping it, and its
        ONLY structural effect on the estimator is to publish the blinded flag that the
        NEXT step feeds into ``observe`` as ``min_stations`` (the numpy version writes
        `lh.cfg.min_stations_for_fix`, which observe next step then reads - the same
        one-step delay).

        The injection is applied to the ESTIMATE, never to the raw fix: the hardware
        failure was not a bad fix, it was a bad ESTIMATE, and nothing downstream could
        tell.  Every DERIVED channel (p_err, v_err, w_err) stays coherent with it because
        they are computed from the same estimate inside the observation builder.
        """
        f = state.failure
        t = state.t
        mode = f.mode
        on = (mode != S.LHF_NONE) & (t >= self.cfg.lhf_start_s)

        # --- runaway: an accelerating lie along a fixed heading --------------------
        lie_v = f.lie_v + f.direction * (1.0 * self.cfg.lhf_severity) * S.SIM_DT
        offset = f.offset + lie_v * S.SIM_DT

        # --- teleport: one jump per period ----------------------------------------
        phase = jnp.maximum(t - self.cfg.lhf_start_s, 0.0)
        stepped = (jnp.floor(phase / self.cfg.lhf_period_s)
                   > jnp.floor(jnp.maximum(phase - S.SIM_DT, 0.0) / self.cfg.lhf_period_s))
        tele_off = f.offset + jnp.where(
            stepped & on & (mode == S.LHF_TELEPORT),
            f.direction * (self.cfg.lhf_teleport_m * self.cfg.lhf_severity), 0.0)

        runaway = (mode == S.LHF_RUNAWAY) & on
        teleport = (mode == S.LHF_TELEPORT) & on
        new_offset = jnp.where(runaway, offset, jnp.where(teleport, tele_off, f.offset))

        # a persistent lie is SET, not nudged (no per-step gate can see it)
        p_est = jnp.where(runaway | teleport,
                          state.data.qpos[0:3] + new_offset, state.lh.p_est)
        v_est = jnp.where(runaway, lie_v, state.lh.v_est)

        # --- reception latch under motor power or blackout ------------------------
        latch_trigger = on & (mode == S.LHF_LATCH) & (lh_out["n_visible"] == 0)
        latched = f.latched | latch_trigger

        # --- blinding for the NEXT step -------------------------------------------
        in_outage = ((t - self.cfg.lhf_start_s) % self.cfg.lhf_period_s) < self.cfg.lhf_outage_s
        blinding = jnp.where(~on, jnp.asarray(False),
                             jnp.where((mode == S.LHF_LOSS) | latched, jnp.asarray(True),
                                       jnp.where(mode == S.LHF_OUTAGE, in_outage,
                                                 jnp.asarray(False))))
        mag = jnp.linalg.norm(p_est - state.data.qpos[0:3])
        f_new = f.replace(lie_v=jnp.where(runaway, lie_v, f.lie_v),
                          offset=new_offset,
                          max_lie=jnp.maximum(f.max_lie, jnp.where(on, mag, 0.0)),
                          blinding=blinding, active=on, latched=latched)
        return state.replace(lh=state.lh.replace(p_est=p_est, v_est=v_est),
                             failure=f_new), tel


# ======================================================================================
# small helpers
# ======================================================================================
def _quat_to_dcm(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])


def _dcm_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    w = jnp.sqrt(jnp.maximum(0.0, 1.0 + tr)) / 2.0
    x = jnp.sqrt(jnp.maximum(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = jnp.sqrt(jnp.maximum(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = jnp.sqrt(jnp.maximum(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = jnp.where(R[2, 1] - R[1, 2] < 0.0, -x, x)
    y = jnp.where(R[0, 2] - R[2, 0] < 0.0, -y, y)
    z = jnp.where(R[1, 0] - R[0, 1] < 0.0, -z, z)
    return _normalize_quat(jnp.array([w, x, y, z]))


def _quat_mult(a, b):
    return jnp.array([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ])


def _normalize_quat(q):
    return q / jnp.maximum(jnp.linalg.norm(q), 1e-12)


def attitude_error_rotvec(R_ref, R):
    """rotvec(R_ref^T R) -- the same convention as QuadFlipEnv._attitude_error_rotvec."""
    R_err = R_ref.T @ R
    tr = jnp.trace(R_err)
    cos_th = jnp.clip(0.5 * (tr - 1.0), -1.0, 1.0)
    th = jnp.arccos(cos_th)
    sin_th = jnp.sin(th)
    axis_unnorm = jnp.array([R_err[2, 1] - R_err[1, 2],
                             R_err[0, 2] - R_err[2, 0],
                             R_err[1, 0] - R_err[0, 1]])
    axis = axis_unnorm / jnp.maximum(2.0 * sin_th, 1e-6)
    vec_std = jnp.where(th < 1e-4, 0.5 * axis_unnorm, th * axis)

    # Near-pi branch: extract axis from symmetric matrix 0.5 * (R_err + I)
    A = 0.5 * (R_err + jnp.eye(3))
    diag = jnp.sqrt(jnp.maximum(0.0, jnp.diag(A)))
    k = jnp.argmax(diag)
    axis_pi = A[:, k] / jnp.maximum(1e-6, diag[k])
    axis_pi = axis_pi / jnp.maximum(1e-6, jnp.linalg.norm(axis_pi))

    return jnp.where(th > jnp.pi - 1e-4, th * axis_pi, vec_std)


def _no_nan(x):
    return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
