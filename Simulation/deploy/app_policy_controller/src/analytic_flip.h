// Analytic Onboard Flip Trajectory Generator
// Evaluates closed-form Flip equations at 100 Hz based on runtime parameters.
// Provides dynamic parametric aerobatics with machine-precision parity to trajectories.py.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#define ANALYTIC_FLIP_DEFAULT_PEAK_DPS   950.0f
#define ANALYTIC_FLIP_DEFAULT_POP_PCT    0.92f
#define ANALYTIC_FLIP_DEFAULT_RATE_FRAC  0.30f
#define ANALYTIC_FLIP_DEFAULT_AXIS       0u     // 0 = pitch, 1 = roll
// Full-throttle thrust of the vehicle the reference is PLANNED for, in newtons.
//
// *** IT MUST MATCH POLICY_SIM_MAX_THRUST_N (2026-09-29). *** The whole analytic profile
// scales with it: `u = pop_pct * max_thrust / mass - g` is the pop's net acceleration, and
// BOTH the climb time `c0 = v0/u` and the altitude excursion `v0^2/(2u) + v0^2/(2g)` blow
// up as `u` shrinks. At the old 0.46 f the DEFAULT preset (950 dps / 0.92 pop) demanded a
// 1.53 m excursion and the weakest one (750 dps / 0.85) demanded 3.35 m - beyond both the
// 2.45 m flip ceiling (`g_max_z + 0.35`) and the 2.0 m cage declared by POLICY_MAX_Z, so
// ABORT_Z / failsafe pre-empted every attempt no matter how the phases were timed.
//
// Why match the sim: the POLICY was trained against the sim plant, so planning the
// reference with the same constant keeps it in distribution. The rule the simulator
// satisfies exactly (0.033*9.81/0.5396 = 0.600 N) is `max_thrust = m*g / hover_fraction`.
// MEASURED after the 20 mm motor swap: hover is 33.7% of the command range, so the REAL
// vehicle's value is ~0.96 N - i.e. it now has MORE margin than this model assumes, which
// is the safe direction. Raise `traj_flip.max_thrust` at runtime to exploit that; the
// default stays conservative so the reference stays in distribution.
#define ANALYTIC_FLIP_DEFAULT_MAX_THRUST 0.60f  // = POLICY_SIM_MAX_THRUST_N
#define ANALYTIC_FLIP_TAIL_S             1.0f   // terminal hover tail duration
#define ANALYTIC_FLIP_DT                 0.01f  // 100 Hz policy tick

typedef struct {
    float peak_dps;    // target peak body rate in deg/s (e.g. 720.0f)
    float pop_pct;     // climb/arrest thrust fraction of MAX_THRUST (0.70..0.98)
    float rate_frac;   // trapezoid ramp duration fraction of coast (0.15..0.45)
    uint8_t axis;      // 0 = pitch (world y), 1 = roll (world x)
    float mass;        // vehicle mass in kg
    float max_thrust;  // vehicle max thrust in N
    float gravity;     // gravity acceleration in m/s^2
} analytic_flip_params_t;

typedef struct {
    analytic_flip_params_t params;
    float p0[3];
    float yaw0;
    float cy;          // cos(yaw0)
    float sy;          // sin(yaw0)
    float z_shift;     // altitude shift applied during catch re-anchor
    uint32_t start_step;

    // Derived kinematics
    float omega_peak;  // rad/s
    float coast;       // s
    float v0;          // m/s
    float u;           // m/s^2 (net climb/arrest accel)
    float c0;          // s (climb duration)
    float dur_maneuver;// s (2*c0 + coast)
    float dur_total;   // s (dur_maneuver + tail)

    // Derived discrete rows (100 Hz)
    uint32_t rot_start_row;
    uint32_t catch_start_row;
    uint32_t total_rows;
} analytic_flip_state_t;

// Set default physical parameters
void analytic_flip_params_default(analytic_flip_params_t *p,
                                  float mass, float max_thrust, float gravity);

// Validate and sanitize parameters (clamps to safe physical limits)
void analytic_flip_params_sanitize(analytic_flip_params_t *p);

// Initialize an analytic flip trajectory at the live pose/heading
void analytic_flip_init(analytic_flip_state_t *s,
                        const analytic_flip_params_t *params,
                        const float p0[3],
                        float yaw0,
                        uint32_t step100);

// Sample the reference at step100: p(3), v(3), R(9, row-major), w(3, body), a(3, world)
void analytic_flip_sample(const analytic_flip_state_t *s,
                          uint32_t step100,
                          float p[3], float v[3], float R[9], float w[3], float a[3]);

// Re-anchor altitude target during catch phase (Option 1 catch)
void analytic_flip_reanchor_z(analytic_flip_state_t *s, float current_z);

// True when the entire flip + terminal hover tail has finished
bool analytic_flip_finished(const analytic_flip_state_t *s, uint32_t step100);
