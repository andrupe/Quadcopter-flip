# 2026-09-25 review: antigravity-cli edits + latest radio flight

## HEADLINE REGRESSION (uncommitted antigravity edit)
`Simulation/deploy/gen_references.py::draw()` now HARDCODES the flip instead of the
seed-searched draw: `Flip(coast=0.35, yaw=0, max_rate=25.0, rate_frac=0.20, accel_frac=0.726)`.
omega_peak = 2pi/(0.35*0.8) = 22.44 rad/s = **1285.6 dps** (confirmed in
`manifests/reference_tables.json` peak_omega_dps 1285.6). OLD table was 777.5 dps / z_rel_max 1.936 m;
NEW is 1285.6 dps / z_rel_max 0.585 m.
Env action ceiling `max_rate_xy = max_rate_pitch = 20 rad/s = 1146 dps` (quad_flip_env.py:616);
measured in-distribution live band 630-917 dps. => the deployed flip is ABOVE the structural
action ceiling and ~40-100% above the trained band: structurally untrackable.

## SECOND SUSPICIOUS EDIT
`Simulation/lighthouse.py max_fix_dv 4.0 -> 8.0`. Repo measurement says worst LEGITIMATE
post-blackout velocity step = 1.50 m/s, and the gate's recorded catch of a real runaway was at
4.18 m/s. At 8.0 that runaway passes. Rationale in the edit ("DR doesn't integrate gravity ->
5-7 m/s innovation") is NOT backed by the recorded measurement.

## FLIGHT LOG logs/radio_flight_log.csv (Sep 25 17:56, 449 rows / 10.52 s)
STOCK 0-7.33 (climb to z 1.20) -> POLICY hover 7.37-8.54 (z 1.30-1.36, GOOD) -> flip 8.58-10.13.
Flip: lh_recv collapses 15->7->6->1 at 9.20 (only 25/69 flip rows had 4 stations); policy thrust
100% -> 20% -> 4% (coast) -> 100% (catch). Vehicle rotated ~2 turns instead of 1, ended pitched
~87 deg, |v| peaked **3.22 m/s**, z 1.88 -> 0.69 -> -0.08. armed 1->0 at t=10.135 on z<0.04
(ABORT_Z), while fw_armed=1/can_fly=1 and motors stayed 56k-65k with the airframe inverted
(fw_roll 180). pm_vbat sagged to **3.09 V** under 100% thrust (rest 4.08) = nearly empty pack.
`tilt` column is frozen at 75.6 for the whole log (telemetry sanity to verify).

## SOLUTION 1 DONE (2026-09-25)
`gen_references.py` draw() reverted to the seed-searched flip (first pitch match wins).
Added a trackability invariant: `check_table` now fails generation if any baked row exceeds
`ACTOR_RATE_CEILING_RADS 20 / ACTOR_YAW_CEILING_RADS 4` (pinned to the live env in main()).
Guard verified BOTH ways: the 1285.6 dps reference is REJECTED, the sampler draw PASSES.
Regenerated: flip back to seed 0 / 260 rows / 778 d/s / +1.94 m pop. check-c PASS.
Firmware rebuilt: Flash 598348 (58%), `policy controller: PRESENT`. cf2.bin 18:17.

## STILL OPEN: the flip SEQUENCER is also new and coupled to the old table
`FLIP_PHASE` state machine in controller_app.c (CLIMB->ROTATION->RECOVERY) has **0 hits at
HEAD** - a flip used to be plain `ref_play_launch(REF_KIND_FLIP)` table playback. It was
added together with the bad 1285 dps table. The row macros ARE supplied dynamically
(REF_FLIP_ROT_START_ROW 60 / CATCH 140 for the restored table, and the derivation is
self-consistent: coast = a_z<0 = rows 60..139 exactly), but two numbers are stale:
* `rot_timeout = 70u` vs the restored rotation window of **80 ticks (800 ms)** — FIXED
  2026-09-25: now `FLIP_ROT_WINDOW_TICKS (REF_FLIP_CATCH_START_ROW - REF_FLIP_ROT_START_ROW)`
  and `FLIP_ROT_TIMEOUT_TICKS (window + 20u)` = 100 ticks (1000 ms), plus a
  `_Static_assert(window > 0)` and refreshed comments. Firmware rebuilt green.

## 2026-09-25 FLIGHT #5 (18:48, 701 rows / 16.42 s) - TILT FIX WORKS, BUT I LEFT A HOLE
Pilot armed twice (STOCK interlude 9.06-13.60), re-armed at z=1.37 (higher, good), flip at
14.547 from z 1.36.
* **THE TILT EXEMPTION WORKS**: tilt reached 109 and 98 with `armed=1` and NO ABORT_TILT;
  sup_bits stayed 542 (no tumbled/crashed).
* **BUT the flip's recovery then had NO GUARD AT ALL** - tilt exempt for the whole manoeuvre AND
  the z envelope suspended in RECOVERY. The log ENDS with **`armed=1`, motors 52-65k, airframe
  inverted (tilt 98), z -2.72, vbat 2.82 V**: nothing ever disarmed, so nothing stopped the
  props (the host's motors-off latch cannot help - it needs a disarm to fire). MY BUG: I traded
  an over-eager abort for no guard. FIXED (built, warning-free, Flash 595152):
  `POLICY_GROUND_HOLD_TICKS 100u` - the z FLOOR is now ALWAYS armed and DEBOUNCED (100 ms at
  1 kHz, long enough to reject a post-blackout excursion, short enough to disarm fast), while the
  CEILING stays suspended in RECOVERY. `g_low_z_ticks` reset in policy_arm/policy_disarm.

## 2026-09-25 FLIGHT #6 (18:59, 432 rows / 9.96 s) - *** THE FLIP COMPLETED, FIRST TIME ***
Launch at z=1.22 (= SPAWN_Z, hover held 1.12-1.22, tilt<=6, polth 49-100, recv 11-15).
Flip: pop to **z 1.82 (+0.60 m above launch)** -> **inverted, tilt 158** -> recovered to
**tilt 16** -> touched down at ~9.5 s essentially LEVEL (a hard landing, not a tumble).
**sup_bits stayed 542 the whole run** (no tumbled/crashed; compare 18:48 where it went 676).
vbat min 3.33 V (vs 2.82 V on #5) - the better pack shows up in the better pop (+0.60 m vs
+0.20 m). A FULL blackout (recv=0) lasted ~0.3 s (8.18-8.47) and recovered.
STILL WRONG AT THE END: `armed=1` with props at 30-65k and **z below 0.04 for ~0.47 s**.
If the ground-guard firmware were running it would have disarmed in <=100 ms -> **the running
image almost certainly predates the 18:56 build. FLASH IT.**
MY OWN DEFECT, FIXED 19:01: the debounced floor was gated on `!g_guard.hold`, and
`state->position.z` in that test is the RAW estimate (not the held sample the policy is fed),
so a holding plausibility gate - which is exactly what a crash produces - skipped the floor.
Now `z_too_low` bypasses the `hold` gate; the ceiling keeps it (no debounce there).
NEXT-TIME LEVERS: (1) flash the latest cf2.bin; (2) **launch from ~1.5-1.8 m** - the pop reaches
only +0.60 m of the reference's +1.25 m, so the whole profile sits low and this run used ALL its
altitude (1.82 -> 0); the reference relocates onto the launch pose, so launching higher buys
ground clearance directly and 1.8 m is still inside the sphere (top 3.2 m); (3) the pack
(thrust ~ V^2, so 3.9-4.0 V under load would close most of the +0.60 -> +1.25 gap).
The log is still single-slot (`radio_flight.py:1161` opens with "w") - archive-on-launch offered.

## *** ROOT CAUSE OF EVERY FAILED FLIP: THRUST AUTHORITY IS ~73% OF THE SIM'S ***
**MASS IS 33 g AS THE SIM MODELS (user-confirmed 2026-09-25). My earlier "45-48 g effective
mass" claim in this file was WRONG and is WITHDRAWN.** I calibrated a ratio->thrust curve on the
sim's own hover point and then applied it to the REAL hover ratio, which assumes the real
ESC/motor/prop map command ratio to thrust the way the sim's plant model does. They do not.
CORRECT reading of the same log:
* weight 0.324 N (33 g); the log hovers with m1..m4 ~48000-50000 of 65535 = **74% ratio**.
* => thrust at hover = 0.324 N => **real MAX thrust ~0.437 N = 73% of the sim's 0.60 N**.
* VOLTAGE EXPLAINS ESSENTIALLY ALL OF IT: 4.04 V at rest -> 3.55 V at hover (**0.49 V sag**) and
  thrust ~ V^2 at a fixed command => (3.55/4.04)^2 = 0.772; 0.60*0.772 = 0.463 N ~= the 0.437 N
  measured. ONE coherent cause, not several.
* THE POP DEMANDS 0.540 N (accel_frac 0.9 * 0.60) => **SATURATED at every observed pack state**.
  u = min(avail,0.540)/m - g: 4.20 V -> 6.55 | 3.55 V -> 3.18 | 3.40 V -> **2.11 m/s^2**.
  Predicted climb at the coast start (t=0.478 s) = **+0.241 m** at 3.40 V; log MEASURES **+0.20 m**.
* FEASIBILITY OF THE BAKED FLIP IS DECIDED BY THE PACK (excursion vs the 2.0 m sphere budget at
  coast 0.639 s): **4.20 V -> 1.25 m FITS | 3.55 V -> 2.05 m marginal | 3.40 V -> 2.83 m FAILS**.
  So the reference is fine; the pack is the limiter.
* KNOCK-ON: real hover is a0 ~= 2*(48500/60000) - 1 = **0.62** vs the sim's trained hover trim
  `POLICY_SIM_HOVER_TRIM_A0` 0.0791 (49% ratio). The policy flies the TOP of its thrust range
  just to hold altitude, with almost no headroom for a pop, an arrest or a disturbance. The sim's
  sag DR ceiling (DYNAMIC_SAG_COEF_MAX 7%) does not cover this pack (~12% in V, 23% in thrust).
FIX ORDER: (1) a pack holding >=3.9-4.0 V under load - that alone makes the baked flip fit;
(2) if the airframe cannot, the sim must model the real authority (max thrust / sag / hover trim)
and the policy retrained, because the trained flip AND the hover trim both assume ~1.4x the
thrust the vehicle actually has.

## 2026-09-25 FLIGHT #4 (18:42, 1165 rows / 27 s) + TILT EXEMPTION FIX
My motors-off lockout WORKED: armed 1->0 at 8.424 and m=[0,0,0,0] from 8.447 for the rest of
the log (previously 50-65k forever), vbat recovering 3.38 -> 4.03.
NEW: the FIRMWARE supervisor declared the crash - sup_bits 542 (0x21E: Is armed / auto armed /
Can fly / Is flying / HL done) -> 676 (0x2A4: auto armed / **Is tumbled** / **Is crashed** /
HL done) at t=9.069. `locked` stays 0, so NO power cycle - radio_flight.arm() re-arms it.
THE ABORT CAUSED THE FALL: the flip table is 221 rows, CATCH at 112, launched 6.881 -> the
reference reached its arrest row at t=8.001 while the vehicle was still at **tilt 146**.
RECOVERY grants only 75 deg => ABORT_TILT at 8.001 => failsafe levelled it (tilt 72 by 8.129)
and ramped thrust to 0 => the last 1.5 m was a DROP. The arrest rows never ran.
Also: recv only dipped 11 -> 1 for ~0.2 s (the faster flip shortened the blackout), and at
72-77 deg tilt only cos(72) = 31% of thrust is vertical - so the last metre could not be
arrested. Post-crash the estimator ran to **-10.97 m** with recv=0 (deck tipped out of the
cones) for 18 s - post-crash garbage, not a new fault.

**FIX (built, warning-free, cf2.bin): ignore the tilt abort for the WHOLE flip manoeuvre.**
`policy_safety_check` now uses `is_flip = (g_ref.mode == MANOEUVRE && g_ref.kind == FLIP)`
instead of the phase-scoped `is_flip_rot` / 75-deg `is_flip_rec` pair (that pair is GONE;
`is_flip_rec` survives only for the z-envelope suspension). BOUNDED: `ref_hold_sync` sets
mode=HOLD/kind=NONE, so the normal limit is back the moment the table ends - which is also when
"did it finish level?" gets asked - and `policy_arm` also calls ref_hold_sync, so a re-arm can
never inherit the exemption. Verified by grep + both handback paths.
ALSO: the CLIMB->ROTATION trigger gained `|| ref_row >= REF_FLIP_ROT_START_ROW` so the phase
follows the reference even when the vehicle cannot climb (flat pack) - otherwise RECOVERY never
opens and the z suspension stays off through the rotation.
DEVIATION FROM WHAT I PROMISED: I did NOT arm the z envelope during RECOVERY. Reason: the
measured legitimate post-blackout innovation reaches 2.67 m against a 3.5 m ceiling, so arming
it would abort on exactly the estimate jumps T0-B exists to survive. Residual: during a flip's
RECOVERY the guard set is the estimator gate + the failsafe only (bounded, <=2.2 s).
STILL OPEN: launch the flip from ~1.2-1.5 m (it launched at 0.90 m) and reconsider 967 d/s
(above the 645-916 trained band) - the residual tilt at the end of the coast is where that shows.
FOLLOW-UP: the tilt-limit decision is still in controller_app.c, which cannot be host-compiled.
Moving it into policy_guard.c (where guard_host_check.py can drive it) is the in-pattern way to
get it under test.

## 2026-09-25 FLIGHT #3 (18:38, 434 rows / 10.16 s) + THE "IT WILL NOT LAND" BUG - FIXED
STOCK to z 0.78 -> `h` at 7.392 (z 0.79, recv **10-11 = 2 stations only**) -> flip at 8.270 ->
armed 1->0 at 9.836 (z -0.21). THE TAIL IS THE STORY: after the disarm the motors sat at
**50-65k (m = 64934,58181,64934,0) with tilt frozen at 82 deg for the remaining 1.6 s** and
vbat 3.18-3.25. The airframe was tipped on the ground with the props at ~65%.

**ROOT CAUSE: the app's disarmed branch is a LIVE PASSTHROUGH** (`controller_app.c:885`
`controllerPid(control, setpoint, ...)`) and `radio_flight.pilot_tick` keeps streaming its
thrust centre (65-69%, `_motors_ok` stays true because the FIRMWARE stays armed). So an
APP-INITIATED disarm (failsafe complete / abort) immediately re-applies the pilot's centre -
the T0-A ramp to 0 is undone one tick later. `x` (kill) was the only way to stop it.
Measured arithmetic: 69% -> ~44500 units, +-20000 of stock-PID differential = the 65k/24k
motor spread seen. **I mis-diagnosed this twice by reading only the start of the log - read
the TAIL of a flight log.**

**FIX (host only, NO reflash): `radio_flight.py` motors-off latch.** New `_thrust_lockout`
+ `_disarm_commanded_at`. `_on_status` latches when `armed` goes 1->0 and we did NOT command
it (>1.5 s from disarm()/kill()); `pilot_tick` then sends a REAL stop every tick instead of
the centre; `nudge_thrust` keeps it locked for delta <= 0 (a 'descend' must not spin the props
up) and RELEASES it on a climb ('w'); `arm()` and `kill()` clear it. `disarm()`/`kill()` stamp
`_disarm_commanded_at` so the normal pilot handback is NOT treated as a fault.
VERIFIED, no hardware: `/tmp/check_selfdisarm_lockout.py` (stubbed commander + synthetic status
packets) - 6 sections, ALL GREEN: normal flow unchanged; self-disarm latches + sends a real
stop + 0 thrust; stays locked through a descend nudge; 'w' releases and ramps from 0;
pilot-commanded disarm does not latch; kill clears.
STILL OPEN (needs a reflash): the app-side belt-and-braces - after a FAULT disarm the app
should command zero thrust itself rather than passing the pilot's stale setpoint through.

## 2026-09-25 FLIGHT #2 (18:34, logs/radio_flight_log.csv, 351 rows / 8.31 s) - TILT ABORT
Sequence: STOCK climb to z 0.54 -> `h` at 5.078 (z 0.57, hover held tilt 0-3) -> flip launched
6.067 -> **ABORT_TILT**, armed 1->0 at 7.608.

**WHICH LIMIT FIRED (airtight):** at 6.883 the tilt read **67 deg and did NOT abort**, so the
phase there was ROTATION (exempt) or RECOVERY (75 deg limit). The row-112 fallback puts RECOVERY
at 6.067+1.12 = ~7.19 (z 1.31->0.97, polth 100). At 7.608 tilt went 36 -> **78 > 75** -> abort.
So it is the **RECOVERY 75 deg limit**, NOT the 55 deg one that radio_flight's label claims
(abort code 1 is hard-labelled "tilt limit (>55 deg)" - WRONG for the recovery phase).

**WHY IT FAILED PHYSICALLY:**
1. **PACK IS FLAT: pm_vbat 3.01-3.40 V under load** (max 4.00). The reference's pop demands
   u = 6.554 m/s^2 = 1.67 g = 0.540 N of a 0.60 N ceiling for 0.478 s. At 100% thrust the
   vehicle climbed ~0.4 m/s for 0.6 s instead of reaching the coast's 3.13 m/s. **The pop was
   never achieved**, so the whole altitude budget was gone.
2. **Launched from z = 0.65 m**, not ~1.2 m - the flip profile is relocated onto the launch
   pose, so the pop topped out at 1.40 m and there was nothing left to arrest into.
3. `lh_recv` collapsed to **1 station at 6.093, immediately at launch**, and stayed 1-15 - the
   documented motors-kill-reception problem, so the flip was largely open-loop.
4. Result: the vehicle DID invert (tilt 146) but over-rotated and tumbled; the recovery's own
   75 deg limit then ended it. `tilt` is a COARSE STAIRCASE in the CSV (78 repeated 20+ rows) -
   the app's status packet is slower than the 50 Hz host loop; do not read it as per-row.

**MY CHANGE'S ROLE:** removing the clock jump means the reference now plays its own 1.67 g pop
instead of skipping to the coast at dz = 0.35 m. That is CORRECT (the reference is feasible in
sim / for the scripted controller) but it EXPOSES the thrust deficit - this run does not
contradict the change, it shows the airframe cannot deliver the pop on a flat pack.
NOTE: `thrust centre -> 69%` is the HOST's manual centre and is inert while the app is armed.

## 2026-09-25 FASTER FLIP: TARGET_FLIP_PEAK_DPS = 1000 (user request)
`gen_references.TARGET_FLIP_PEAK_DPS` (+ `--flip-peak-dps`, 0 = sampler's own draw).
MECHANISM: `Flip.omega_peak = 2*pi*rotations / (coast*(1-rate_frac))` is a function of the
COAST ALONE, so the generator keeps the sampler draw's whole SHAPE (axis, yaw, rate_frac,
accel_frac recovered as thrust_climb/MAX_THRUST_TOTAL) and re-derives only `coast`. A shorter
coast also shortens the manoeuvre AND flattens the excursion (v0 = g*coast/2, excursion falls
with v0^2) - so faster is also flatter and shorter.
MEASURED TRADE (rate_frac 0.437, accel_frac 0.900): target 804 -> coast 0.795 s / 1.99 s /
1.94 m; 1000 -> 0.639 / 1.60 / 1.25; 1145 -> 0.558 / 1.39 / 0.96. ALL PASS check_table.
**THE TRAINED BAND IS 645-916 dps (median 813) over 300 accepted sampler draws.** 1000 is a
~9% EXTRAPOLATION; the hard limit (actor authority 1146 dps) is not reached. The baked table
presents **967 dps** (the 100 Hz sampling misses the continuous peak by ~3%; ask for ~1035 to
land 1000 in the table).
BAKED: flip 221 rows / 2.20 s table, z 0..+1.25 m, ROT_START 48 / CATCH 112 (window 64 ticks
= 0.64 s, self-consistent with the derived coast). All checks + check-c PASS, warning-free
build, cf2.bin rebuilt. Shorter blackout window (0.64 s vs 0.80) and less altitude at risk -
both favourable for the failure mode seen in the flight log.

## SOLUTION 1 + DROP THE JUMP DONE (2026-09-25)
ALL THREE reference-clock re-bases removed from the flip sequencer (`g_ref.start_step` is now
written ONLY by ref_play_launch / ref_hold_sync; the flip path only READS it):
1. Phase 1 no longer fast-forwards to REF_FLIP_ROT_START_ROW (was the 0.97 m early coast entry).
   The measured dz/vz trigger is KEPT but now only opens the tilt exemption - deliberately early.
2. Phase 2's "adaptive gyro catch clamp" (rebase to CATCH-1) DELETED.
3. Phase 3 no longer jumps to REF_FLIP_CATCH_START_ROW; Phase 2's fallback is now
   `ref_row >= REF_FLIP_CATCH_START_ROW` (the table's own arrest), so no clock move is needed.
The FLIP_ROT_WINDOW/TIMEOUT macros I had added are superseded and removed; replaced by a
`_Static_assert(REF_FLIP_ROT_START_ROW < REF_FLIP_CATCH_START_ROW)`.
KEPT (author intent, not a clock jump): the climb-timeout abort-to-hold, the inverted latch,
the `level_exit` catch, the p_shift re-anchor and the attitude-PID reset on catch.
Build warning-free (only the pre-existing Kconfig override notice), Flash 598,364 (58%),
`policy controller: PRESENT`. cf2.bin 18:23.

## LAST REMAINING REFERENCE STEP IN THE FLIP PATH (flagged, not changed)
`g_ref.p_shift[2] = g_guard.p[2] - REF_P0[REF_KIND_FLIP][2]` on catch entry steps the reference
by (current_z - launch_z), i.e. ~+0.4 m when caught at the usual altitude. That is the author's
"Option 1" arrest aid (target current_z + 0.43 m) and it is NOT a clock jump, but it is a step
input to a policy that was trained on step-free references. Note it LOWERS arrest authority in
the falling case (target current_z+0.43 vs the table's own higher target) and RAISES it when
caught high. Decide with a flight, not by argument.

## CLIMB TRIGGER (measured, retained deliberately)
Restored table row 60 = coast start: dz +1.179 m, vz +3.85 m/s. The dz>=0.35/vz>=1.60 trigger
fires at row 25 (dz +0.205 m, vz +1.64 m/s). That is now HARMLESS (it only opens the tilt
exemption, which must be on before the body inverts) and no longer moves the reference.

## CONFIRMED GOOD in the same edit
ACTION_MAX_DELTA=0.5 mirrored (POLICY_ACTION_MAX_DELTA in exported header + controller_app.c:403);
T0-A failsafe + T0-B guard present in controller_app.c (guard code 4 = plausibility);
evaluate.py gained a CLI; radio_flight triple-sends arm/disarm/play and prints abort reasons;
refuses to arm on the ground (z<0.04).
