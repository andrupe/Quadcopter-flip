# Deployment: the trained policy on a real Crazyflie 2.1

This directory is the SIM → REAL bridge. The simulator is untouched: everything here is
additive tooling plus one out-of-tree firmware app. The plan of record (user decisions,
phases, risks) is in the repo notes and the session plan.

## Status (measured, 2026-09-14)

| Item | State |
|---|---|
| Policy export (C arrays) | ✅ verified vs torch: z 9.7e-07, action 7.8e-07 over 3000 real frames |
| Policy export (ONNX, primitive ops, state I/O) | ✅ verified vs torch: z 1.3e-06, action 1.0e-06 |
| C forward pass compiled with clang vs torch | ✅ ALL GREEN (1.2e-06 / 1.1e-06) |
| ST Edge AI backend (build) | ✅ both backends selectable; `stai_network_run` present, legacy weights provably absent |
| ST Edge AI backend (numerics) | ✅ vs torch: z 5.26e-04 / action 1.61e-04 (that IS the export's GELU cost); conversion fidelity 1.3e-06 |
| Baked manoeuvre tables (6 kinds, 2 496 rows) | ✅ every table matches `ShiftedTrajectory` to ≤ 4.4e-07 and starts/ends in hover |
| Firmware app builds into the image | ✅ `controllerOutOfTreeInit` present in the ELF, warning-free |
| flash budget (legacy backend) | 646 700 / 1 032 192 B (63%), **377 KB free** |
| Flash budget (ST backend) | 661 684 / 1 032 192 B (64%), **362 KB free** |
| RAM budget | legacy 109 096 B (83%, 21.5 KB free) · ST 111 664 B (85%, 19.4 KB free) |
| Hardware bring-up (flash, shadow, hover, manoeuvres) | ⬜ not started — needs the Crazyradio + Lighthouse |

## Toolchain

- ARM GCC (official tarball, no sudo): `/Users/apan/University/IntelligentControl/tools/arm-gnu-toolchain-14.3.rel1-darwin-arm64-arm-none-eabi/bin`
- Firmware clone: `firmware/` (gitignored), current master.
- `cflib` in `.venv` for flashing/telemetry.

## Build

```
Simulation/deploy/build_app.sh          # one command; verifies the app is in the image
```

**macOS trap worth knowing.** The firmware's own `scripts/kconfig/merge_config.sh` is
GNU-only (`readlink -m`, `sed -i`, `cp -T`). On macOS it fails on all three, prints
`merged configuration written to  (needs make)` with an EMPTY path and **exits 0** — so
`make` silently proceeds with an UNMERGED config (no `CONFIG_APP_ENABLE`, no app, no
warning; the binary grows ~3 KB instead of ~178 KB). `build_app.sh` merges `app-config`
with the tracked stdlib-only `apply_oot_config.py` instead, and then *asserts* that
`controllerOutOfTreeInit` is in the ELF — the failure cannot recur silently.

## Flash

**The flashing CLI comes from the Bitcraze client package.** `cfloader` is not part of
`cflib`, and there is no Bitcraze `cfloader` on PyPI (the package under that name is an
unrelated config-loader library — do not install it). It ships **inside the `cfclient`
wheel** (`cfloader/__main__.py`, console script `cfloader = cfloader:main`), which is also
the tool for the Lighthouse calibration, so install that once — **in a separate venv**,
because it pins `numpy<2.5` and `cflib~=0.1.33` and pulls PyQt6/vispy/pyopengl/pyzmq,
which must not disturb the pinned training environment:

```
python3 -m venv .venv-client
.venv-client/bin/pip install cfclient
.venv-client/bin/python -m cfloader            # usage; there is NO --help flag
.venv-client/bin/cfclient                           # the GUI (Lighthouse, params, logs)
```

**`cfloader` takes an ACTION as its first argument**, so `--help` is parsed as an action and
gets you `Action --help unknown!` (exit 255). Bare, it prints the usage; the actions are
`info` (leaves the target in the bootloader), `reset` (back to firmware - the recovery after
an interrupted flash) and `flash <file> [targets]`, with `-c` / `-w <uri>` choosing cold or
warm boot.

**Two traps, both measured, both with a confusing message:**

- `Failed to flash: Could not connect to bootloader` does **not** mean the URI is wrong. The
  radio link opens first, then cfloader sends reset-to-bootloader and waits 5 s for the
  drone's reply (`cflib/bootloader/cloader.py::reset_to_bootloader`); a silent drone times
  out into exactly this error. Confirm the drone answers at all with
  `cflib.crtp.scan_interfaces()` - if it only lists `usb://0`, the radio is the problem.
  `-c` (cold boot) is the way in when the app does not answer: it scans channel 110 and
  channel 0 (`_available_boot_uri = ('radio://0/110/2M/E7E7E7E7E7',
  'radio://0/0/2M/E7E7E7E7E7')`) for the nRF51 bootloader during the 10 s window after a
  power cycle, so it does not need the app's channel/address to be reachable.
- **`-w usb://0` cannot work with cflib 0.1.33.** `open_bootloader_uri()` appends
  `?safelink=0` to every URI it opens, and `crtp/usbdriver.py` matches
  `^usb://([0-9]+)$` exactly - so `get_link_driver('usb://0?safelink=0')` returns `None`
  (measured) and the bootloader is handed a null link: `'NoneType' object has no attribute
  'send_packet'`. The USB link itself is fine (a plain `usb://0` CRTP session works); it is
  only cfloader's warm-boot path that is broken. Note that even a USB-triggered flash
  finishes over the **radio** bootloader (`radio://0/0/2M/B1xxxxxxxx`), because the nRF51 is
  the one that can put the STM32 into its ROM bootloader.
- **The radio settings are STORED, not compiled in.** `hal/src/radiolink.c` boots from
  `configblockGetRadioChannel/Speed/Address()`, i.e. the config block in the board's I2C
  EEPROM, and `configblockeeprom.c` only rewrites that EEPROM if its magic/version/checksum
  is invalid. Flashing - warm *or* cold boot - leaves it alone, so a second-hand board (or
  one from a lab where every drone gets its own address) keeps the old settings. The
  fingerprint: dongle opens, `radio://0/80/2M/E7E7E7E7E7` gives `Too many packets lost`,
  `scan_interfaces()` finds nothing (it scans all channels but only at the DEFAULT address,
  so a changed address is invisible), while USB and cold-boot flashing work because the nRF51
  bootloader uses its own hard-coded channel 0/110 settings.
  `Simulation/deploy/radio_config.py` reads the block over USB and prints the URI that
  matches, or writes the factory values back (`--defaults` = channel 80, 2M, `E7E7E7E7E7`,
  trims preserved). It is the scriptable form of cfclient's *Configure 2.x*: the block is a
  `MemoryElement.TYPE_I2C` element (`radio_channel`, `radio_speed` 0/1/2 = 250K/1M/2M,
  `radio_address` on version-1 blocks, plus `roll_trim`/`pitch_trim`). **A reboot is required
  for a write to take effect** (`radiolinkInit` runs at boot). Every other tool takes the
  address with `--uri`.

**Everything runs over the Crazyradio PA** - flashing, the bench gate, the lighthouse
checks, flying - on `radio://0/80/2M/E7E7E7E7E7`, the factory address. Every tool defaults
to that URI, so `--uri` is only needed for a different address (`--scan` finds one):

```
.venv-client/bin/python -m cfloader flash Simulation/deploy/app_policy_controller/build/cf2.bin stm32-fw -w radio://0/80/2M/E7E7E7E7E7
```

One fallback exists for the day the radio refuses: the Crazyflie 2.1's micro-USB port is
wired to the STM32F405 itself (USB `0483:5740`), so a data cable gives a full CRTP link as
`usb://0`, and the ROM bootloader (`0483:df11` with `dfu-util`; hold the button while
plugging in) can always reflash a bricked board. Recovery only - not the working path.

## What needs the GUI client, and what does not

| task | tool |
|---|---|
| flash the firmware | `cfloader` (from cfclient), over the radio |
| params, logs, console, appchannel, arm/disarm, shadow capture | **cflib only** — `bench_bringup.py` |
| fly it with the policy on a switch | **cflib only** — `radio_flight.py` + `radio_gui.py` |
| Lighthouse geometry calibration | **cfclient** — its Lighthouse tab drives the whole estimation/alignment flow. cflib has the pieces (`cflib/localization/lighthouse_*`, `mem/lighthouse_memory.py`) but no runnable calibration CLI, so scripting it means reimplementing the client |
| deck firmware updates, a live param/log GUI, estimator sanity check | cfclient |

Then, with the vehicle connected, **select the app's controller**:

```
stabilizer.controller = 6      # 0=auto,1=PID,2=Mellinger,3=INDI,4=Brescianini,5=Lee,6=OutOfTree
```

(The console prints `CONTROLLER: Using OutOfTree (6) controller`.) The 2023 Bitcraze blog
says 5 — that was before the Lee controller was inserted.

## Lighthouse calibration (the step before any hover)

The base stations need no calibration - they are factory devices that broadcast. What is
estimated once per ROOM is the installation geometry, and that estimate lives in THIS
drone. A drone from somewhere else carries the previous room's geometry, which does not
fail loudly: it produces a plausible, wrong position. So gate first, verify last.

```
Simulation/deploy/lighthouse_check.py                    # report + verdict
Simulation/deploy/lighthouse_check.py --monitor 30       # live status + estimate -> CSV
Simulation/deploy/lighthouse_check.py --point 0 0 1.0    # verify at a tape-measured mark
```

It reads what the firmware actually exposes (all in the `lighthouse` group, discovered from
the TOC at runtime): params `systemType` (1=V1, 2=V2), `method` (0=CrossingBeam, 1=Sweep in
EKF), `bsAvailable`; and logs `status`, `bsReceive`, `bsActive`, `bsGeoVal`, `bsCalVal`,
`bsCalCon`, `bsCalUd` - all the bitmaps indexed by station channel.

The verdict tells you which of the three situations you are in:

| `lighthouse.status` | meaning | what to do |
|---|---|---|
| `0` | no base stations received | they are off, not v2, or out of sight - check `systemType` first |
| `1` | stations seen, geometry/calibration **missing** | **calibrate** (below) |
| `2` | base-station data reaching the estimator | verify with `--point`, then fly |

New drone: you will be in state 1. Inherited drone: you may be in state 2 with the *other
room's* geometry - `--reset-calib` (writes `lighthouse.bsCalibReset=1`) marks it for
re-estimation, and do not trust it before `--point` passes.

### The estimation itself (cfclient)

1. `cfclient` -> connect over the radio -> **Lighthouse positioning** tab.
2. Set the system type to match the stations (v2 normally) and start the geometry
   estimation. Props off; the procedure needs **motion, not flight**.
3. Carry the vehicle slowly **around the whole volume** - corners, centre, varying height -
   for a minute or so, keeping the deck's top facing the stations, so every station is seen
   from many angles. Watch `lighthouse_check.py --monitor 30` in another terminal for the
   `geo` count and a live `x/y/z`.
4. Write/save the result, then run `--point` at one or two measured marks. A **systematic
   offset or rotation** is a geometry problem, not a tuning problem - re-estimate.

Then, and only then, move on to the bench sequence.

## Flying it

**Params** (client parameter tab or `param set`):

| Param | Default | Meaning |
|---|---|---|
| `policy.shadow` | **1** | Run the pipeline + logs, command NOTHING. Keep at 1 until the shadow log has been verified. |
| `policy.arm` | 0 | Arm/disarm (same as the appchannel commands). Syncing the hold reference happens on arm. |
| `policy.thrust_scale` | 60000 | Action +1 → legacy thrust units (60000 = firmware max setpoint). Calibrate on the bench. |
| `policy.max_tilt_deg` | 55 | Auto-disarm if exceeded while armed. |

**Appchannel commands** (`cf.appchannel.send_packet(bytes([...]))` via cflib):
`0x01` arm · `0x02` disarm · `0x03` flip (shorthand) · `0x04` status · `0x05 <kind>` play
manoeuvre, `0xFF` = stop and hold.
Status packets (`0xA5` magic, 10 Hz) carry armed/mode/shadow/abort flags, the **active
manoeuvre byte**, z, tilt, thrust units, reference error, vbat, action[0].

**Disarmed == stock.** When not armed the app calls `controllerPid` with the incoming
setpoint, i.e. the pilot's radio has authority at every instant. Disarm mid-air is
instantaneous. Safety envelope while armed: tilt > `max_tilt_deg`, z outside [0.04, 3.5] m,
or non-finite state → auto-disarm with a latched reason code.

## Bench bring-up over the radio

```
.venv/bin/python Simulation/deploy/bench_bringup.py        # radio://0/80/2M/E7E7E7E7E7
```

In order it: connects - prints the console banner and the app's params (proving the policy
image is really flashed) - sets `stabilizer.controller = 6` - **refuses to continue unless
`policy.shadow == 1`** - arms and disarms over the appchannel, confirming via the `policy`
log group - then runs the SHADOW CAPTURE.

**The shadow capture IS the `SIGN_*` gate.** With the vehicle in your hand and the props
off, it walks the pilot through four motions - roll right, nose down, yaw left, lift -
logging `policy.*`, `stateEstimate.*`, `gyro.*`, `acc.*` and `pm.vbat` at 10 Hz into
`logs/bench_shadow.csv`, then compares the measured gyro signs against the `SIGN_*`
constants in `controller_app.c`:

```
roll right (right wing down)        -> gyro.x POSITIVE
nose down                           -> gyro.y POSITIVE
yaw left  (counter-clockwise above) -> gyro.z POSITIVE
```

A mirrored vehicle flies *plausibly*, which is exactly what makes a wrong sign dangerous;
this catches it in the hand, before a single motor turns. The verdict names the constant
to flip. The logic is unit-tested from both sides (a correct capture passes, an inverted
`gyro.y` is reported as `INVERTED: flip SIGN_GYRO_PITCH_TO_SIM`).

### Lighthouse: what the sim assumes, and what to install

Training randomises the installation over **2-4 stations** (4 is the deck's maximum), on a
ring around the volume at 1.5-4.0 m, elevated above it, each visible only inside a 55-80
deg half-angle cone about the deck normal (body +z), a fix needing **>=2 stations at once**,
per-station dropout up to 3%/step, and noise of 3-10 mm / 20-60 mm/s (z x2). Four stations
is therefore the best case and comfortably in-distribution.

Two consequences worth designing around:

- **A flip is a modelled BLACKOUT.** The photodiodes face up, so inverted every station is
  behind the deck and the fix disappears for the whole rotation. The policy and the encoder
  were trained with exactly that (dead reckoning on the IMU) - which is why the flip is the
  last step, not the first.
- **Mounting 2 of the 4 stations lower** (below hover height, ~0.5-0.8 m) buys partial
  coverage while inverted. That is *better* than the training assumption and costs nothing
  the policy cannot absorb.

Before any of this the receiver needs its station geometry estimated. Do that with the
vehicle in your hand and the props off (the Lighthouse geometry estimation in the client
needs motion, not flight) and confirm `stateEstimate` is sane before arming anything;
`bench_bringup.py --no-capture` checks the link and params immediately.

## Flying it from the Mac: `radio_flight.py` (+ GUI)

With a Crazyradio PA this is the real-vehicle twin of `Simulation/live_flight.py` - same
workflow, same kind of panel, but the plant is the drone and the handovers go through the
app's appchannel:

```
.venv/bin/python Simulation/deploy/radio_flight.py \
    --uri radio://0/80/2M/E7E7E7E7E7 --arm-ok --hover 50
```

| panel button | what it does |
|---|---|
| **STOCK** | appchannel `0x02`: the app is DISARMED and your sticks are streamed over CRTP - the drone is a normal Crazyflie again |
| **HOVER** | appchannel `0x01`: the policy takes the hover at that instant and holds it |
| **flip / orbit / figure8 / lissajous / slalom / waypoints** | appchannel `0x05 <kind>`: the policy flies that baked trajectory and hands back to a hold where it ended. One button per table, built from the list the drone publishes in its status (see below) |
| **HOLD** | `0x05 0xFF`: stop the running manoeuvre and hold wherever it is |
| **KILL** | `0x02` + a motors-off setpoint. Also on the pad (X / square) and on Ctrl-C |

Take-off is yours, deliberately: arming syncs the hover reference to where the vehicle IS,
so fly it up in STOCK, settle, then hand over (the app's z envelope refuses a ground arm
anyway). Thrust is HOVER-ANCHORED because a gamepad stick self-centres - the middle of the
stick is `--hover` percent and the stick spans `--thrust-span` either side. **Calibrate
`--hover`**: read the answer off the policy itself, `policy.thrust_units / 60000 * 100`,
while it hovers.

### The baked manoeuvres (what the buttons actually play)

`Simulation/deploy/gen_references.py` bakes one trained trajectory family per kind into a
100 Hz C table (`p(3)|v(3)|R(9)|omega(3)`, float32, 72 B/row):

| kind | seed | rows | duration | vs `ShiftedTrajectory` | launch v0 | launch w0 | z range | peak rate |
|---|---|---|---|---|---|---|---|---|
| flip | 0 | 223 | 2.22 s | 4.4e-07 | 0.000 m/s | 0.000 rad/s | 0.00 … +1.58 m | 778 °/s |
| orbit | 378 | 491 | 4.90 s | 1.2e-07 | 0.369 | 0.080 | −0.07 … +0.00 | 118 °/s |
| figure8 | 71 | 499 | 4.98 s | 4.8e-08 | 0.458 | 0.262 | 0.00 | 46 °/s |
| lissajous | 233 | 501 | 4.99 s | 6.0e-08 | 0.535 | 0.430 | −0.03 … +0.03 | 57 °/s |
| slalom | 297 | 448 | 4.46 s | 5.9e-08 | 0.407 | 0.443 | 0.00 | 85 °/s |
| waypoints | 258 | 334 | 3.32 s | 1.2e-07 | 0.000 | 0.000 | 0.00 … +0.82 | 33 °/s |

2 496 rows → **176 KiB of flash** (const, so RAM is unchanged), emitted as
`src/generated/reference_tables.{c,h}` (gitignored, regenerable) with a manifest and
sha256 in `manifests/reference_tables.json`.

Three properties make this safe to press in the air:

- **Relocation, not repetition.** At launch the app re-anchors the table on the pose and
  heading the vehicle has AT THAT INSTANT (`REF_YAW0` rotated out, `ShiftedTrajectory`
  rotated in), so a manoeuvre started 2 m away and 40° off heading still flies the same
  motion in the same frame it was trained in. The comparison above is the *same* helper the
  simulator uses, so there is no second implementation to disagree with.
- **It starts and ends in the hover equilibrium**, and the app hands over to a hold
  anchored where the table ended - so the handover at the end is not a step.
- **No launch transient.** The periodic families (orbit / figure8 / lissajous / slalom)
  only reach zero velocity at rest, so instead of blending, the generator *selects*: it
  scores draws by the initial state they would demand and keeps one inside the training
  initial-kick range (0.10–0.60 m/s, ≤ ¼ of the 2.5 rad/s rate tolerance). Measured over 80
  seeds, selection lands between 0.32–0.54 m/s and 0.10–0.48 rad/s - the flip and waypoints
  start at rest outright. The flip is additionally pinned to the first pitch-axis/single-
  rotation draw, i.e. the flight verified above, so no reseeding can silently change it.

The **kind list is not duplicated in the host**: `radio_flight.py` parses the `REF_KIND_*`
enum out of the generated header and publishes it in every status packet, and the panel
builds one button per entry - so the GUI cannot offer a manoeuvre the firmware does not
have, and adding a seventh table needs no host edit. Status byte 6 (`manoeuvre`) tells the
panel which table is playing now, and `0xFF` means "holding".

### The host setpoint protocol (verified against the firmware, not assumed)

The firmware's `manualDecoder` maps the generic commander setpoint as
`roll/pitch -> attitude.* (degrees, self-levelling - the default)` or
`-> attitudeRate.* (deg/s, with --rate-mode)`, `yawrate -> attitudeRate.yaw (always)`, and
`thrust -> setpoint->thrust` in the raw units where `MIN_THRUST = 1000` and
`MAX_THRUST = 60000`. Four consequences, each of which would otherwise be a surprise in the
air:

- cflib PACKS `-pitch`. Together with the app's own `SIGN_RATE_PITCH = -1` derivation (the
  legacy loop tracks `-gyro.y`), `pitch = +P` is a NOSE-DOWN command - which is why "stick
  forward" in the panel is `pitch > 0`.
- **`send_setpoint_manual(..., thrust_pct=0)` is not off.** It packs raw thrust 10001, above
  `MIN_THRUST`, so the motors idle. KILL and idle use `send_stop_setpoint()`.
- The FIRST spool-up needs a zero-thrust packet: the firmware latches `thrustLocked` after a
  `COMMANDER_PRIORITY_DISABLE` and only `thrust == 0` clears it. The script sends three on
  connect.
- **There is no firmware-side deadman.** If the host dies mid-air the vehicle keeps the last
  setpoint, so the script sends stop + disarm on every exit path (including Ctrl-C),
  slew-limits thrust, and refuses to arm without `--arm-ok`.

Stick mapping signs are unit-tested without hardware (`commands_from_pad`: stick right ->
roll +, stick forward -> pitch +, yaw right -> a NEGATIVE rate because the estimator's yaw
rate is positive to the left, throttle centre-anchored), as is the panel protocol
(`GuiServer` + `LineDecoder` round trip, both directions). `--no-motors` computes and prints
setpoints while sending only stop+disarm, for checking a mapping on the bench.

### Bring-up sequence (from the plan)

1. **Props off**: flash, confirm the console banner, `stabilizer.controller = 6`, arm/disarm
   via appchannel, verify the status packets, verify disarmed behaviour is stock.
2. **SHADOW** (`policy.shadow = 1`): fly by hand ~60 s; log the `policy` group plus
   `stateEstimate`/`gyro`/`acc`; confirm the `SIGN_*` constants in `controller_app.c`
   (roll right → +gyro.x; pitch; yaw) and the obs magnitudes. This is the gate that
   protects everything after it.
3. **Hover** (`policy.shadow = 0`): arm in hover; acceptance: 20–30 s hold, position RMS
   < 0.5 m (the sim's live point-hold measures 0.2–0.7 m), instant disarm ×5.
4. **Thrust calibration**: measure hover throttle units; compare with the sim's 0.275 N
   hover; set `policy.thrust_scale`. **This is also the mass check** (sim 28 g vs real
   ~33–40 g) — see risks.
5. **Manoeuvres** (`0x05 <kind>`): start with the flip; keep a pilot ready. Acceptance:
   recovery, min z > 0.3 m, abort codes, and the vehicle holding still where the table
   ended. Then orbit → figure8 → lissajous → slalom → waypoints in that order (the flip is
   the violent one, the rest are gentler and mostly horizontal).

## Inference backend

Two backends sit behind ONE api (`policy_reset()` / `policy_step()`, selected by
`policy_backend.h`), so `controller_app.c` is identical either way:

| | legacy (default) | stedgeai |
|---|---|---|
| code | `src/policy_net.c` | `src/policy_net_stedgeai.c` + `src/generated/stedgeai/` |
| weights | `generated/policy_weights.c` (152 KB flash) | `network_data.c` (const, flash) |
| runtime | none | ST archive, merged in by `src/Kbuild` |
| vs torch | z 1.2e-06 / action 1.1e-06 | z 5.26e-04 / action 1.61e-04 |
| flash / RAM | 646 700 B / 109 096 B | 661 684 B / 111 664 B |

```
Simulation/deploy/build_app.sh              # legacy
Simulation/deploy/build_app.sh --stedgeai    # ST Edge AI Core generated network
```

`build_app.sh` proves which backend landed: it asserts the controller symbol, then checks
that `stai_network_run` is present **and** that the legacy `POLICY_HID_W_0` weights array is
absent (an ST image that still carried the dead float weights would be the obvious silent
failure).

**The ST numbers, measured.** Conversion is exact (generated code vs the ONNX it came from:
z 1.3e-06, action 1.0e-06 — `stedgeai_host_check.py`), and end to end vs torch it is
5.26e-04 / 1.61e-04, i.e. exactly the export's own `--compat` tanh-GELU cost and nothing
more. 43 826 MAC per inference = 4.4 M MAC/s at 100 Hz.

**Verifying it yourself:**

```
Simulation/deploy/install_stedgeai.py     # copies the net in + checks shapes & constants
Simulation/deploy/stedgeai_host_check.py  # builds it for the host, diffs vs torch
```

The check compiles the SAME `policy_net_stedgeai.c` + `network.c` that fly. A wrinkle worth
knowing: `stedgeai validate --mode host` (the natural tool) **cannot finish on an
Apple-silicon Mac** — ST ships its host runtime as x86_64-only, and the Command Line Tools
no longer provide an x86_64 `xcrun`, which is what the CLI's "E103: unable to build the
shared library" means. A *native* clang cross-compiling with `-arch x86_64` links against
those libraries fine (they run under Rosetta), which is exactly what `stedgeai_host_check.py`
does. `stedgeai_validate.py` drives the real `validate` command and is kept for machines
with a full Xcode install.

**ST pack location**: `STEDGEAI_DIR` (default `/Applications/ST/STEdgeAI/4.0`) — both the
runtime headers and `Middlewares/ST/AI/Lib/GCC/ARMCortexM4/NetworkRuntime1201_CM4_GCC.a`
come from there; `install_stedgeai.py` fails loudly if it is missing.

Notes that apply to either backend:
- If the runtime lacks `Erf` (encoder GELU), `export_onnx.py --gelu tanh` costs 1.6e-04 on
  the action (~0.18 deg/s) — measured, acceptable, but not free. That is what `--compat` does.
- The session must use a STATIC arena: 19–22 KB RAM free, so target ≤16 KB (the generated
  ST model needs 1 408 B of activations).
- fp32 weights are ~152 KB; int8 would need re-validation (the flip is sensitive).

### Regenerating the ST network (done once already)

`STEdgeAI-Core` lives at `/Applications/ST/STEdgeAI/4.0` (`Utilities/mac/stedgeai`,
v4.0.1-20581). To reproduce the network from scratch:

```
Simulation/deploy/export_onnx.py --compat        # -> models/policy_step_stedgeai.onnx
S=/Applications/ST/STEdgeAI/4.0/Utilities/mac/stedgeai
$S analyze  --model Simulation/deploy/models/policy_step_stedgeai.onnx --target stm32f4
$S generate --model Simulation/deploy/models/policy_step_stedgeai.onnx --target stm32f4 \
            --output Simulation/deploy/stedgeai_out < /dev/null
Simulation/deploy/install_stedgeai.py            # copy into the app + cross-checks
```

`--compat` removes the three ops such importers reject (`Slice`, `Div`, `Erf`) with zero
functional change apart from the measured tanh-GELU cost. Its graph is
`Add Clip Concat Constant MatMul Mul Sigmoid Sub Tanh` with **split inputs**
(`o_t[1,29]`, `aux[1,4]`, `h_in[1,48]` → `action[1,4]`, `h_out[1,48]`, `z[1,16]`).

Two things about the CLI that cost time to learn:
- **`--target stm32f4` on its own.** Adding `--device STM32F405RG` makes it answer
  `E102: stm32f4 not recognized as target` while listing `stm32f4` as supported.
- **Redirect stdin** (`< /dev/null`). Both commands finish with an interactive "send
  statistics?" prompt that aborts the process (Python fatal error) if stdin closes late.
  The tool also drops `st_ai_output/` and `st_ai_ws/` into the CWD (both gitignored).

`analyze` on this model: **PASS, no unsupported ops, no warnings** — 42 C nodes, 77 arrays,
weights 153 568 B, activations 1 408 B, 43 826 MAC. The generated `LICENSE.txt` is ST's
SLA0104 agreement (the pack also carries Apache-2.0 for parts).

What to download on another machine: **STEdgeAI-Core, macOS** — the only item needed. Skip
X-CUBE-AI (NRND, superseded by STEDGEAI-CUBEAI), STM32CubeMX (we do not generate an STM32Cube
project; the network sources are injected into the Bitcraze build), NanoEdgeAIStudio (autoML
for a different problem) and the `*-DC` cloud variants. Download needs a free ST account.

## Known risks (carried from the plan)

1. **Mass/thrust mismatch** (sim 28 g / 0.60 N vs real ≈33–40 g): hover trim moves and the
   flip's vertical profile was learned in thrust units. Gate: step 4 above. Fallback:
   fine-tune with the real mass in the DR range.
2. **Inner loop differs** (sim PID reads ground-truth rates; the firmware PID reads the
   filtered gyro and outputs motor ratios): compare a rate step response; then retune
   `pid_rate_*`, port the sim PID, or retrain with the real loop in the DR.
3. **Sign constants** are the #1 silent failure mode → step 2 exists solely for that.
