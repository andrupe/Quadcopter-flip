# QUICKSTART - the real Crazyflie, over the Crazyradio PA

Copy-paste order. **Everything runs over the radio** (`radio://0/80/2M/E7E7E7E7E7`); USB is
recovery-only. Details and reasoning are in `README.md`; this page is just the sequence.

```
cd /Users/apan/University/IntelligentControl/Quadcopter-flip
```

Two venvs, on purpose:

| venv | what runs in it |
|---|---|
| `.venv` | everything we wrote (cflib 0.1.33, torch) - bench gate, lighthouse check, flight. **Already set up.** |
| `.venv-client` | `cfclient` + `cfloader` (the Bitcraze client; pins numpy<2.5 and pulls PyQt6). **Not created yet - do step 0.** |

---

## 0. One-time: the Bitcraze client venv (needed to flash and to calibrate)

```
python3 -m venv .venv-client
.venv-client/bin/pip install -U pip
.venv-client/bin/pip install cfclient
.venv-client/bin/python -m cfloader            # NO --help: it takes an ACTION as argv[1]
```

`.venv-client/bin/python -m cfloader` with no arguments prints the usage. Actions are
`info` (leave the target in the bootloader), `reset` (back to firmware) and
`flash <file> [targets]`; CRTP options are `-c` (cold boot, default) and
`-w <uri>` (warm boot). A typo in the action - including `--help` - prints
`Action <x> unknown!` and exits 255.

It must stay separate: it pins `numpy<2.5` and pulls PyQt6/vispy, which would disturb the
pinned training environment.

---

## 1. Build the firmware

```
./Simulation/deploy/build_app.sh              # legacy C net (default, the verified one)
./Simulation/deploy/build_app.sh --stedgeai   # ST Edge AI generated network
```

Both must end with `policy controller: PRESENT in the image`. Current cost:
**legacy 646 700 B flash / 109 096 B RAM**, ST 661 684 B / 111 664 B (the six manoeuvre
tables are 176 KiB of that, in flash, so RAM is unchanged).

## 2. Flash it

```
.venv-client/bin/python -m cfloader flash \
    Simulation/deploy/app_policy_controller/build/cf2.bin stm32-fw \
    -w radio://0/80/2M/E7E7E7E7E7
```

`cfloader` ships inside the `cfclient` package - it is **not** in cflib and the PyPI package
called `cfloader` is unrelated. Then `stabilizer.controller = 6` selects the app (step 4
does it for you).

**If the flash is refused, read the message - they mean different things:**

| message | what it means | what to do |
|---|---|---|
| `Could not connect to bootloader` | the radio link opened, but the drone never answered the reset-to-bootloader request (cfloader waits 5 s). Nothing wrong with the URI - **the drone is not answering on the air at all** | check the drone is powered, then use **cold boot**: `-c` and power-cycle the drone inside the 10 s window (see below) |
| `'NoneType' object has no attribute 'send_packet'` | you flashed with `-w usb://0`. cfloader appends `?safelink=0` to every URI and the USB driver only accepts `^usb://([0-9]+)$`, so no driver matches | use the radio (`-w radio://0/80/2M/E7E7E7E7E7`) or cold boot; the USB warm path is broken in cflib 0.1.33 |
| `Action <x> unknown!` | `cfloader` takes an ACTION as its first argument | bare `cfloader` prints the usage |

**Cold boot does not need the app to answer.** `cfloader ... -c` prints "Restart the Crazyflie
you want to bootload in the next 10 seconds", then scans **channel 110 and channel 0** at the
default address for the nRF51 bootloader - which is what listens on a fresh power-up. So it
works even when the app's channel/address is unknown, or the app is wedged:

```
.venv-client/bin/python -m cfloader flash \
    Simulation/deploy/app_policy_controller/build/cf2.bin stm32-fw -c
# power-cycle the drone (battery off/on, or unplug/replug) INSIDE the 10 s window
```

## 3. Lighthouse geometry - the gate before anything is armed

```
.venv/bin/python Simulation/deploy/lighthouse_check.py                 # verdict
.venv/bin/python Simulation/deploy/lighthouse_check.py --monitor 30    # live x/y/z -> CSV
```

* `status 2` + full geometry -> verify with `--point X Y Z` at a **tape-measured** mark.
  A systematic offset/rotation is a geometry fault, not a tuning fault.
* `status 1`, or a drone that came from another room -> `--reset-calib`, then estimate:
  `.venv-client/bin/cfclient` -> connect -> **Lighthouse positioning** tab ->
  estimate. **Props off, in your hand**: it needs motion, not flight. Carry it slowly around
  the whole volume (corners, centre, two heights) for a minute; watch
  `lighthouse_check.py --monitor 30` meanwhile.
* Four base stations is the best case and matches training. Mounting two of them low
  (~0.5-0.8 m) buys partial coverage while inverted - useful for the flip.

## 4. Props-off bench gate (your hand, no props)

```
.venv/bin/python Simulation/deploy/bench_bringup.py            # --no-capture for a quick link test
```

It prints the app banner + `policy.*` params (proof the image is flashed), sets
`stabilizer.controller = 6`, **refuses to continue unless `policy.shadow == 1`**, arms and
disarms over the appchannel, then walks you through the shadow capture: roll right, nose
down, yaw left, lift. It logs `logs/bench_shadow.csv` and checks the `SIGN_*` contract.

* Verdict `PASS` -> continue.
* `INVERTED: flip SIGN_GYRO_PITCH_TO_SIM` (or roll/yaw) -> edit the constant at the top of
  `Simulation/deploy/app_policy_controller/src/controller_app.c`, rebuild, reflash, re-run.

Optional, still props off: rehearse the mapping with the motors silent -
`.venv/bin/python Simulation/deploy/radio_flight.py --no-motors` prints the setpoints it
would send and only ever sends stop+disarm.

## 5. Fly it

```
.venv/bin/python Simulation/deploy/radio_flight.py \
    --arm-ok --hover 50 --set policy.shadow=0
```

`--set policy.shadow=0` is what turns the app from "compute and log" into "compute and fly".
The panel opens by itself (`radio_gui.py` in its own process; `--no-gui` to suppress).

Flight order:

1. **STOCK** with the props on: hover it by hand to ~1.2 m. Take-off is deliberately yours -
   arming syncs the hover reference to where the vehicle is, and the app's z envelope refuses
   a ground arm.
2. Settle, then **HOVER** (panel button, or Y / triangle on the pad).
3. Calibrate thrust: watch **policy thrt** on the panel. That number IS your hover, in %.
   Use the `-5% / +5%` buttons (they retrim the stick centre live) or restart with
   `--hover <that number>`. The gamepad stick is centred on it, so releasing the stick holds
   altitude instead of dropping.
4. **flip / orbit / figure8 / lissajous / slalom / waypoints** - one button each; it plays
   the trajectory relocated onto the current pose and heading, then holds where it ended.
   **HOLD** stops mid-manoeuvre and holds there.
5. **STOCK** takes it back anytime; **KILL** (panel, X / square, or Ctrl-C) is a motors-off
   disarm. Also `--max-thrust 70` is a hard ceiling, `--thrust-span` sets the stick range.

Pad: Y / triangle = HOVER, B / circle = STOCK, X / square = KILL, top-right shoulder = FLIP.

Other params you can write the same way: `--set policy.thrust_scale=55000`,
`--set policy.max_tilt_deg=45`, `--set policy.arm=1`.

## 6. What to read afterwards

| file | what it is |
|---|---|
| `logs/bench_shadow.csv` | the sign-verification capture (props off) |
| `logs/lighthouse_log.csv` | `--monitor` samples: status, geometry count, x/y/z |
| `logs/radio_flight_log.csv` | the flight: mode, armed, sticks, z, tilt, policy thrust, vbat |
| `Simulation/deploy/manifests/*.json` | hashes/derivation of every baked artifact |

## 7. Limits to keep in mind

* **No firmware-side deadman.** If the host dies mid-air the vehicle keeps the last setpoint.
  Every exit path in the scripts sends stop+disarm, and KILL is on the pad and the panel.
* Envelope while armed: z outside [0.04, 3.5] m, tilt > `policy.max_tilt_deg` (55 default),
  or a non-finite state -> auto-disarm with a latched reason (`app abort` on the panel).
* **A flip is a modelled Lighthouse blackout** (the deck faces away from every station while
  inverted); the policy was trained for it, but that is why the flip is the last step, not
  the first.
* Props off for the first run of every new mapping.

## 8. If something looks wrong

| symptom | cause |
|---|---|
| HOVER arms but nothing changes | `policy.shadow` is still 1 (the panel says `shadow: 1`) |
| Panel never connects | it is waiting for `radio_flight` - start that first, or check the port |
| `No driver found or malformed URI` | the script forgot `cflib.crtp.init_drivers()` - fixed, but any new hardware script needs it (CLASSES is empty until then) |
| `Too many packets lost` at the default URI | the drone is off, **or** its stored radio settings are not the factory ones - see below |
| `cfloader: command not found` | it comes from `cfclient`, i.e. run it with `.venv-client/bin/python -m cfloader` |
| `no baked manoeuvre called 'x'` | regenerate + reflash after editing `gen_references.py` |
| motors idle but won't spin | a stale thrust lock; the script sends three zero-thrust packets on connect |
| estimate drifts / jumps | Lighthouse: run step 3 again, then `--point` |

### "Too many packets lost" and the stored radio settings

The channel / datarate / address the drone uses come from the **config block in its I2C
EEPROM**, not from the firmware image: `firmware/src/hal/src/radiolink.c` calls
`configblockGetRadioChannel/Speed/Address()` at boot, and `configblockeeprom.c` only rewrites
that EEPROM when its magic/version/checksum is invalid. **Flashing does not change it**, so a
second-hand drone (or one from a lab that gave every drone its own address) keeps the old
settings. The signature is distinctive:

* the dongle opens but `radio://0/80/2M/E7E7E7E7E7` gives `Too many packets lost`;
* `scan_interfaces()` finds nothing - it scans every channel but only at the **default
  address**, so a changed address is invisible to it;
* USB works, and cold-boot flashing works, because the nRF51 bootloader uses its own
  hard-coded settings (channel 0 / 110) and ignores the config block.

Read the drone's real settings over the cable, or put it back on the factory values:

```
.venv/bin/python Simulation/deploy/radio_config.py                 # read + prints the URI to use
.venv/bin/python Simulation/deploy/radio_config.py --defaults      # 80 / 2M / E7E7E7E7E7
# then POWER-CYCLE the drone: the radio settings are applied at boot
```

It is the scriptable form of cfclient's *Configure 2.x* (the config block is a
`MemoryElement.TYPE_I2C` element) and it preserves the roll/pitch trims. Any other tool then
takes the address with `--uri`, e.g.
`bench_bringup.py --uri radio://0/80/2M/<ADDRESS>`.

Better: the read **remembers** the URI in `logs/drone_uri.txt` (gitignored), and
`bench_bringup.py`, `lighthouse_check.py` and `radio_flight.py` all resolve `--uri` through
`drone_link.py`, so from then on the plain commands above just work - `--uri` stays an
override. Two cflib quirks are handled inside `radio_config.py`: `I2CElement.update()` is a
silent no-op if a previous update left its callback set (it retries), and the config block is
only read after the connection setup has finished walking the memory TOC (it waits).

## 9. Changing the manoeuvres

`Simulation/deploy/gen_references.py` selects a seed per family and bakes
`generated/reference_tables.{c,h}` (100 Hz, 72 B/row) plus a manifest. Edit the budgets
there, re-run it, rebuild, reflash - the panel picks up new/changed kinds automatically
(it reads the enum out of the generated header), so no host edit is needed. `reference.h`
documents the row layout and the relocation contract.
