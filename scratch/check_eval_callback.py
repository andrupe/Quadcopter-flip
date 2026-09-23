"""Exercise the modified DeterministicEvalCallback end to end.

WHY: the callback is a diagnostic that runs inside the trainer process, and a bug in it
either kills or silently corrupts a multi-hour run. It cannot be validated by reading it.
This drives the REAL callback against a REAL checkpoint and checks the two CSVs and the
per-term decomposition.

Deliberately small: one family, one episode, so it costs seconds. The point is that the
plumbing works, not that the numbers are good.

Run:  .venv/bin/python scratch/check_eval_callback.py
"""
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "Simulation"))

import csv                                                       # noqa: E402
import glob                                                      # noqa: E402

import train as T                                                # noqa: E402
from actor_input import load_checkpoint                          # noqa: E402
from quad_flip_env import (                                      # noqa: E402
    REWARD_CEILING_PER_STEP, TRACK_W_ATT, TRACK_W_POS, TRACK_W_RATE, TRACK_W_VEL,
)

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


MODEL = os.path.join(_ROOT, "logs", "rl_model_14000000_steps.zip")
if not os.path.isfile(MODEL):
    raise SystemExit(f"need a checkpoint to test with: {MODEL} missing")

tmp = tempfile.mkdtemp(prefix="evalcb_")
csv_path = os.path.join(tmp, "eval_metrics.csv")

print("=" * 78)
print("DeterministicEvalCallback: two sweeps + per-term decomposition")
print("=" * 78)

model = load_checkpoint(MODEL)
cb = T.DeterministicEvalCallback(
    freq_steps=0,
    episodes_per_family=1,
    families=["hover"],
    episode_seconds=T.EPISODE_SECONDS,
    encoder_path=os.path.join(_ROOT, "logs", "encoder_gru.pt"),
    csv_path=csv_path,
    robust_dr=1.0,
    robust_every=1,          # every evaluation, so one call produces both files
    verbose=1,
)
cb.model = model
cb.num_timesteps = 12_345
cb._evaluate_and_log()

robust_path = cb.robust_csv_path
check("nominal csv written", os.path.isfile(csv_path), csv_path)
check("robustness csv written", os.path.isfile(robust_path), robust_path)
check("robustness csv path derived from the nominal one",
      robust_path == os.path.join(tmp, "eval_metrics_robust.csv"), robust_path)


def read(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], [dict(zip(rows[0], r)) for r in rows[1:]]


h_nom, nom = read(csv_path)
h_rob, rob = read(robust_path)
check("nominal header has the term columns",
      all(f"err_{t}" in h_nom for t in cb.TERM_NAMES)
      and all(f"kern_{t}" in h_nom for t in cb.TERM_NAMES), ",".join(h_nom))
check("nominal header keeps the original first 4 columns",
      h_nom[:4] == ["timestep", "steps", "overall_per_step", "pct_of_ceiling"])
check("robustness header matches", h_rob == h_nom)
check("exactly one row in each", len(nom) == 1 and len(rob) == 1,
      f"{len(nom)} / {len(rob)}")

r = nom[0]
check("nominal timestep recorded", r["timestep"] == "12345", r["timestep"])
check("nominal steps > 0", int(r["steps"]) > 0, r["steps"])

overall = float(r["overall_per_step"])
kerns = {t: float(r[f"kern_{t}"]) for t in cb.TERM_NAMES}
errs = {t: float(r[f"err_{t}"]) for t in cb.TERM_NAMES}
weights = {"pos": TRACK_W_POS, "vel": TRACK_W_VEL, "att": TRACK_W_ATT, "rate": TRACK_W_RATE}

check("all per-term errors present and finite",
      all(v == v for v in errs.values()), str({k: round(v, 3) for k, v in errs.items()}))
check("each kernel value is within [0, 1]",
      all(0.0 <= v <= 1.0 for v in kerns.values()),
      str({k: round(v, 3) for k, v in kerns.items()}))

tracking = sum(weights[t] * kerns[t] for t in cb.TERM_NAMES)
residual = overall - tracking
check("tracking terms cannot exceed their weights",
      tracking <= sum(weights.values()) + 1e-6, f"{tracking:.3f}")
check("the unexplained residual is the action term (0..0.5)",
      0.0 <= residual <= 0.5 + 1e-6, f"{residual:.3f}")
check("overall is inside the ceiling",
      0.0 < overall <= REWARD_CEILING_PER_STEP, f"{overall:.3f}")

rr = rob[0]
check("robustness row carries the dr=1 timestep", rr["timestep"] == "12345")
check("robustness row leaves the term columns BLANK (not instrumented)",
      all(rr[c] == "" for c in h_rob if c.startswith(("err_", "kern_"))))
check("robustness sweep ran real steps", int(rr["steps"]) > 0, rr["steps"])
print(f"\n  nominal    {overall:.3f}/step = {float(r['pct_of_ceiling']):.1f}%")
print(f"  dr=1       {float(rr['overall_per_step']):.3f}/step = "
      f"{float(rr['pct_of_ceiling']):.1f}%")

# --- CSV schema guard -----------------------------------------------------------
# logs/eval_metrics.csv from the 2026-09-16 run has 13 columns; this callback writes 21.
# Appending would produce a file whose header and rows disagree - silently, hours into a run.
print("\n" + "=" * 78)
print("CSV schema guard: an older, NARROWER file must be rotated, not appended to")
print("=" * 78)
old_path = os.path.join(tmp, "legacy_eval_metrics.csv")
old_header = "timestep,steps,overall_per_step,pct_of_ceiling,hover"
with open(old_path, "w", encoding="utf-8") as f:
    f.write(old_header + "\n")
    f.write("512000,1268,3.3089,45.33,2.5285\n")

cb2 = T.DeterministicEvalCallback(
    csv_path=old_path, families=["hover"], episodes_per_family=1, verbose=0,
)
cb2._append_csv(old_path, 999, 10, 3.0, 41.0, {"hover": 3.0}, None)

rotated = glob.glob(os.path.join(tmp, "legacy_eval_metrics.*.legacy.csv"))
check("old file was rotated", len(rotated) == 1,
      str([os.path.basename(r) for r in rotated]))
if rotated:
    with open(rotated[0], encoding="utf-8") as fh:
        kept = fh.readline().strip()
    check("rotated file kept the ORIGINAL narrow header", kept == old_header, kept)

h_new, rows_new = read(old_path)
# Derived, not hardcoded: 4 fixed + one per family + err_*/kern_* for each term. With the
# full 9-family default this is 21 columns, which is exactly what the 2026-09-16 file (13)
# would have been appended to.
expect_cols = 4 + len(cb2.families) + 2 * len(cb2.TERM_NAMES)
check("replacement file has the full new header", len(h_new) == expect_cols,
      f"{len(h_new)} cols (expected {expect_cols})")
check("the wide header really is wider than the old one", len(h_new) > 5,
      f"{len(h_new)} > 5")
check("replacement file has exactly one data row", len(rows_new) == 1)
check("the fresh row is correctly aligned",
      rows_new[0]["timestep"] == "999" and rows_new[0]["hover"] == "3.0000",
      str({k: rows_new[0][k] for k in ("timestep", "steps", "hover", "err_pos")}))

# Appending to a file with the CORRECT header must still append.
cb2._append_csv(old_path, 1000, 11, 3.1, 42.0, {"hover": 3.1}, None)
h_again, rows_again = read(old_path)
check("a matching header is appended to, not rotated",
      len(rows_again) == 2 and h_again == h_new, f"{len(rows_again)} rows")

print("\n" + "=" * 78)
if FAILS:
    print(f"FAILURES ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("ALL CHECKS PASSED")
