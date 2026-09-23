# Waits for the current training (ppo_15m) to finish, then runs the whole chain:
#   best-checkpoint selection -> comparison -> reward-shaping run -> selection -> comparison
#
# Run this in a SECOND PowerShell window while training is still going:
#     cd C:\Users\user\Desktop\eyfyhs\Quadcopter-flip-master
#     powershell -ExecutionPolicy Bypass -File run_overnight.ps1
#
# All output is also written to overnight_log.txt so you can read it in the morning.
# NOTE: ASCII only on purpose - PowerShell 5.1 reads .ps1 as ANSI without a BOM,
# which corrupts non-ASCII characters.

$ErrorActionPreference = "Continue"

# Without this, PowerShell captures Python's stdout using the OEM codepage and
# Greek text is mangled IRREVERSIBLY inside the log file.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$log = Join-Path $PSScriptRoot "overnight_log.txt"
Set-Location $PSScriptRoot

function Say($msg) {
    $line = "[{0:HH:mm:ss}] {1}" -f (Get-Date), $msg
    Write-Output $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

function RunStep($label, $argList) {
    Say "=== $label ==="
    $out = & py -3.11 @argList 2>&1 | Out-String
    Write-Output $out
    Add-Content -Path $log -Value $out -Encoding utf8
}

Say "Started. Waiting for current training (ppo_15m) to finish..."

# --- 1. Wait for completion -------------------------------------------------------
# IMPORTANT: do NOT wait on ppo_15m_vecnormalize.pkl. train.py's
# VecNormalizeCheckpointCallback rewrites that file at EVERY checkpoint, so it
# already exists mid-run. Only the final .zip is written after model.learn()
# returns, so that is the correct completion signal.
$finalModel = Join-Path $PSScriptRoot "ppo_15m.zip"
if (Test-Path $finalModel) {
    Say "WARNING: ppo_15m.zip already exists - is training really still running?"
    Say "Aborting so we do not start a second training on top of it."
    exit 1
}
$waited = 0
while ($true) {
    if (Test-Path $finalModel) {
        # Let the stats file be written and handles close before reading.
        Start-Sleep -Seconds 30
        Say "Training finished."
        break
    }
    Start-Sleep -Seconds 60
    $waited = $waited + 1
    if (($waited % 10) -eq 0) { Say "...waiting ($waited min)" }
    if ($waited -gt 180) {
        Say "WARNING: 3 hours with no completion. Aborting."
        exit 1
    }
}

# --- 2. Baseline: pick best checkpoint, then compare ------------------------------
RunStep "Select best checkpoint (baseline)" @(
    "select_best_checkpoint.py", "--seeds", "10", "--vary",
    "--min-steps", "5000000", "--save", "ppo_best")

RunStep "Compare PID vs PPO (baseline)" @(
    "compare_pid_ppo.py", "--model", "ppo_best", "--episodes", "20", "--vary")

# --- 3. Reward-shaping experiment (~1.3 h) ----------------------------------------
RunStep "Train with shaped reward" @(
    "train_ppo_baseline.py", "--steps", "20000000", "--workers", "4",
    "--resume-from", "ppo_15m", "--name", "ppo_shaped",
    "--clean", "--shaped", "--ent", "0.0")

# --- 4. Shaped: pick best checkpoint, then compare --------------------------------
RunStep "Select best checkpoint (shaped)" @(
    "select_best_checkpoint.py", "--seeds", "10", "--vary",
    "--min-steps", "15000000", "--save", "ppo_shaped_best")

RunStep "Compare PID vs PPO (shaped)" @(
    "compare_pid_ppo.py", "--model", "ppo_shaped_best", "--episodes", "20", "--vary")

Say "DONE. Read overnight_log.txt."
