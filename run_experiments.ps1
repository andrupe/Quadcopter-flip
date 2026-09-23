# Runs the two remaining experiments back to back, then evaluates everything.
#
#   1. CONTROL ABLATION (~1.3h): baseline reward, 15M -> 20M.
#      Same as the shaped run in every way EXCEPT the reward, so it isolates
#      whether reward shaping or the extra 5M steps caused the improvement.
#
#   2. ADR RUN (~2h): domain randomization ramp, 20M -> 28M, from ppo_shaped.
#      Targets the measured robustness gap (PPO 5% vs PID 20% at dr_level=1.0).
#
# Usage:
#     cd C:\Users\user\Desktop\eyfyhs\Quadcopter-flip-master
#     powershell -ExecutionPolicy Bypass -File run_experiments.ps1
#
# Output also goes to experiments_log.txt.
# NOTE: ASCII only - PowerShell 5.1 reads .ps1 as ANSI without a BOM.

$ErrorActionPreference = "Continue"

# Without this, PowerShell captures Python's stdout using the OEM codepage and
# Greek text is mangled IRREVERSIBLY inside the log file.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$log = Join-Path $PSScriptRoot "experiments_log.txt"
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

# Guard: do not start if a training is already running.
$busy = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='py.exe'" |
        Where-Object { $_.CommandLine -like '*train_ppo_baseline*' }
if ($busy) {
    Say "A training run is already active. Aborting so they do not compete for CPU."
    exit 1
}

Say "Starting. Total estimated time: ~3.5 hours."

# --- EXPERIMENT 1: control ablation (baseline reward, same 5M steps) --------------
RunStep "Control ablation: baseline reward 15M -> 20M" @(
    "train_ppo_baseline.py", "--steps", "20000000", "--workers", "4",
    "--resume-from", "ppo_15m", "--name", "ppo_control",
    "--clean", "--ent", "0.0")

RunStep "Select best checkpoint (control)" @(
    "select_best_checkpoint.py", "--logs", "logs\ppo_control",
    "--seeds", "10", "--vary", "--save", "ppo_control_best")

RunStep "Compare PID vs control-ablation PPO" @(
    "compare_pid_ppo.py", "--model", "ppo_control_best", "--episodes", "20", "--vary")

# --- EXPERIMENT 2: ADR ------------------------------------------------------------
# NOTE: deliberately NO --clean here. --clean disables observation noise and wind
# entirely, which would defeat the whole point of domain randomization.
RunStep "ADR run: dr_level 0 -> 1 over 20M -> 28M" @(
    "train_ppo_baseline.py", "--steps", "28000000", "--workers", "4",
    "--resume-from", "ppo_shaped", "--name", "ppo_adr",
    "--shaped", "--ent", "0.0",
    "--dr", "--dr-start", "20000000", "--dr-end", "28000000")

RunStep "Select best checkpoint (ADR)" @(
    "select_best_checkpoint.py", "--logs", "logs\ppo_adr",
    "--seeds", "10", "--vary", "--save", "ppo_adr_best")

# --- FINAL EVALUATIONS ------------------------------------------------------------
RunStep "ADR model on CLEAN benchmark" @(
    "compare_pid_ppo.py", "--model", "ppo_adr_best", "--episodes", "20", "--vary")

RunStep "ADR model under FULL randomization (the decisive test)" @(
    "compare_pid_ppo.py", "--model", "ppo_adr_best", "--episodes", "20", "--dr", "1.0")

RunStep "Shaped model under FULL randomization (for reference)" @(
    "compare_pid_ppo.py", "--model", "ppo_shaped_best", "--episodes", "20", "--dr", "1.0")

Say "DONE. Read experiments_log.txt."
