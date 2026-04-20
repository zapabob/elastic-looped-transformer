# Registers the ELT-LM training pipeline as a Windows Task Scheduler entry
# that fires at user logon. The pipeline self-removes this entry on final
# completion (see scripts/pipeline_unregister.ps1).
#
# Run once from an admin PowerShell in the repo root:
#
#     powershell -ExecutionPolicy Bypass -File scripts/pipeline_register.ps1
#
# Re-running is idempotent; the task is replaced.

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$taskName = "ELT-LM-Pipeline"
$logDir   = "H:\elt_data\pipeline_logs"
$null = New-Item -ItemType Directory -Force -Path $logDir

$launcher = Join-Path $PSScriptRoot "pipeline_launcher.ps1"
if (-not (Test-Path $launcher)) {
    @"
# Auto-generated launcher — keep a rolling boot log, then run the pipeline.
`$ErrorActionPreference = "Continue"
`$repoRoot = "$repoRoot"
`$logDir   = "$logDir"
`$stamp    = Get-Date -Format "yyyyMMdd-HHmmss"
`$logFile  = Join-Path `$logDir "pipeline-`$stamp.log"

Set-Location `$repoRoot
Write-Output "=== boot `$stamp, repo=`$repoRoot ===" | Tee-Object -FilePath `$logFile

try {
    & uv run python scripts/pipeline.py *>&1 | Tee-Object -Append -FilePath `$logFile
} catch {
    "`$_" | Tee-Object -Append -FilePath `$logFile
    exit 1
}
"@ | Set-Content -Encoding UTF8 $launcher
    Write-Output "  wrote launcher: $launcher"
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$launcher`""

$trigger = New-ScheduledTaskTrigger -AtLogOn

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Days 14) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 10)

if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Output "  removed existing task: $taskName"
}

Register-ScheduledTask `
    -TaskName $taskName `
    -Description "ELT-LM end-to-end training pipeline (auto-resume, self-removes on completion)" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -RunLevel Limited | Out-Null

Write-Output "  registered: $taskName  (at logon)"
Write-Output "  launcher  : $launcher"
Write-Output "  logs      : $logDir"
Write-Output ""
Write-Output "  to verify : Get-ScheduledTask -TaskName $taskName"
Write-Output "  to remove : powershell -File scripts/pipeline_unregister.ps1"
