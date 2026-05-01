# Registers a lightweight Windows Task Scheduler progress reporter.
#
# This is separate from ELT-LM-Pipeline: it never starts training. It only
# snapshots status/metrics/GPU/disk into H:/elt_data/pipeline_state.

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [int]$IntervalMinutes = 5
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$taskName = "ELT-LM-Progress-Report"
$scriptPath = Join-Path $repoRoot "scripts\pipeline_progress_report.ps1"
$logDir = "H:\elt_data\pipeline_logs"
$null = New-Item -ItemType Directory -Force -Path $logDir

$launcher = Join-Path $logDir "pipeline_progress_report_launcher.ps1"
$launcherBody = @"
`$ErrorActionPreference = "Continue"
`$repoRoot = "$repoRoot"
`$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
`$logFile = Join-Path "$logDir" "progress-report-`$stamp.log"
Set-Location `$repoRoot
try {
    powershell -ExecutionPolicy Bypass -File "$scriptPath" *>&1 | Tee-Object -FilePath `$logFile
} catch {
    "`$_" | Tee-Object -Append -FilePath `$logFile
    exit 1
}
"@

if ($PSCmdlet.ShouldProcess($launcher, "write progress launcher")) {
    $launcherBody | Set-Content -Encoding UTF8 $launcher
}

$taskRun = "powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$launcher`""
if ($PSCmdlet.ShouldProcess($taskName, "register progress scheduled task")) {
    & schtasks.exe /Create /TN $taskName /TR $taskRun /SC MINUTE /MO $IntervalMinutes /F | Write-Output
    if ($LASTEXITCODE -ne 0) {
        throw "schtasks.exe failed with exit code $LASTEXITCODE"
    }
    Write-Output "registered: $taskName (every $IntervalMinutes minutes)"
    Write-Output "launcher  : $launcher"
}
