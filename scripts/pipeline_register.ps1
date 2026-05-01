# Registers the ELT-LM resumable long-run pipeline as a Windows Task Scheduler
# entry. It starts at user logon and also ticks every 5 minutes so interrupted
# work resumes after power loss or reboot.
#
# Run from the project root:
#
#     powershell -ExecutionPolicy Bypass -File scripts/pipeline_register.ps1
#
# Dry-run:
#
#     powershell -ExecutionPolicy Bypass -File scripts/pipeline_register.ps1 -WhatIf

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$StartLongTrain,
    [switch]$WriteLauncherOnly,
    [ValidateSet("full", "posttrain-grpo", "replay-refresh", "side-lora", "v1-pretrain-posttrain", "synthetic-v1-pretrain-posttrain")]
    [string]$Profile = "full"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$taskName = "ELT-LM-Pipeline"
$logDir   = "H:\elt_data\pipeline_logs"
$cacheRoot = "H:\elt_data\cache"
$tempDir = Join-Path $cacheRoot "tmp"
$null = New-Item -ItemType Directory -Force -Path $logDir
$null = New-Item -ItemType Directory -Force -Path $tempDir
$null = New-Item -ItemType Directory -Force -Path (Join-Path $cacheRoot "uv")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $cacheRoot "uv\python")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $cacheRoot "uv\tools")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $cacheRoot "pip")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $cacheRoot "hf")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $cacheRoot "hf\datasets")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $cacheRoot "torch")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $cacheRoot "triton")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $cacheRoot "torchinductor")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $cacheRoot "cuda")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $cacheRoot "numba")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $cacheRoot "matplotlib")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $cacheRoot "xdg")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $cacheRoot "pycache")

$launcher = Join-Path $logDir "pipeline_launcher.ps1"
$startLongTrainEnv = if ($StartLongTrain) { "1" } else { "" }
$launcherBody = @"
# Auto-generated launcher. Keep a rolling scheduler log, then run the pipeline.
`$ErrorActionPreference = "Continue"
`$repoRoot = "$repoRoot"
`$logDir   = "$logDir"
`$cacheRoot = "$cacheRoot"
`$tempDir = "$tempDir"
`$env:ELT_PIPELINE_START_LONG = "$startLongTrainEnv"
`$env:ELT_PIPELINE_PROFILE = "$Profile"
`$env:TMP = `$tempDir
`$env:TEMP = `$tempDir
`$env:TMPDIR = `$tempDir
`$env:UV_CACHE_DIR = Join-Path `$cacheRoot "uv"
`$env:UV_PYTHON_INSTALL_DIR = Join-Path `$env:UV_CACHE_DIR "python"
`$env:UV_TOOL_DIR = Join-Path `$env:UV_CACHE_DIR "tools"
`$env:PIP_CACHE_DIR = Join-Path `$cacheRoot "pip"
`$env:HF_HOME = Join-Path `$cacheRoot "hf"
`$env:HF_HUB_CACHE = Join-Path `$env:HF_HOME "hub"
`$env:HF_DATASETS_CACHE = Join-Path `$env:HF_HOME "datasets"
`$env:TRANSFORMERS_CACHE = Join-Path `$env:HF_HOME "transformers"
`$env:TORCH_HOME = Join-Path `$cacheRoot "torch"
`$env:XDG_CACHE_HOME = Join-Path `$cacheRoot "xdg"
`$env:TRITON_CACHE_DIR = Join-Path `$cacheRoot "triton"
`$env:TORCHINDUCTOR_CACHE_DIR = Join-Path `$cacheRoot "torchinductor"
`$env:CUDA_CACHE_PATH = Join-Path `$cacheRoot "cuda"
`$env:NUMBA_CACHE_DIR = Join-Path `$cacheRoot "numba"
`$env:MPLCONFIGDIR = Join-Path `$cacheRoot "matplotlib"
`$env:PYTHONPYCACHEPREFIX = Join-Path `$cacheRoot "pycache"
foreach (`$path in @(
    `$env:TMP, `$env:UV_CACHE_DIR, `$env:UV_PYTHON_INSTALL_DIR, `$env:UV_TOOL_DIR,
    `$env:PIP_CACHE_DIR, `$env:HF_HOME, `$env:HF_HUB_CACHE, `$env:HF_DATASETS_CACHE,
    `$env:TRANSFORMERS_CACHE, `$env:TORCH_HOME,
    `$env:XDG_CACHE_HOME, `$env:TRITON_CACHE_DIR, `$env:TORCHINDUCTOR_CACHE_DIR,
    `$env:CUDA_CACHE_PATH, `$env:NUMBA_CACHE_DIR, `$env:MPLCONFIGDIR, `$env:PYTHONPYCACHEPREFIX
)) {
    `$null = New-Item -ItemType Directory -Force -Path `$path
}
`$stamp    = Get-Date -Format "yyyyMMdd-HHmmss"
`$logFile  = Join-Path `$logDir "pipeline-`$stamp.log"
`$pipelineArgs = @("--profile", "$Profile")
if (`$env:ELT_PIPELINE_START_LONG -ne "1") {
    `$pipelineArgs += "--no-start-long-train"
}

Set-Location `$repoRoot
Write-Output "=== tick `$stamp, repo=`$repoRoot ===" | Tee-Object -FilePath `$logFile
Write-Output "pipeline args: `$(`$pipelineArgs -join ' ')" | Tee-Object -Append -FilePath `$logFile
Write-Output "runtime cache: `$cacheRoot" | Tee-Object -Append -FilePath `$logFile

try {
    & uv run --no-sync python scripts/pipeline.py @pipelineArgs *>&1 | Tee-Object -Append -FilePath `$logFile
} catch {
    "`$_" | Tee-Object -Append -FilePath `$logFile
    exit 1
}
"@

if ($PSCmdlet.ShouldProcess($launcher, "write launcher")) {
    $launcherBody | Set-Content -Encoding UTF8 $launcher
    Write-Output "  wrote launcher: $launcher"
} else {
    Write-Output "  would write launcher: $launcher"
}

if ($WriteLauncherOnly) {
    Write-Output "  launcher-only mode: scheduled task registration left unchanged."
    Write-Output "  launcher  : $launcher"
    Write-Output "  cache     : $cacheRoot"
    return
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$launcher`""

$triggerLogon = New-ScheduledTaskTrigger -AtLogOn
$triggerCron = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Days 14) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 10)

if ($PSCmdlet.ShouldProcess($taskName, "register scheduled task")) {
    if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Output "  removed existing task: $taskName"
    }

    try {
        Register-ScheduledTask `
            -TaskName $taskName `
            -Description "ELT-LM resumable long-run pipeline (5-minute tick, checkpoint-aware)" `
            -Action $action `
            -Trigger @($triggerLogon, $triggerCron) `
            -Settings $settings `
            -RunLevel Limited | Out-Null

        Write-Output "  registered: $taskName  (at logon + every 5 minutes)"
    } catch {
        Write-Output "  Register-ScheduledTask failed: $($_.Exception.Message)"
        Write-Output "  falling back to schtasks.exe XML registration"
        $startBoundary = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
        $taskXml = Join-Path $logDir "pipeline_task.xml"
        @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>ELT-LM resumable long-run pipeline (5-minute tick, checkpoint-aware)</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
    <CalendarTrigger>
      <StartBoundary>$startBoundary</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
      <Repetition>
        <Interval>PT5M</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>P14D</ExecutionTimeLimit>
    <Priority>7</Priority>
    <RestartOnFailure>
      <Interval>PT10M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>powershell.exe</Command>
      <Arguments>-ExecutionPolicy Bypass -WindowStyle Hidden -File "$launcher"</Arguments>
    </Exec>
  </Actions>
</Task>
"@ | Set-Content -Encoding Unicode $taskXml
        & schtasks.exe /Create /TN $taskName /XML $taskXml /F | Write-Output
        if ($LASTEXITCODE -ne 0) {
            Write-Output "  schtasks XML failed with exit code $LASTEXITCODE"
            Write-Output "  falling back to simple 5-minute schtasks trigger"
            $taskRun = "powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File $launcher"
            & schtasks.exe /Create /TN $taskName /TR $taskRun /SC MINUTE /MO 5 /F | Write-Output
            if ($LASTEXITCODE -ne 0) {
                throw "schtasks.exe simple fallback failed with exit code $LASTEXITCODE"
            }
            Write-Output "  registered: $taskName  (every 5 minutes via schtasks.exe simple fallback)"
            Write-Output "  note      : battery/logon settings require elevated Task Scheduler registration."
            if (-not $StartLongTrain) {
                Write-Output "  safe mode : long train stages are deferred; pass -StartLongTrain after dry-run validation."
            }
            return
        }
        Write-Output "  registered: $taskName  (at logon + every 5 minutes via schtasks.exe XML)"
        Write-Output "  task xml  : $taskXml"
    }
} else {
    Write-Output "  would register: $taskName (at logon + every 5 minutes)"
}

Write-Output "  launcher  : $launcher"
Write-Output "  logs      : $logDir"
Write-Output ""
Write-Output "  to verify : Get-ScheduledTask -TaskName $taskName"
Write-Output "  to remove : powershell -File scripts/pipeline_unregister.ps1"
if (-not $StartLongTrain) {
    Write-Output "  safe mode : long train stages are deferred; pass -StartLongTrain after dry-run validation."
}
