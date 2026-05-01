# Writes a compact ELT pipeline progress snapshot for cron-style monitoring.
#
# The app heartbeat can read these files, and the Windows scheduled task keeps
# recording progress even if the UI-side heartbeat cannot resume a thread.

[CmdletBinding()]
param(
    [string]$StateDir = "H:\elt_data\pipeline_state",
    [string]$RunsDir = "H:\elt_data\runs",
    [string]$LogDir = "H:\elt_data\pipeline_logs"
)

$ErrorActionPreference = "Continue"
$null = New-Item -ItemType Directory -Force -Path $StateDir
$null = New-Item -ItemType Directory -Force -Path $LogDir

function Read-JsonObject {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Read-LastJsonLine {
    param(
        [string]$Path,
        [string]$EventName = ""
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    $lines = Get-Content -LiteralPath $Path -Tail 80
    for ($i = $lines.Count - 1; $i -ge 0; $i--) {
        try {
            $obj = $lines[$i] | ConvertFrom-Json
            if ([string]::IsNullOrWhiteSpace($EventName) -or $obj.event -eq $EventName) {
                return $obj
            }
        } catch {
            continue
        }
    }
    return $null
}

function Read-RecentJsonLines {
    param(
        [string]$Path,
        [int]$Tail = 240
    )
    $items = @()
    if (-not (Test-Path -LiteralPath $Path)) {
        return $items
    }
    $lines = Get-Content -LiteralPath $Path -Tail $Tail
    foreach ($line in $lines) {
        try {
            $items += ($line | ConvertFrom-Json)
        } catch {
            continue
        }
    }
    return $items
}

function Get-LatestMetricsFile {
    param([string]$Root)
    if (-not (Test-Path -LiteralPath $Root)) {
        return $null
    }
    return Get-ChildItem -LiteralPath $Root -Filter "metrics.jsonl" -Recurse -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

function Get-LatestDistillStatus {
    param([string]$Root = "H:\elt_data\gguf_distill")
    if (-not (Test-Path -LiteralPath $Root)) {
        return $null
    }
    $file = Get-ChildItem -LiteralPath $Root -Filter "status.json" -Recurse -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $file) {
        return $null
    }
    $payload = Read-JsonObject $file.FullName
    if ($null -eq $payload) {
        return $null
    }
    return [ordered]@{
        path = $file.FullName
        age_sec = [math]::Round(((Get-Date) - $file.LastWriteTime).TotalSeconds, 1)
        state = $payload.state
        stage = $payload.current_stage
        processed_tasks = $payload.processed_tasks
        total_tasks = $payload.total_tasks
        progress_pct = $payload.progress_pct
        eta_sec = $payload.eta_sec
        error_count = $payload.error_count
        last_error = $payload.last_error
    }
}

function Get-GpuSnapshot {
    try {
        $line = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if ([string]::IsNullOrWhiteSpace($line)) {
            return $null
        }
        $parts = $line.Split(",") | ForEach-Object { $_.Trim() }
        return [ordered]@{
            memory_used_mb = [double]$parts[0]
            memory_total_mb = [double]$parts[1]
            utilization_pct = [double]$parts[2]
            temperature_c = [double]$parts[3]
            power_w = [double]$parts[4]
        }
    } catch {
        return $null
    }
}

function Get-DiskSnapshot {
    $items = @()
    foreach ($name in @("C", "H")) {
        try {
            $drive = Get-PSDrive -Name $name -ErrorAction Stop
            $items += [ordered]@{
                name = $name
                free_gb = [math]::Round($drive.Free / 1GB, 2)
                used_gb = [math]::Round($drive.Used / 1GB, 2)
            }
        } catch {
            continue
        }
    }
    return $items
}

function Format-DurationText {
    param([double]$Seconds)
    if ($Seconds -lt 0 -or [double]::IsNaN($Seconds)) {
        return "unknown"
    }
    $span = [TimeSpan]::FromSeconds([math]::Round($Seconds))
    if ($span.TotalDays -ge 1) {
        return "{0}d {1}h {2}m" -f [math]::Floor($span.TotalDays), $span.Hours, $span.Minutes
    }
    if ($span.TotalHours -ge 1) {
        return "{0}h {1}m" -f [math]::Floor($span.TotalHours), $span.Minutes
    }
    if ($span.TotalMinutes -ge 1) {
        return "{0}m {1}s" -f [math]::Floor($span.TotalMinutes), $span.Seconds
    }
    return "{0}s" -f $span.Seconds
}

function Get-TrainingEta {
    param(
        [string]$MetricsPath,
        [datetime]$Now
    )
    $events = Read-RecentJsonLines $MetricsPath 300
    if ($events.Count -eq 0) {
        return $null
    }

    $trainConfig = $events | Where-Object { $_.event -eq "train_config" } | Select-Object -Last 1
    $trainSteps = @($events | Where-Object { $_.event -eq "train_step" } | Sort-Object { [double]$_.ts })
    if ($null -eq $trainConfig -or $trainSteps.Count -eq 0) {
        return $null
    }

    $totalSteps = [int]$trainConfig.total_steps
    $latest = $trainSteps[-1]
    $latestStep = [int]$latest.step
    $completedSteps = [math]::Min($totalSteps, $latestStep + 1)
    $remainingSteps = [math]::Max(0, $totalSteps - $completedSteps)

    $durations = @()
    for ($i = 1; $i -lt $trainSteps.Count; $i++) {
        $prev = $trainSteps[$i - 1]
        $curr = $trainSteps[$i]
        $stepDelta = [int]$curr.step - [int]$prev.step
        $timeDelta = [double]$curr.ts - [double]$prev.ts
        if ($stepDelta -gt 0 -and $timeDelta -gt 0) {
            $durations += ($timeDelta / $stepDelta)
        }
    }

    $recentDurations = @($durations | Select-Object -Last 6)
    $avgStepSec = $null
    if ($recentDurations.Count -gt 0) {
        $avgStepSec = ($recentDurations | Measure-Object -Average).Average
    }

    $etaSec = $null
    $etaAt = $null
    $etaText = "unknown"
    $ageSinceLatestStepSec = $null
    if ($avgStepSec -ne $null) {
        $nowUnix = [DateTimeOffset]::new($Now).ToUnixTimeSeconds()
        $ageSinceLatestStepSec = [math]::Max(0.0, [double]$nowUnix - [double]$latest.ts)
        $etaSec = [math]::Max(0.0, ($remainingSteps * [double]$avgStepSec) - $ageSinceLatestStepSec)
        $etaAt = $Now.AddSeconds($etaSec).ToString("o")
        $etaText = Format-DurationText $etaSec
    }

    $progressPct = 0.0
    if ($totalSteps -gt 0) {
        $progressPct = [math]::Round(($completedSteps / $totalSteps) * 100.0, 2)
    }

    return [ordered]@{
        scope = "current_training_run"
        total_steps = $totalSteps
        latest_step = $latestStep
        completed_steps = $completedSteps
        remaining_steps = $remainingSteps
        progress_pct = $progressPct
        avg_step_sec_recent = if ($avgStepSec -ne $null) { [math]::Round([double]$avgStepSec, 1) } else { $null }
        latest_step_age_sec = if ($ageSinceLatestStepSec -ne $null) { [math]::Round([double]$ageSinceLatestStepSec, 1) } else { $null }
        eta_sec = if ($etaSec -ne $null) { [math]::Round([double]$etaSec, 1) } else { $null }
        eta_text = $etaText
        estimated_completion_time = $etaAt
        basis = "recent train_step timestamp deltas from metrics.jsonl"
    }
}

$statusPath = Join-Path $StateDir "status.json"
$status = Read-JsonObject $statusPath
$metricsFile = Get-LatestMetricsFile $RunsDir
$latestStep = $null
$latestEvent = $null
$metricsAgeSec = $null
$checkpoint = $null
$eta = $null

if ($metricsFile -ne $null) {
    $latestStep = Read-LastJsonLine $metricsFile.FullName "train_step"
    $latestEvent = Read-LastJsonLine $metricsFile.FullName
    $metricsAgeSec = [math]::Round(((Get-Date) - $metricsFile.LastWriteTime).TotalSeconds, 1)
    if ($latestEvent -ne $null -and $latestEvent.ts -ne $null) {
        try {
            $nowUnix = [DateTimeOffset]::Now.ToUnixTimeSeconds()
            $metricsAgeSec = [math]::Round([math]::Max(0.0, [double]$nowUnix - [double]$latestEvent.ts), 1)
        } catch {
            $metricsAgeSec = [math]::Round(((Get-Date) - $metricsFile.LastWriteTime).TotalSeconds, 1)
        }
    }
    $eta = Get-TrainingEta $metricsFile.FullName (Get-Date)
    $runDir = $metricsFile.Directory
    $lastPt = Join-Path $runDir.FullName "last.pt"
    if (Test-Path -LiteralPath $lastPt) {
        $lastItem = Get-Item -LiteralPath $lastPt
        $checkpoint = [ordered]@{
            path = $lastItem.FullName
            bytes = $lastItem.Length
            age_sec = [math]::Round(((Get-Date) - $lastItem.LastWriteTime).TotalSeconds, 1)
            last_write_time = $lastItem.LastWriteTime.ToString("o")
        }
    }
}

$disk = Get-DiskSnapshot
$gpu = Get-GpuSnapshot
$distill = Get-LatestDistillStatus
$now = Get-Date
$summary = [ordered]@{
    generated_at = $now.ToString("o")
    status = $status
    metrics_path = if ($metricsFile) { $metricsFile.FullName } else { $null }
    metrics_age_sec = $metricsAgeSec
    latest_train_step = $latestStep
    latest_event = $latestEvent
    checkpoint = $checkpoint
    eta = $eta
    distill = $distill
    gpu = $gpu
    disk = $disk
}

$jsonPath = Join-Path $StateDir "progress_report.json"
$jsonlPath = Join-Path $StateDir "progress_reports.jsonl"
$mdPath = Join-Path $StateDir "progress_report.md"
$heartbeatPath = Join-Path $StateDir "progress_heartbeat.json"

($summary | ConvertTo-Json -Depth 16) | Set-Content -Encoding UTF8 $jsonPath
($summary | ConvertTo-Json -Depth 16 -Compress) | Add-Content -Encoding UTF8 $jsonlPath

$heartbeatSummary = [ordered]@{
    generated_at = $summary.generated_at
    state = if ($status) { $status.state } else { "unknown" }
    stage = if ($status) { $status.current_stage } else { "unknown" }
    metrics_path = if ($metricsFile) { $metricsFile.FullName } else { $null }
    step = if ($latestStep) { $latestStep.step } else { $null }
    loss = if ($latestStep) { $latestStep.loss } else { $null }
    eta = $eta
    distill = $distill
    checkpoint = $checkpoint
    gpu = $gpu
    disk = $disk
}
($heartbeatSummary | ConvertTo-Json -Depth 16) | Set-Content -Encoding UTF8 $heartbeatPath

$stage = if ($status) { $status.current_stage } else { "unknown" }
$state = if ($status) { $status.state } else { "unknown" }
$stepText = if ($latestStep) { "step $($latestStep.step), loss $($latestStep.loss)" } else { "no train_step yet" }
$gpuText = if ($gpu) { "$($gpu.memory_used_mb)/$($gpu.memory_total_mb) MB, util $($gpu.utilization_pct)%" } else { "unavailable" }
$diskText = ($disk | ForEach-Object { "$($_.name): $($_.free_gb) GB free" }) -join ", "
$ckptText = if ($checkpoint) { "$($checkpoint.age_sec) sec old, $($checkpoint.bytes) bytes" } else { "none" }
$etaText = if ($eta) { "$($eta.eta_text), estimated complete at $($eta.estimated_completion_time), progress $($eta.progress_pct)%" } else { "unknown" }
$distillText = if ($distill) { "$($distill.state)/$($distill.stage), $($distill.processed_tasks)/$($distill.total_tasks), progress $($distill.progress_pct)%, eta_sec $($distill.eta_sec), errors $($distill.error_count)" } else { "none" }

@"
# ELT pipeline progress

- generated_at: $($summary.generated_at)
- state: $state
- stage: $stage
- latest_step: $stepText
- eta_current_run: $etaText
- latest_distill: $distillText
- metrics_age_sec: $metricsAgeSec
- checkpoint: $ckptText
- gpu: $gpuText
- disk: $diskText
"@ | Set-Content -Encoding UTF8 $mdPath

Write-Output "wrote $jsonPath"
Write-Output "wrote $mdPath"
Write-Output "wrote $heartbeatPath"
