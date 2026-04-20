# Removes the ELT-LM-Pipeline scheduled task. Called by scripts/pipeline.py
# after the final stage completes, so the pipeline does not re-trigger on
# subsequent boots. Also safe to run manually.

$ErrorActionPreference = "Continue"
$taskName = "ELT-LM-Pipeline"

if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Output "  unregistered: $taskName"
} else {
    Write-Output "  task not present: $taskName"
}
