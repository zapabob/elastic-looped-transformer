# Removes the ELT-LM-Pipeline scheduled task. This is now a manual stop helper;
# the long-run pipeline no longer self-removes on success because the 5-minute
# monitor should remain available for future resumable stages.

$ErrorActionPreference = "Continue"
$taskName = "ELT-LM-Pipeline"

if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Output "  unregistered: $taskName"
} else {
    Write-Output "  task not present: $taskName"
}
