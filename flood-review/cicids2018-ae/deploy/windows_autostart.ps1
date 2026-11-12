# Optional: start the dashboard automatically when you log in to Windows (a per-user Scheduled Task, no administrator rights needed).
#   Install:  powershell -ExecutionPolicy Bypass -File deploy\windows_autostart.ps1
#   Remove:   powershell -ExecutionPolicy Bypass -File deploy\windows_autostart.ps1 -Remove
# It only changes your own user's scheduled tasks. The server listens on 127.0.0.1 only.
param([switch]$Remove)
$name = "IDS-Autoencoder-Dashboard"
if ($Remove) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed task $name"
    return
}
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$py = (Get-Command python -ErrorAction Stop).Source
$action = New-ScheduledTaskAction -Execute $py -Argument "run_app.py --host 127.0.0.1 --port 8000" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Description "Autoencoder IDS dashboard (local)" -Force | Out-Null
Write-Host "Installed task $name (starts at your next logon). Open http://127.0.0.1:8000"
