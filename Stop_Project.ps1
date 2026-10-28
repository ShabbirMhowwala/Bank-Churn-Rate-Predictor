$ErrorActionPreference = 'Stop'
$appPath = Join-Path $PSScriptRoot 'app.py'
$appPattern = '"' + [regex]::Escape($appPath) + '"|' + [regex]::Escape($appPath) + '(?=\s|$)'
$targets = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @('python.exe', 'pythonw.exe') -and
    $_.CommandLine -match '\s-m\s+streamlit\s+run\s' -and
    $_.CommandLine -match $appPattern
})
if ($targets.Count -eq 0) {
    Write-Host 'FinGuard AI is already stopped.'
} else {
    foreach ($targetProcess in $targets) {
        Stop-Process -Id $targetProcess.ProcessId -ErrorAction SilentlyContinue
    }
    Write-Host 'FinGuard AI has stopped. Double-click Run_Project.bat to start it again.'
}
