# iSpy one-liner installer for Windows.
# Usage: irm https://raw.githubusercontent.com/<org>/iSpy-FRC/main/iSpy/boot/install_windows.ps1 | iex
# What ever a ps1 file is *shrug emoji if I could*

$ErrorActionPreference = "Stop"

Write-Host "iSpy Windows Installer" -ForegroundColor Cyan
Write-Host "======================"

# 1. Check for Python 3.10+
function Get-PythonCmd {
    foreach ($cmd in @("python", "py -3")) {
        try {
            $ver = & cmd /c "$cmd --version" 2>&1
            if ($ver -match "Python 3\.(1[0-9]|[2-9][0-9])") {
                return $cmd
            }
        } catch { continue }
    }
    return $null
}

$pythonCmd = Get-PythonCmd
if (-not $pythonCmd) {
    Write-Host "Python 3.10+ not found. Opening the Microsoft Store Python page." -ForegroundColor Yellow
    Start-Process "https://apps.microsoft.com/detail/9pjpw5ldxlz5"
    Write-Host "Install Python, then re-run this installer." -ForegroundColor Yellow
    exit 1
}
Write-Host "Found: $pythonCmd" -ForegroundColor Green

# 2. Clone or update the repo
$installDir = "$env:USERPROFILE\iSpy-FRC"
if (Test-Path $installDir) {
    Write-Host "Existing install found at $installDir - pulling latest..."
    Push-Location $installDir
    git pull
    Pop-Location
} else {
    Write-Host "Cloning iSpy-FRC to $installDir ..."
    git clone https://github.com/aidan-j532/iSpy-FRC.git $installDir
}

Push-Location $installDir

# 3. Install the package
Write-Host "Installing iSpy and dependencies..."
& cmd /c "$pythonCmd -m pip install -e . --break-system-packages" 2>$null
if ($LASTEXITCODE -ne 0) {
    & cmd /c "$pythonCmd -m pip install -e ."
}

# 4. Run fresh setup (prefer the `ispy` CLI; fall back to `python -m` if the
# console script didn't register, e.g. on some non-editable installs)
Write-Host "Running first-time setup..."
$ispyCommand = Get-Command ispy -ErrorAction SilentlyContinue
if ($ispyCommand) {
    & ispy setup
} else {
    & cmd /c "$pythonCmd -m iSpy.boot.boot -f"
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Run 'ispy start' from $installDir to start iSpy"
Write-Host "  (fallback: 'python -m iSpy.boot.boot')."
Write-Host "Run 'ispy start -s' -- or 'python -m iSpy.boot.boot -s' -- to start iSpy as a background service."
Write-Host "Dashboard: http://localhost:5000"

Pop-Location