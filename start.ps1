# ============================================================
# CitizenAI Enterprise Platform — Start All Services
# Run from project root:  .\start.ps1
# ============================================================
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $project

Write-Host ""
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host "  CitizenAI Enterprise Platform" -ForegroundColor Cyan
Write-Host "  Multi-Agent Government Services Intelligence" -ForegroundColor Cyan
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host ""

# ── Kill existing processes on ports 8080 / 3000 ──────────────
foreach ($port in 8080, 3000) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        $ownerPid = $conn.OwningProcess
        Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
        Write-Host "  [cleanup] Killed old process on port $port" -ForegroundColor Yellow
    }
}
Start-Sleep -Milliseconds 600

# ── Set env var in parent so child processes inherit it ─────────
# (works in both PowerShell 5.1 and 7+; more reliable than -Environment)
$env:LLM_PROVIDER = "mock"

# ── Backend: FastAPI on port 8080 ─────────────────────────────
Write-Host "  [backend] Starting FastAPI server (LLM_PROVIDER=mock)..." -ForegroundColor Green
$backendJob = Start-Process `
    -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList "-m", "backend.api.main" `
    -WorkingDirectory $project `
    -PassThru `
    -WindowStyle Normal
Write-Host "  [backend] PID $($backendJob.Id)" -ForegroundColor Green

# ── Frontend: Python HTTP server on port 3000 ─────────────────
Write-Host "  [frontend] Starting HTTP server..." -ForegroundColor Green
$frontendJob = Start-Process `
    -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList "frontend\serve.py" `
    -WorkingDirectory $project `
    -PassThru `
    -WindowStyle Normal
Write-Host "  [frontend] PID $($frontendJob.Id)" -ForegroundColor Green

# ── Wait for backend readiness ────────────────────────────────
Write-Host ""
Write-Host "  Waiting for backend to start..." -ForegroundColor Yellow
$retries = 0
$ready   = $false
$health  = $null
while ($retries -lt 25) {
    Start-Sleep -Seconds 1
    $retries++
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:8080/api/v1/health" `
            -TimeoutSec 2 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $health = $resp.Content | ConvertFrom-Json
            $ready  = $true
            break
        }
    } catch { }
}

Write-Host ""
if ($ready) {
    Write-Host "  =============================================" -ForegroundColor Green
    Write-Host "  All services running!" -ForegroundColor Green
    Write-Host "  =============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "    Frontend  ->  http://localhost:3000" -ForegroundColor Cyan
    Write-Host "    Backend   ->  http://localhost:8080" -ForegroundColor Cyan
    Write-Host "    Swagger   ->  http://localhost:8080/docs" -ForegroundColor Cyan
    Write-Host "    ReDoc     ->  http://localhost:8080/redoc" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "    Provider  :  $($health.provider)" -ForegroundColor White
    Write-Host "    Model     :  $($health.model)" -ForegroundColor White
    Write-Host ""
    Write-Host "  Press Enter to stop all services..." -ForegroundColor Yellow
    Read-Host | Out-Null
} else {
    Write-Host "  [ERROR] Backend did not respond within 25 seconds." -ForegroundColor Red
    Write-Host "  Run manually to see the error:" -ForegroundColor Red
    Write-Host "    .\.venv\Scripts\python.exe -m backend.api.main" -ForegroundColor Yellow
}

# ── Cleanup ────────────────────────────────────────────────────
Stop-Process -Id $backendJob.Id  -Force -ErrorAction SilentlyContinue
Stop-Process -Id $frontendJob.Id -Force -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "  Services stopped." -ForegroundColor Yellow
