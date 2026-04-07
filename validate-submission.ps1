#!/usr/bin/env pwsh
<#
.SYNOPSIS
    OpenEnv Submission Validator (Windows PowerShell version)
    Mirrors exactly what validate-submission.sh checks.

.DESCRIPTION
    Step 1 — Ping HF Space: POST /reset must return 200
    Step 2 — Docker build: Dockerfile must build successfully
    Step 3 — openenv validate: openenv.yaml must be valid

.PARAMETER PingUrl
    Your HuggingFace Space URL (e.g. https://rash1453-data.hf.space)

.PARAMETER RepoDir
    Path to your repo directory (default: current directory)

.EXAMPLE
    .\validate-submission.ps1 -PingUrl https://rash1453-data.hf.space
    .\validate-submission.ps1 -PingUrl https://rash1453-data.hf.space -RepoDir C:\Users\HP\Downloads\datapipe_env\datapipe_env
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$PingUrl,

    [Parameter(Mandatory=$false)]
    [string]$RepoDir = "."
)

# ── Helpers ───────────────────────────────────────────────────────────────────
$PASS = 0
$FAIL = 0

function Log-Info  { param($msg) Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $msg" }
function Log-Pass  { param($msg) Write-Host "[$(Get-Date -Format 'HH:mm:ss')] PASSED -- $msg" -ForegroundColor Green;  $script:PASS++ }
function Log-Fail  { param($msg) Write-Host "[$(Get-Date -Format 'HH:mm:ss')] FAILED -- $msg" -ForegroundColor Red;    $script:FAIL++ }
function Log-Hint  { param($msg) Write-Host "  Hint: $msg" -ForegroundColor Yellow }
function Stop-At   { param($step) Write-Host "`nValidation stopped at $step. Fix above before continuing.`n" -ForegroundColor Red; exit 1 }

# ── Resolve repo dir ──────────────────────────────────────────────────────────
$RepoDir = Resolve-Path $RepoDir -ErrorAction SilentlyContinue
if (-not $RepoDir) {
    Write-Host "Error: directory '$RepoDir' not found"
    exit 1
}
$PingUrl = $PingUrl.TrimEnd("/")

Write-Host ""
Write-Host "========================================" -ForegroundColor White
Write-Host "  OpenEnv Submission Validator"           -ForegroundColor White
Write-Host "========================================" -ForegroundColor White
Log-Info "Repo:     $RepoDir"
Log-Info "Ping URL: $PingUrl"
Write-Host ""

# ── Step 1: Ping HF Space ─────────────────────────────────────────────────────
Log-Info "Step 1/3: Pinging HF Space ($PingUrl/reset) ..."

try {
    $response = Invoke-WebRequest `
        -Method POST `
        -Uri "$PingUrl/reset" `
        -ContentType "application/json" `
        -Body '{}' `
        -TimeoutSec 30 `
        -ErrorAction Stop

    if ($response.StatusCode -eq 200) {
        Log-Pass "HF Space is live and responds to /reset"
        $body = $response.Content | ConvertFrom-Json
        Log-Info "  Response preview: task_id=$($body.observation.task_id) rows=$($body.observation.n_rows)"
    } else {
        Log-Fail "HF Space /reset returned HTTP $($response.StatusCode) (expected 200)"
        Log-Hint "Make sure your Space is running and the URL is correct."
        Stop-At "Step 1"
    }
} catch {
    Log-Fail "HF Space not reachable: $_"
    Log-Hint "Check that your Space is running at $PingUrl"
    Log-Hint "Try opening $PingUrl in your browser first."
    Stop-At "Step 1"
}

# ── Step 2: Docker build ──────────────────────────────────────────────────────
Log-Info "Step 2/3: Running docker build ..."

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Log-Fail "docker command not found"
    Log-Hint "Install Docker: https://docs.docker.com/get-docker/"
    Stop-At "Step 2"
}

$dockerfilePath = Join-Path $RepoDir "Dockerfile"
$serverDockerfilePath = Join-Path $RepoDir "server\Dockerfile"

if (Test-Path $dockerfilePath) {
    $dockerContext = $RepoDir
} elseif (Test-Path $serverDockerfilePath) {
    $dockerContext = Join-Path $RepoDir "server"
} else {
    Log-Fail "No Dockerfile found in repo root or server\ directory"
    Stop-At "Step 2"
}

Log-Info "  Found Dockerfile in $dockerContext"

try {
    $buildOutput = docker build $dockerContext 2>&1
    if ($LASTEXITCODE -eq 0) {
        Log-Pass "Docker build succeeded"
    } else {
        Log-Fail "Docker build failed"
        $buildOutput | Select-Object -Last 20 | ForEach-Object { Write-Host "  $_" }
        Stop-At "Step 2"
    }
} catch {
    Log-Fail "Docker build error: $_"
    Stop-At "Step 2"
}

# ── Step 3: openenv validate ──────────────────────────────────────────────────
Log-Info "Step 3/3: Validating openenv.yaml ..."

# Check openenv-core is installed
if (Get-Command openenv -ErrorAction SilentlyContinue) {
    try {
        Push-Location $RepoDir
        $validateOutput = openenv validate 2>&1
        Pop-Location
        if ($LASTEXITCODE -eq 0) {
            Log-Pass "openenv validate passed"
            if ($validateOutput) { Log-Info "  $validateOutput" }
        } else {
            Log-Fail "openenv validate failed"
            Write-Host $validateOutput
            Stop-At "Step 3"
        }
    } catch {
        Pop-Location -ErrorAction SilentlyContinue
        Log-Fail "openenv validate error: $_"
        Stop-At "Step 3"
    }
} else {
    # openenv-core not installed — do manual yaml validation instead
    Log-Info "  openenv CLI not found — running manual openenv.yaml validation..."

    $yamlPath = Join-Path $RepoDir "openenv.yaml"
    if (-not (Test-Path $yamlPath)) {
        Log-Fail "openenv.yaml not found in $RepoDir"
        Log-Hint "Make sure openenv.yaml is in your repo root."
        Stop-At "Step 3"
    }

    $yaml = Get-Content $yamlPath -Raw

    # Check required fields
    $checks = @{
        "name field present"        = $yaml -match "^name:"
        "tags includes openenv"     = $yaml -match "openenv"
        "sdk: docker"               = $yaml -match "sdk:\s*docker"
        "api.reset defined"         = $yaml -match "reset:"
        "api.step defined"          = $yaml -match "step:"
        "api.state defined"         = $yaml -match "state:"
        "3+ tasks defined"          = ([regex]::Matches($yaml, "- id:")).Count -ge 3
        "reward section present"    = $yaml -match "reward:"
        "grader_weights present"    = $yaml -match "grader_weights:"
        "score_range [0,1]"         = $yaml -match "score_range"
    }

    $allPassed = $true
    foreach ($check in $checks.GetEnumerator()) {
        if ($check.Value) {
            Write-Host "  [OK] $($check.Key)" -ForegroundColor Green
        } else {
            Write-Host "  [MISSING] $($check.Key)" -ForegroundColor Red
            $allPassed = $false
        }
    }

    if ($allPassed) {
        Log-Pass "openenv.yaml validation passed (manual check)"
        Log-Hint "Install openenv-core for full validation: pip install openenv-core"
    } else {
        Log-Fail "openenv.yaml is missing required fields"
        Stop-At "Step 3"
    }
}

# ── Also validate inference.py has required stdout format ─────────────────────
Log-Info "Bonus check: Validating inference.py stdout format ..."

$inferencePath = Join-Path $RepoDir "inference.py"
if (Test-Path $inferencePath) {
    $inferenceContent = Get-Content $inferencePath -Raw
    $hasStart  = $inferenceContent -match "\[START\]"
    $hasStep   = $inferenceContent -match "\[STEP\]"
    $hasEnd    = $inferenceContent -match "\[END\]"
    $hasApiUrl = $inferenceContent -match "API_BASE_URL"
    $hasModel  = $inferenceContent -match "MODEL_NAME"
    $hasToken  = $inferenceContent -match "HF_TOKEN"
    $hasOpenAI = $inferenceContent -match "OpenAI"

    if ($hasStart -and $hasStep -and $hasEnd -and $hasApiUrl -and $hasModel -and $hasToken -and $hasOpenAI) {
        Log-Pass "inference.py has all required fields ([START][STEP][END] + env vars + OpenAI SDK)"
    } else {
        if (-not $hasStart)  { Log-Fail "inference.py missing [START] log format" }
        if (-not $hasStep)   { Log-Fail "inference.py missing [STEP] log format" }
        if (-not $hasEnd)    { Log-Fail "inference.py missing [END] log format" }
        if (-not $hasApiUrl) { Log-Fail "inference.py missing API_BASE_URL" }
        if (-not $hasModel)  { Log-Fail "inference.py missing MODEL_NAME" }
        if (-not $hasToken)  { Log-Fail "inference.py missing HF_TOKEN" }
        if (-not $hasOpenAI) { Log-Fail "inference.py missing OpenAI SDK usage" }
    }
} else {
    Log-Fail "inference.py not found in repo root"
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor White

if ($FAIL -eq 0) {
    Write-Host "  All checks passed! ($PASS passed, $FAIL failed)" -ForegroundColor Green
    Write-Host "  Your submission is ready to submit."             -ForegroundColor Green
    Write-Host "  Space URL: $PingUrl"                             -ForegroundColor Green
} else {
    Write-Host "  $PASS passed, $FAIL failed — fix above issues before submitting." -ForegroundColor Red
}

Write-Host "========================================" -ForegroundColor White
Write-Host ""

if ($FAIL -gt 0) { exit 1 } else { exit 0 }