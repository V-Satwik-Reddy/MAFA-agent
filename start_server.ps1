#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Starts the MAFA Agents API server.
.DESCRIPTION
    Loads environment variables from .env file and starts the FastAPI server.
.PARAMETER Port
    Port to run the server on (default: 5001)
.PARAMETER AutoPortFallback
    If enabled, automatically increments the port until a free one is found.
#>

param(
    [int]$Port = 5001,
    [switch]$AutoPortFallback = $true
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Test-PortAvailable {
    param([int]$CandidatePort)

    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $CandidatePort)
        $listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($listener -ne $null) {
            $listener.Stop()
        }
    }
}

# Load .env file if it exists
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Write-Host "Loading environment from .env file..." -ForegroundColor Cyan
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
} else {
    Write-Host "Warning: .env file not found. Using existing environment variables." -ForegroundColor Yellow
}

# Suppress TensorFlow oneDNN warnings
$env:TF_ENABLE_ONEDNN_OPTS = "0"

# Find Python executable
$pythonPaths = @(
    (Join-Path (Split-Path $PSScriptRoot -Parent) ".venv/Scripts/python.exe"),
    (Join-Path $PSScriptRoot ".venv/Scripts/python.exe"),
    "python"
)

$python = $null
foreach ($path in $pythonPaths) {
    if (Test-Path $path -ErrorAction SilentlyContinue) {
        $python = $path
        break
    }
}

if (-not $python) {
    $python = "python"
}

if ($AutoPortFallback -and -not (Test-PortAvailable -CandidatePort $Port)) {
    Write-Host "Port $Port is already in use. Searching for a free port..." -ForegroundColor Yellow
    $originalPort = $Port
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        $candidate = $originalPort + $attempt + 1
        if (Test-PortAvailable -CandidatePort $candidate) {
            $Port = $candidate
            break
        }
    }
}

if (-not (Test-PortAvailable -CandidatePort $Port)) {
    throw "No free port found near the requested port. Try: .\start_server.ps1 -Port 5100"
}

Write-Host "Starting MAFA Agents API on port $Port..." -ForegroundColor Green
Write-Host "Python: $python" -ForegroundColor Gray
Write-Host "Health URL: http://127.0.0.1:$Port/health" -ForegroundColor Gray
& $python -m uvicorn API:app --host 0.0.0.0 --port $Port
