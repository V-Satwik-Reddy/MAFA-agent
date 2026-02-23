#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Starts the MAFA Agents API server.
.DESCRIPTION
    Loads environment variables from .env file and starts the FastAPI server.
.PARAMETER Port
    Port to run the server on (default: 5001)
#>

param(
    [int]$Port = 5001
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

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

Write-Host "Starting MAFA Agents API on port $Port..." -ForegroundColor Green
Write-Host "Python: $python" -ForegroundColor Gray
& $python -m uvicorn API:app --host 0.0.0.0 --port $Port
