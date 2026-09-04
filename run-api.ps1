$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunDir = Join-Path $Root '.run'
$TempDir = Join-Path $RunDir 'tmp'
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
$env:TEMP = $TempDir
$env:TMP = $TempDir
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv is required.' }
Push-Location $Root
try {
    uv sync
    uv run uvicorn local_rag.api:demo_app --host 127.0.0.1 --port 8000
} finally { Pop-Location }
