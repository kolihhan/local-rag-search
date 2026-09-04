$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunDir = Join-Path $Root '.run'
$TempDir = Join-Path $RunDir 'tmp'
$CacheDir = Join-Path $RunDir 'rag-cache'
New-Item -ItemType Directory -Force -Path $TempDir, $CacheDir | Out-Null
$env:TEMP = $TempDir
$env:TMP = $TempDir
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv is required.' }
Push-Location $Root
try {
    uv sync
    uv run rag-search --corpus demo_docs --cache-dir $CacheDir query "why were buyers unable to finish a purchase?" --explain
} finally { Pop-Location }
