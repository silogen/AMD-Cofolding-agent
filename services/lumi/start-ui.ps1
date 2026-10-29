# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Open the UI against your running cofolding-session job on LUMI.
#
#   .\services\lumi\start-ui.ps1               waits for the job to say READY
#   .\services\lumi\start-ui.ps1 -Node nid005001
#   .\services\lumi\start-ui.ps1 -Dev          live-reloading dev server
#   .\services\lumi\start-ui.ps1 -Ssh mylumi   ssh host alias (default: lumi)
#
# Tunnels localhost:8000 through the login node to dd-web, then serves the UI
# on http://localhost:5173 with the access token added by Vite's proxy.
param([string]$Node, [switch]$Dev, [string]$Ssh = "lumi")

$ErrorActionPreference = "Stop"
$Frontend = Join-Path $PSScriptRoot "..\ui\frontend"

if (-not $Node) {
    while ($true) {
        $Job = "$(ssh $Ssh "squeue --me -h -n cofolding-session -o '%i %T %N'")".Trim()
        if (-not $Job) { throw "No cofolding-session job. Submit services/lumi/session.sbatch first." }
        $Id, $State, $Node = ($Job -split "`n")[0] -split "\s+", 3
        if ($State -eq "RUNNING") {
            $Ready = "$(ssh $Ssh "grep -c READY `$(scontrol show job $Id | grep -oP 'StdOut=\K\S+') 2>/dev/null")".Trim()
            if ($Ready -and $Ready -ne "0") { break }
            Write-Host "$(Get-Date -Format HH:mm) job $Id running on $Node, services still starting..."
        }
        else {
            Write-Host "$(Get-Date -Format HH:mm) job $Id is $State, waiting..."
        }
        Start-Sleep 30
    }
}
$Token = "$(ssh $Ssh "cat ~/.dd-web-token")".Trim()
if (-not $Token) { throw "No ~/.dd-web-token on LUMI; session.sbatch writes it." }
Write-Host "Session node: $Node"

$Tunnel = Start-Process ssh -ArgumentList "-N", "-o", "ExitOnForwardFailure=yes", "-L", "8000:${Node}:8000", $Ssh -PassThru -WindowStyle Hidden
try {
    Start-Sleep 4
    if ($Tunnel.HasExited) { throw "SSH tunnel failed (is local port 8000 in use?)" }
    $env:VITE_PROXY_TARGET = "http://localhost:8000"
    $env:VITE_PROXY_TOKEN = $Token
    Set-Location $Frontend
    if (-not (Test-Path node_modules)) { npm.cmd ci; if ($LASTEXITCODE -ne 0) { throw "npm ci failed" } }
    if ($Dev) {
        npm.cmd run dev
    }
    else {
        npx.cmd vite build
        if ($LASTEXITCODE -ne 0) { throw "UI build failed" }
        npx.cmd vite preview --port 5173 --strictPort
    }
}
finally {
    if (-not $Tunnel.HasExited) { Stop-Process -Id $Tunnel.Id }
}
