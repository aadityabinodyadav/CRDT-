<#
.SYNOPSIS
Builds and runs the ns-3 docker container, compiling our CRDT mesh network.

.DESCRIPTION
This script will build a local docker image named 'ns3-crdt' if it doesn't already exist.
It then mounts the local 'scratch' directory into the container's ns-3 scratch folder,
so any changes to crdt_mesh.cc are immediately picked up and compiled natively inside Linux.
Results (e.g. CSV traces) written to the scratch folder or the working dir will persist back to Windows.
#>

$ImageName = "ns3-crdt"
$ContainerName = "ns3-crdt-run"

# Check if image exists; if not, build it.
$imageExists = (docker images -q $ImageName)
if (-not $imageExists) {
    Write-Host "Building ns-3 Docker Image (this will take several minutes)..." -ForegroundColor Cyan
    docker build -t $ImageName .
} else {
    Write-Host "ns-3 Docker Image found." -ForegroundColor Green
}

# Ensure the scratch directory exists
if (-not (Test-Path "$PSScriptRoot\scratch")) {
    New-Item -ItemType Directory -Path "$PSScriptRoot\scratch" | Out-Null
}

Write-Host "Running ns-3 simulation via Docker..." -ForegroundColor Cyan

# Mount the scratch folder and run the container
# Note: we use PowerShell's automatic $PWD.Path variable to mount the current directory
docker run --rm -v "${PSScriptRoot}\scratch:/usr/local/src/ns-allinone-3.37/ns-3.37/scratch" -v "${PSScriptRoot}:/workspace" $ImageName

Write-Host "Simulation complete." -ForegroundColor Green
