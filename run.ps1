# DataEval Application Runner (PowerShell)
# Usage: .\run.ps1 -Config "C:\path\to\config" -Dataset "C:\path\to\dataset" [-Output "C:\path\to\output"] [-CPU]

param(
    [string]$Config,

    [Parameter(Position=0)]
    [string]$Dataset,

    [Parameter(Position=1)]
    [string]$Output,

    [switch]$CPU,

    [switch]$Help
)

# Show help
if ($Help -or [string]::IsNullOrEmpty($Dataset) -or [string]::IsNullOrEmpty($Config)) {
    Write-Host "DataEval Application Runner"
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\run.ps1 -Config PATH -Dataset PATH [-Output PATH] [-CPU]"
    Write-Host ""
    Write-Host "Parameters:"
    Write-Host "  -Config   Path to config folder (required)"
    Write-Host "  -Dataset  Path to dataset (required)"
    Write-Host "  -Output   Path for output files (optional)"
    Write-Host "  -CPU      Use CPU container (default: GPU)"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run.ps1 -Config C:\data\config -Dataset C:\data\cifar10_test"
    Write-Host "  .\run.ps1 -Config C:\data\config -Dataset C:\data\cifar10_test -Output C:\data\output"
    Write-Host "  .\run.ps1 -Config C:\data\config -Dataset C:\data\cifar10_test -CPU"
    Write-Host ""
    exit 0
}

# Build mount arguments (config and dataset are required)
$Mounts = "--mount type=bind,source=$Config,target=/data/config,readonly"
$Mounts = "$Mounts --mount type=bind,source=$Dataset,target=/data/dataset,readonly"

if (-not [string]::IsNullOrEmpty($Output)) {
    $Mounts = "$Mounts --mount type=bind,source=$Output,target=/output"
}

# Run container
if ($CPU) {
    Write-Host "Running with CPU..."
    Invoke-Expression "docker run $Mounts dataeval:cpu"
} else {
    Write-Host "Running with GPU..."
    Invoke-Expression "docker run --gpus all $Mounts dataeval:gpu"
}
