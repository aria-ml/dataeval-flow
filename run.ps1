# DataEval Application Runner (PowerShell)
# Usage: .\run.ps1 -Config PATH -Dataset PATH -Output PATH [-Model PATH] [-Cache PATH] [-CPU]

param(
    [Parameter(Mandatory=$true)]
    [string]$Config,

    [Parameter(Mandatory=$true)]
    [string]$Dataset,

    [Parameter(Mandatory=$true)]
    [string]$Output,

    [string]$Model,

    [string]$Cache,

    [switch]$CPU,

    [switch]$Help
)

# Show help
if ($Help) {
    Write-Host "DataEval Application Runner"
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\run.ps1 -Config PATH -Dataset PATH -Output PATH [-Model PATH] [-Cache PATH] [-CPU]"
    Write-Host ""
    Write-Host "Parameters:"
    Write-Host "  -Config   Path to config folder (required)"
    Write-Host "  -Dataset  Path to dataset (required)"
    Write-Host "  -Output   Path for output files (required)"
    Write-Host "  -Model    Path to model files (optional)"
    Write-Host "  -Cache    Path for computation cache (optional)"
    Write-Host "  -CPU      Use CPU container (default: GPU)"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run.ps1 -Config C:\data\config -Dataset C:\data\cifar10_test -Output C:\data\output"
    Write-Host "  .\run.ps1 -Config C:\data\config -Dataset C:\data\cifar10_test -Output C:\data\output -Model C:\data\model -Cache C:\data\cache -CPU"
    Write-Host ""
    exit 0
}

# Build mount arguments
$Mounts = @(
    "-v", "${Config}:/data/config:ro",
    "-v", "${Dataset}:/data/dataset:ro",
    "-v", "${Output}:/output"
)

if (-not [string]::IsNullOrEmpty($Model)) {
    $Mounts += "-v", "${Model}:/data/model:ro"
}

if (-not [string]::IsNullOrEmpty($Cache)) {
    $Mounts += "-v", "${Cache}:/cache"
}

# Run container
if ($CPU) {
    Write-Host "Running with CPU..."
    docker run --rm @Mounts dataeval-app:cpu
} else {
    Write-Host "Running with GPU..."
    docker run --rm --gpus all @Mounts dataeval-app:gpu
}
