# DataEval Application Runner (PowerShell)
# Usage: .\run.ps1 -Dataset "C:\path\to\dataset" [-Output "C:\path\to\output"] [-Split "test"] [-CPU]

param(
    [Parameter(Position=0)]
    [string]$Dataset,

    [Parameter(Position=1)]
    [string]$Output,

    [string]$Split,

    [switch]$CPU,

    [switch]$Help
)

# Show help
if ($Help -or [string]::IsNullOrEmpty($Dataset)) {
    Write-Host "DataEval Application Runner"
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\run.ps1 -Dataset PATH [-Output PATH] [-Split NAME] [-CPU]"
    Write-Host ""
    Write-Host "Parameters:"
    Write-Host "  -Dataset  Path to dataset (required)"
    Write-Host "  -Output   Path for output files (optional)"
    Write-Host "  -Split    Dataset split name for DatasetDict (optional, e.g., train, test)"
    Write-Host "  -CPU      Use CPU container (default: GPU)"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run.ps1 -Dataset C:\data\cifar10_test"
    Write-Host "  .\run.ps1 -Dataset C:\data\cifar10_test -Output C:\data\output"
    Write-Host "  .\run.ps1 -Dataset C:\data\cifar10_full -Split test"
    Write-Host "  .\run.ps1 -Dataset C:\data\cifar10_test -CPU"
    Write-Host ""
    exit 0
}

# Build mount arguments
$Mounts = "--mount type=bind,source=$Dataset,target=/data/dataset,readonly"

if (-not [string]::IsNullOrEmpty($Output)) {
    $Mounts = "$Mounts --mount type=bind,source=$Output,target=/output"
}

# Build env arguments
$EnvArgs = ""
if (-not [string]::IsNullOrEmpty($Split)) {
    $EnvArgs = "-e DATASET_SPLIT=$Split"
}

# Run container
if ($CPU) {
    Write-Host "Running with CPU..."
    Invoke-Expression "docker run $EnvArgs $Mounts dataeval:cpu"
} else {
    Write-Host "Running with GPU..."
    Invoke-Expression "docker run --gpus all $EnvArgs $Mounts dataeval:gpu"
}
