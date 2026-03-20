# DataEval Workflows Runner (PowerShell)
# Usage: .\run.ps1 -Data PATH -Output PATH [-Config PATH] [-Cache PATH] [-Verbosity N] [-CPU] [-Cuda VERSION]

param(
    [Parameter(Mandatory=$true)]
    [string]$Data,

    [Parameter(Mandatory=$true)]
    [string]$Output,

    [string]$Config,

    [string]$Cache,

    [ValidateRange(0, 3)]
    [int]$Verbosity = 0,

    [ValidateSet("cu118", "cu124", "cu128")]
    [string]$Cuda = "cu124",

    [switch]$CPU,

    [switch]$Help
)

# Show help
if ($Help) {
    Write-Host "DataEval Workflows Runner"
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\run.ps1 -Data PATH -Output PATH [-Config PATH] [-Cache PATH] [-Verbosity N] [-CPU] [-Cuda VERSION]"
    Write-Host ""
    Write-Host "Parameters:"
    Write-Host "  -Data       Path to data directory (required, mounted read-only)"
    Write-Host "  -Output     Path for output files (required, mounted read-write)"
    Write-Host "  -Config     Config file or folder relative to data dir (optional)"
    Write-Host "  -Cache      Path for computation cache (optional, mounted read-write)"
    Write-Host "  -Verbosity  Verbosity level: 1=report, 2=+INFO, 3=+DEBUG (default: 0)"
    Write-Host "  -Cuda       CUDA variant: cu118, cu124, cu128 (default: cu124)"
    Write-Host "  -CPU        Use CPU container (default: GPU)"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run.ps1 -Data C:\data\myproject -Output C:\data\output"
    Write-Host "  .\run.ps1 -Data C:\data\myproject -Output C:\data\output -Config config/"
    Write-Host "  .\run.ps1 -Data C:\data\myproject -Output C:\data\output -Cuda cu118"
    Write-Host "  .\run.ps1 -Data C:\data\myproject -Output C:\data\output -Cache C:\data\cache -Verbosity 1 -CPU"
    Write-Host ""
    exit 0
}

# Build mount arguments
$Mounts = @(
    "-v", "${Data}:/dataeval:ro",
    "-v", "${Output}:/output"
)

if (-not [string]::IsNullOrEmpty($Cache)) {
    $Mounts += "-v", "${Cache}:/cache"
}

# Build container command — forward flags to container_run.py
$Cmd = @("python", "src/container_run.py")
if (-not [string]::IsNullOrEmpty($Config)) {
    $Cmd += "--config", $Config
}
if ($Verbosity -gt 0) {
    $Cmd += "-" + ("v" * $Verbosity)
}

# Run container
if ($CPU) {
    Write-Host "Running with CPU..."
    docker run --rm @Mounts dataeval:cpu @Cmd
} else {
    $ImageTag = "dataeval:${Cuda}"
    Write-Host "Running with GPU (${ImageTag})..."
    docker run --rm --gpus all @Mounts $ImageTag @Cmd
}
