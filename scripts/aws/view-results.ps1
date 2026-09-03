param(
    [Parameter(ParameterSetName = "Start", Position = 0)]
    [string]$RunId = "",
    [Parameter(ParameterSetName = "Start")]
    [string]$ResultsRoot = "",
    [ValidateRange(1, 65535)]
    [int]$LocalPort = 5003,
    [Parameter(ParameterSetName = "Start")]
    [switch]$NoBrowser,
    [Parameter(ParameterSetName = "Stop", Mandatory = $true)]
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$image = "ghcr.io/mlflow/mlflow:v3.15.2"
$containerName = "all-tmd-mlflow-viewer-$LocalPort"
$viewerLabel = "all-tmd-v2.mlflow-viewer"
$runIdLabel = "all-tmd-v2.run-id"
$url = "http://127.0.0.1:$LocalPort"

function Assert-DockerAvailable {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker was not found on PATH. Install and start Docker Desktop."
    }
    & docker info --format "{{.ServerVersion}}" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker is installed but its daemon is unavailable. Start Docker Desktop and try again."
    }
}

function Get-ContainerDetails {
    $details = & docker container inspect `
        --format "{{.State.Running}}|{{index .Config.Labels `"$viewerLabel`"}}|{{index .Config.Labels `"$runIdLabel`"}}" `
        $containerName 2> $null
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    $fields = (($details | Out-String).Trim()) -split '\|', 3
    return [pscustomobject]@{
        Running = $fields[0] -eq "true"
        Viewer = $fields[1] -eq "true"
        RunId = $fields[2]
    }
}

function Open-ViewerBrowser {
    if ($NoBrowser) {
        return
    }
    try {
        Start-Process $url
    }
    catch {
        Write-Warning "MLflow is ready, but the browser could not be opened automatically: $($_.Exception.Message)"
    }
}

Assert-DockerAvailable

if ($Stop) {
    $existing = Get-ContainerDetails
    if ($null -eq $existing) {
        Write-Host "No All-TMD MLflow viewer is using local port $LocalPort."
        exit 0
    }
    if (-not $existing.Viewer) {
        throw "Container $containerName was not created by this script and will not be stopped."
    }
    & docker stop $containerName | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker could not stop MLflow viewer container $containerName."
    }
    Write-Host "Stopped the All-TMD MLflow viewer on local port $LocalPort."
    exit 0
}

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
if (-not $ResultsRoot) {
    $ResultsRoot = Join-Path $projectRoot "aws-results"
}
$resultsRootPath = [System.IO.Path]::GetFullPath($ResultsRoot)
if (-not (Test-Path -LiteralPath $resultsRootPath -PathType Container)) {
    throw "AWS results directory was not found: $resultsRootPath"
}

if ($RunId) {
    if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$') {
        throw "RunId contains unsupported characters."
    }
    $runDirectory = Get-Item -LiteralPath (Join-Path $resultsRootPath $RunId) `
        -ErrorAction SilentlyContinue
    if ($null -eq $runDirectory -or -not $runDirectory.PSIsContainer) {
        throw "Downloaded AWS run was not found: $(Join-Path $resultsRootPath $RunId)"
    }
}
else {
    $runDirectory = Get-ChildItem -LiteralPath $resultsRootPath -Directory |
        Where-Object {
            Test-Path -LiteralPath (Join-Path $_.FullName "mlflow\mlflow.db") `
                -PathType Leaf
        } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $runDirectory) {
        throw "No downloaded AWS run with an MLflow database was found beneath $resultsRootPath."
    }
    $RunId = $runDirectory.Name
}

$mlflowDirectory = Join-Path $runDirectory.FullName "mlflow"
$databasePath = Join-Path $mlflowDirectory "mlflow.db"
$artifactsPath = Join-Path $mlflowDirectory "mlartifacts"
if (-not (Test-Path -LiteralPath $databasePath -PathType Leaf)) {
    throw "The downloaded run does not contain mlflow\mlflow.db: $($runDirectory.FullName)"
}
if (-not (Test-Path -LiteralPath $artifactsPath -PathType Container)) {
    throw "The downloaded run does not contain mlflow\mlartifacts: $($runDirectory.FullName)"
}

$summaryPath = Join-Path $runDirectory.FullName "run\run-summary.json"
if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    if ($summary.exit_code -ne 0) {
        Write-Warning "AWS run $RunId finished with exit code $($summary.exit_code); MLflow may contain only partial results."
    }
}

$existing = Get-ContainerDetails
if ($null -ne $existing) {
    if (-not $existing.Viewer) {
        throw "Container $containerName already exists but was not created by this script."
    }
    if (-not $existing.Running) {
        throw "Viewer container $containerName exists but is stopped. Remove it with 'docker rm $containerName' and retry."
    }
    if ($existing.RunId -ne $RunId) {
        throw "Local port $LocalPort is already viewing AWS run $($existing.RunId). Stop it with '.\scripts\aws\view-results.ps1 -Stop -LocalPort $LocalPort'."
    }
    Write-Host "AWS run $RunId is already available in MLflow at $url"
    Open-ViewerBrowser
    exit 0
}

$legacyMount = "type=bind,source=$mlflowDirectory,target=/data/all-tmd-work"
$serverlessMount = "type=bind,source=$mlflowDirectory,target=/mlflow-data"
$dockerArguments = @(
    "run", "--detach", "--rm",
    "--name", $containerName,
    "--label", "$viewerLabel=true",
    "--label", "$runIdLabel=$RunId",
    "--publish", "127.0.0.1:${LocalPort}:5002",
    "--mount", $legacyMount,
    "--mount", $serverlessMount,
    $image,
    "mlflow", "server",
    "--host", "0.0.0.0",
    "--port", "5002",
    "--backend-store-uri", "sqlite:////data/all-tmd-work/mlflow.db",
    "--artifacts-destination", "/data/all-tmd-work/mlartifacts",
    "--allowed-hosts", "localhost:*,127.0.0.1:*"
)

Write-Host "Starting MLflow for downloaded AWS run $RunId..."
$containerId = & docker @dockerArguments
if ($LASTEXITCODE -ne 0) {
    throw "Docker could not start the MLflow viewer."
}

$deadline = [DateTimeOffset]::UtcNow.AddSeconds(60)
$ready = $false
do {
    try {
        $response = Invoke-WebRequest "$url/health" -UseBasicParsing -TimeoutSec 2
        $ready = $response.StatusCode -eq 200
    }
    catch {
        Start-Sleep -Seconds 1
    }
} while (-not $ready -and [DateTimeOffset]::UtcNow -lt $deadline)

if (-not $ready) {
    $logs = (& docker logs $containerName 2>&1 | Out-String).Trim()
    & docker stop $containerName | Out-Null
    throw "MLflow did not become healthy within 60 seconds.`n$logs"
}

Write-Host "MLflow is ready for AWS run $RunId at $url"
Write-Host "Stop it with: .\scripts\aws\view-results.ps1 -Stop -LocalPort $LocalPort"
Open-ViewerBrowser
