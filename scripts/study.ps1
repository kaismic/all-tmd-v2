param([Parameter(ValueFromRemainingArguments = $true)][string[]]$StudyArgs)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $env:ALL_TMD_GIT_COMMIT = (& git -c "safe.directory=$($projectRoot.Replace('\', '/'))" `
        -C $projectRoot rev-parse HEAD).Trim()
    docker compose build study
    if ($LASTEXITCODE -ne 0) { throw "Study image build failed with exit code $LASTEXITCODE." }
    docker compose --profile mlflow up -d --wait mlflow
    if ($LASTEXITCODE -ne 0) { throw "MLflow failed to become healthy." }
    docker compose run --rm study @StudyArgs
    if ($LASTEXITCODE -ne 0) { throw "Study command failed with exit code $LASTEXITCODE." }
}
finally {
    Remove-Item Env:ALL_TMD_GIT_COMMIT -ErrorAction SilentlyContinue
    Pop-Location
}
