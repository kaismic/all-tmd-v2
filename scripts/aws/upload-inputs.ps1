param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,
    [string]$StackName = "all-tmd-v2-worker",
    [string]$Region = "ap-southeast-2",
    [string]$Profile = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
Initialize-AwsContext -Region $Region -Profile $Profile
$outputs = Get-AllTmdStackOutputs -StackName $StackName
$bucket = $outputs.BucketName
$resolvedDataDir = (Resolve-Path -LiteralPath $DataDir).Path

foreach ($source in @("nor-tmd-data", "us-tmd-data")) {
    $sourcePath = Join-Path $resolvedDataDir $source
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Container)) {
        throw "Input directory was not found: $sourcePath"
    }
    Write-Host "Uploading $source to s3://$bucket/all-tmd-v2/inputs/$source/"
    Invoke-AllTmdAws -Arguments @(
        "s3", "sync", $sourcePath,
        "s3://$bucket/all-tmd-v2/inputs/$source/",
        "--only-show-errors"
    ) -AllowEmpty
}
Write-Host "All immutable training inputs are present in S3. Collector sessions are downloaded directly by the EC2 worker."
