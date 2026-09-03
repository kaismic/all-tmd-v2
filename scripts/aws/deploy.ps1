param(
    [Parameter(Mandatory = $true)]
    [string]$BudgetEmail,
    [string]$StackName = "all-tmd-v1-worker",
    [string]$Region = "ap-southeast-2",
    [string]$Profile = "",
    [string]$InstanceType = "c7i.4xlarge",
    [int]$DataVolumeSizeGiB = 200,
    [string]$BucketName = "",
    [string]$CollectorStackName = "transport-data-collector",
    [string]$NtfyTokenParameterName = "/all-tmd-v1/ntfy-token",
    [switch]$LeaveRunning,
    [switch]$NoExecuteChangeSet
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$sharedDeploy = Join-Path $projectRoot "..\all-tmd-v1\scripts\aws\deploy.ps1"
if (-not (Test-Path -LiteralPath $sharedDeploy -PathType Leaf)) {
    throw "The all-tmd-v1 deployment script was not found at $sharedDeploy. The v1 project owns the shared AWS worker and must be checked out beside all-tmd-v2."
}

$deployParameters = @{
    BudgetEmail = $BudgetEmail
    StackName = $StackName
    Region = $Region
    Profile = $Profile
    InstanceType = $InstanceType
    DataVolumeSizeGiB = $DataVolumeSizeGiB
    CollectorStackName = $CollectorStackName
    NtfyTokenParameterName = $NtfyTokenParameterName
    ConsumerProjectName = "all-tmd-v2"
}
if ($BucketName) { $deployParameters.BucketName = $BucketName }
if ($LeaveRunning) { $deployParameters.LeaveRunning = $true }
if ($NoExecuteChangeSet) { $deployParameters.NoExecuteChangeSet = $true }

Write-Host "All-TMD v1 owns the shared worker. Delegating deployment of $StackName to $sharedDeploy."
& $sharedDeploy @deployParameters
