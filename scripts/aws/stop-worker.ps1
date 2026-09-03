param(
    [string]$StackName = "all-tmd-v1-worker",
    [string]$Region = "ap-southeast-2",
    [string]$Profile = "",
    [switch]$ForceSharedWorker
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
Initialize-AwsContext -Region $Region -Profile $Profile
$outputs = Get-AllTmdSharedStackOutputs -StackName $StackName
$instanceState = Get-AllTmdEc2InstanceState -InstanceId $outputs.InstanceId
if ($instanceState -eq "running") {
    Wait-AllTmdSsmOnline -InstanceId $outputs.InstanceId -TimeoutSeconds 120
    $activeOwner = Get-AllTmdActiveRunOwner -InstanceId $outputs.InstanceId
    if ($activeOwner -and $activeOwner -ne "all-tmd-v2" -and -not $ForceSharedWorker) {
        throw "Shared worker $($outputs.InstanceId) is running $activeOwner. Use that project's stop script or pass -ForceSharedWorker intentionally."
    }
}
Invoke-AllTmdAws -Arguments @(
    "ec2", "stop-instances", "--instance-ids", $outputs.InstanceId,
    "--output", "json"
) | Out-Null
Write-Host "Stopping worker $($outputs.InstanceId). EBS and S3 data remain persistent."
