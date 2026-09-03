param(
    [string]$StackName = "all-tmd-v2-worker",
    [string]$Region = "ap-southeast-2",
    [string]$Profile = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
Initialize-AwsContext -Region $Region -Profile $Profile
$outputs = Get-AllTmdStackOutputs -StackName $StackName
Invoke-AllTmdAws -Arguments @(
    "ec2", "stop-instances", "--instance-ids", $outputs.InstanceId,
    "--output", "json"
) | Out-Null
Write-Host "Stopping worker $($outputs.InstanceId). EBS and S3 data remain persistent."
