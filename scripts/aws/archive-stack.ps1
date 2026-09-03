param(
    [Parameter(Mandatory = $true)]
    [switch]$ConfirmArchive,
    [string]$StackName = "all-tmd-v2-worker",
    [string]$Region = "ap-southeast-2",
    [string]$Profile = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
Initialize-AwsContext -Region $Region -Profile $Profile
$outputs = Get-AllTmdStackOutputs -StackName $StackName
Write-Host "Deleting stack $StackName."
Write-Host "CloudFormation will snapshot EBS volume $($outputs.DataVolumeId) and retain S3 bucket $($outputs.BucketName)."
Invoke-AllTmdAws -Arguments @(
    "cloudformation", "delete-stack", "--stack-name", $StackName
) -AllowEmpty
Invoke-AllTmdAws -Arguments @(
    "cloudformation", "wait", "stack-delete-complete", "--stack-name", $StackName
) -AllowEmpty
Write-Host "Stack archived. Review and delete the retained snapshot or S3 data manually when no longer required."
