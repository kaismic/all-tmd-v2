param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string]$StackName = "all-tmd-v2-worker",
    [string]$Region = "ap-southeast-2",
    [string]$Profile = "",
    [int]$LogLines = 80
)

$ErrorActionPreference = "Stop"
if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$') {
    throw "RunId contains unsupported characters."
}
. (Join-Path $PSScriptRoot "common.ps1")
Initialize-AwsContext -Region $Region -Profile $Profile
$outputs = Get-AllTmdStackOutputs -StackName $StackName
$instanceId = $outputs.InstanceId
$state = Invoke-AllTmdAws -Arguments @(
    "ec2", "describe-instances", "--instance-ids", $instanceId,
    "--query", "Reservations[0].Instances[0].State.Name", "--output", "text"
)
$state = ($state | Out-String).Trim()
Write-Host "Instance: $instanceId ($state)"
if ($state -eq "running") {
    Wait-AllTmdSsmOnline -InstanceId $instanceId -TimeoutSeconds 120
    $commandId = Send-AllTmdSsmCommand -InstanceId $instanceId -Commands @(
        "systemctl show all-tmd-trials.service --property=ActiveState,SubState,Result,ExecMainStatus",
        "journalctl -u all-tmd-trials.service --no-pager -n $LogLines"
    ) -Comment "Inspect All-TMD run $RunId"
    Wait-AllTmdSsmCommand -CommandId $commandId -InstanceId $instanceId | Out-Null
}
else {
    $summaryUri = "s3://$($outputs.BucketName)/all-tmd-v2/results/$RunId/run/run-summary.json"
    Write-Host "Worker is not running; reading the uploaded summary."
    Invoke-AllTmdAws -Arguments @("s3", "cp", $summaryUri, "-") -AllowEmpty
}
