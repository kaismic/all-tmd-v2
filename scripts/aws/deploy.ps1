param(
    [Parameter(Mandatory = $true)]
    [string]$BudgetEmail,
    [string]$StackName = "all-tmd-v2-worker",
    [string]$Region = "ap-southeast-2",
    [string]$Profile = "",
    [string]$InstanceType = "c7i.4xlarge",
    [int]$DataVolumeSizeGiB = 200,
    [string]$BucketName = "",
    [string]$CollectorStackName = "transport-data-collector",
    [string]$NtfyTokenParameterName = "/all-tmd-v2/ntfy-token",
    [switch]$LeaveRunning
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $PSScriptRoot "common.ps1")
Initialize-AwsContext -Region $Region -Profile $Profile
$collectorOutputs = Get-AllTmdStackOutputs -StackName $CollectorStackName
if (-not $collectorOutputs.SessionsBucketName -or -not $collectorOutputs.SessionsTableName) {
    throw "Collector stack $CollectorStackName does not expose SessionsBucketName and SessionsTableName."
}

$template = Join-Path $projectRoot "aws\cloudformation.yaml"
& python -c `
    "import sys; from pathlib import Path; import yaml; yaml.compose(Path(sys.argv[1]).read_text(encoding='utf-8'))" `
    $template
if ($LASTEXITCODE -ne 0) {
    throw "CloudFormation template is not well-formed YAML: $template"
}
Invoke-AllTmdAws -Arguments @(
    "cloudformation", "validate-template",
    "--template-body", "file://$template",
    "--output", "json"
) | Out-Null

$parameterOverrides = @(
    "InstanceType=$InstanceType",
    "DataVolumeSizeGiB=$DataVolumeSizeGiB",
    "BudgetNotificationEmail=$BudgetEmail",
    "NtfyTokenParameterName=$NtfyTokenParameterName"
    "CollectorSessionsBucketName=$($collectorOutputs.SessionsBucketName)"
    "CollectorSessionsTableName=$($collectorOutputs.SessionsTableName)"
)
if ($BucketName) {
    $parameterOverrides += "BucketName=$BucketName"
}

$deployArguments = @(
    "cloudformation", "deploy",
    "--stack-name", $StackName,
    "--template-file", $template,
    "--capabilities", "CAPABILITY_IAM",
    "--parameter-overrides"
) + $parameterOverrides
Invoke-AllTmdAws -Arguments $deployArguments -AllowEmpty

$outputs = Get-AllTmdStackOutputs -StackName $StackName
$instanceId = $outputs.InstanceId
$instanceState = Get-AllTmdEc2InstanceState -InstanceId $instanceId
if ($instanceState -eq "stopping") {
    Invoke-AllTmdAws -Arguments @(
        "ec2", "wait", "instance-stopped", "--instance-ids", $instanceId
    ) -AllowEmpty
    $instanceState = "stopped"
}
if ($instanceState -eq "stopped") {
    Invoke-AllTmdAws -Arguments @(
        "ec2", "start-instances", "--instance-ids", $instanceId,
        "--output", "json"
    ) | Out-Null
}
elseif ($instanceState -in @("shutting-down", "terminated")) {
    throw "Worker $instanceId cannot be validated because it is $instanceState."
}
Wait-AllTmdSsmOnline -InstanceId $instanceId

$ready = $false
for ($attempt = 0; $attempt -lt 60 -and -not $ready; $attempt++) {
    try {
        $commandId = Send-AllTmdSsmCommand -InstanceId $instanceId -Commands @(
            "test -f /var/lib/all-tmd-v2/bootstrap-complete",
            "docker compose version",
            "mountpoint -q /mnt/all-tmd-data"
        ) -Comment "Validate All-TMD worker bootstrap"
        Wait-AllTmdSsmCommand -CommandId $commandId -InstanceId $instanceId |
            Out-Null
        $ready = $true
    }
    catch {
        if ($attempt -eq 59) { throw }
        Start-Sleep -Seconds 10
    }
}

if (-not $LeaveRunning) {
    Invoke-AllTmdAws -Arguments @(
        "ec2", "stop-instances", "--instance-ids", $instanceId,
        "--output", "json"
    ) | Out-Null
    Write-Host "Worker $instanceId is initialized and stopping to avoid idle compute charges."
}
else {
    Write-Host "Worker $instanceId is initialized and remains running."
}
Write-Host "S3 bucket: $($outputs.BucketName)"
Write-Host "Data volume: $($outputs.DataVolumeId)"
