param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string]$StackName = "all-tmd-v2-worker",
    [string]$Region = "ap-southeast-2",
    [string]$Profile = "",
    [int]$LocalPort = 5002
)

$ErrorActionPreference = "Stop"
if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$') {
    throw "RunId contains unsupported characters."
}
. (Join-Path $PSScriptRoot "common.ps1")
Initialize-AwsContext -Region $Region -Profile $Profile
$outputs = Get-AllTmdStackOutputs -StackName $StackName
$instanceId = $outputs.InstanceId
$instanceState = Get-AllTmdEc2InstanceState -InstanceId $instanceId
if ($instanceState -ne "running") {
    throw "EC2 worker $instanceId is '$instanceState'; MLflow can only be viewed while the run is active."
}
Wait-AllTmdSsmOnline -InstanceId $instanceId -TimeoutSeconds 120

$parametersFile = New-TemporaryFile
$serverStartRequested = $false
try {
    $startCommandId = Send-AllTmdSsmCommand -InstanceId $instanceId -Commands @(
        "/usr/local/lib/all-tmd-v2/run-trials-cloud.sh start-mlflow --run-id '$RunId'"
    ) -Comment "Start on-demand MLflow for All-TMD run $RunId"
    $serverStartRequested = $true
    Wait-AllTmdSsmCommand -CommandId $startCommandId -InstanceId $instanceId |
        Out-Null

    @{ portNumber = @("5002"); localPortNumber = @([string]$LocalPort) } |
        ConvertTo-Json -Compress |
        Set-Content -LiteralPath $parametersFile -Encoding utf8NoBOM
    $arguments = @(
        "ssm", "start-session",
        "--target", $instanceId,
        "--document-name", "AWS-StartPortForwardingSession",
        "--parameters", "file://$parametersFile",
        "--region", $Region
    )
    if ($Profile) { $arguments += @("--profile", $Profile) }
    Write-Host "Forwarding http://localhost:$LocalPort to MLflow for run $RunId. Press Ctrl+C to stop."
    & aws @arguments
    if ($LASTEXITCODE -ne 0) { throw "Session Manager port forwarding failed." }
}
finally {
    Remove-Item -LiteralPath $parametersFile -Force -ErrorAction SilentlyContinue
    if ($serverStartRequested) {
        try {
            $stopCommandId = Send-AllTmdSsmCommand -InstanceId $instanceId `
                -Commands @(
                    "/usr/local/lib/all-tmd-v2/run-trials-cloud.sh stop-mlflow --run-id '$RunId'"
                ) `
                -Comment "Stop on-demand MLflow for All-TMD run $RunId"
            Wait-AllTmdSsmCommand -CommandId $stopCommandId `
                -InstanceId $instanceId | Out-Null
            Write-Host "Stopped on-demand MLflow for run $RunId."
        }
        catch {
            Write-Warning "Could not stop on-demand MLflow cleanly: $($_.Exception.Message)"
        }
    }
}
