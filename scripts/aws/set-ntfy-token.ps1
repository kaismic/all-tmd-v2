param(
    [string]$ParameterName = "/all-tmd-v1/ntfy-token",
    [string]$Region = "ap-southeast-2",
    [string]$Profile = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
Initialize-AwsContext -Region $Region -Profile $Profile
$secureToken = Read-Host "ntfy access token" -AsSecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
$requestFile = New-TemporaryFile
$plainToken = $null
try {
    $plainToken = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    @{
        Name = $ParameterName
        Type = "SecureString"
        Value = $plainToken
        Overwrite = $true
    } | ConvertTo-Json | Set-Content -LiteralPath $requestFile -Encoding utf8NoBOM
    Invoke-AllTmdAws -Arguments @(
        "ssm", "put-parameter", "--cli-input-json", "file://$requestFile",
        "--output", "json"
    ) | Out-Null
    Write-Host "Stored encrypted ntfy token in $ParameterName."
}
finally {
    $plainToken = $null
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    Remove-Item -LiteralPath $requestFile -Force -ErrorAction SilentlyContinue
}
