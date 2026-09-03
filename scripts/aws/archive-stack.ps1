param(
    [Parameter(Mandatory = $true)]
    [switch]$ConfirmArchive,
    [string]$StackName = "all-tmd-v1-worker",
    [string]$Region = "ap-southeast-2",
    [string]$Profile = ""
)

$ErrorActionPreference = "Stop"
throw "All-TMD v2 does not own stack $StackName. Archive the shared worker from all-tmd-v1 only; doing so disables AWS runs for both projects, retains S3, and snapshots EBS."
