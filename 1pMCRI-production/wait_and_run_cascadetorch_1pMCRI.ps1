param(
    [string]$ExpectedOutput = "R:\code\EXTRACT-public\1pMCRI-production\output_250810-Ras2-GC#78_reg.mat",
    [int]$StableSeconds = 300,
    [int]$PollSeconds = 10,
    [int]$TimeoutMinutes = 1440,
    [int]$MaxRetries = 2
)

$ErrorActionPreference = "Stop"

$CondaExe = "C:\Users\sterada\anaconda3\Scripts\conda.exe"
$EnvName = "CascadeTorch"
$ScriptPath = "R:\code\CascadeTorch\Demo scripts\Demo_predict_1pMCRI_allcells.py"
$LogDir = "R:\code\EXTRACT-public\1pMCRI-production\logs"

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $LogDir "wait_and_run_cascadetorch_$timestamp.log"

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $logPath -Value $line
}

function Wait-ForStableFile {
    param(
        [string]$Path,
        [int]$StableSec,
        [int]$PollSec,
        [int]$TimeoutMin
    )

    $deadline = (Get-Date).AddMinutes($TimeoutMin)
    $stableStart = $null
    $lastLen = -1L
    $lastWrite = $null

    Write-Log "Waiting for file: $Path"
    Write-Log "Condition: unchanged for $StableSec seconds"

    while ((Get-Date) -lt $deadline) {
        if (Test-Path $Path) {
            $f = Get-Item $Path
            $sameLen = ($f.Length -eq $lastLen)
            $sameWrite = ($lastWrite -ne $null -and $f.LastWriteTimeUtc -eq $lastWrite)

            if ($sameLen -and $sameWrite) {
                if ($stableStart -eq $null) {
                    $stableStart = Get-Date
                }
                $elapsed = ((Get-Date) - $stableStart).TotalSeconds
                Write-Log ("File exists and unchanged: {0:N0}/{1}s (size={2})" -f $elapsed, $StableSec, $f.Length)
                if ($elapsed -ge $StableSec) {
                    Write-Log "File is stable."
                    return $true
                }
            }
            else {
                $stableStart = $null
                Write-Log ("File changed. size={0}, lastWriteUtc={1}" -f $f.Length, $f.LastWriteTimeUtc.ToString("o"))
            }

            $lastLen = $f.Length
            $lastWrite = $f.LastWriteTimeUtc
        }
        else {
            Write-Log "File not found yet."
            $stableStart = $null
            $lastLen = -1L
            $lastWrite = $null
        }

        Start-Sleep -Seconds $PollSec
    }

    Write-Log "Timeout reached while waiting for stable file."
    return $false
}

function Run-CascadeTorch {
    param(
        [string]$CondaPath,
        [string]$Env,
        [string]$PyScript
    )

    Write-Log "Running CascadeTorch script: $PyScript"
    & $CondaPath run -n $Env python $PyScript 2>&1 | Tee-Object -FilePath $logPath -Append
    return $LASTEXITCODE
}

Write-Log "=== Start wait-and-run pipeline ==="
Write-Log "Log file: $logPath"
Write-Log "Expected output: $ExpectedOutput"
Write-Log "Conda env: $EnvName"

if (-not (Test-Path $CondaExe)) {
    throw "Conda not found: $CondaExe"
}
if (-not (Test-Path $ScriptPath)) {
    throw "Script not found: $ScriptPath"
}

$ready = Wait-ForStableFile -Path $ExpectedOutput -StableSec $StableSeconds -PollSec $PollSeconds -TimeoutMin $TimeoutMinutes
if (-not $ready) {
    Write-Log "Abort: expected output did not become stable in time."
    exit 1
}

$attempt = 0
while ($attempt -le $MaxRetries) {
    $attempt += 1
    Write-Log "CascadeTorch run attempt $attempt/$($MaxRetries + 1)"
    $code = Run-CascadeTorch -CondaPath $CondaExe -Env $EnvName -PyScript $ScriptPath
    if ($code -eq 0) {
        Write-Log "CascadeTorch completed successfully."
        Write-Log "=== Done ==="
        exit 0
    }
    Write-Log "CascadeTorch failed with exit code $code"
}

Write-Log "All retry attempts failed."
Write-Log "=== Failed ==="
exit 2
