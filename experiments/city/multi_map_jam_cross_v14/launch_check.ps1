$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = 'C:/Users/hades/OneDrive/Documents/RL--Autonomus-car/.venv/Scripts/python.exe'
$lockPath = Join-Path $PSScriptRoot 'run.lock'
$lockStream = $null
$nativeCode = 125
$success = $false
try {
    $preflightOutput = & $python -B (Join-Path $PSScriptRoot 'preflight.py') --verify-only 2>&1
    if ($LASTEXITCODE -ne 0) { throw 'Exact release preflight failed.' }
    $preflightLine = [string]($preflightOutput | Select-Object -Last 1)
    if ($preflightLine -notmatch '^PREFLIGHT_PASS ([0-9a-f]{64}) 10$') { throw 'Preflight output did not bind the exact release and fixed 10-episode budget.' }
    $releaseHash = $Matches[1]
    foreach ($name in @('native.stdout.log','native.stderr.log','native.exit.json','native.failure.json')) {
        if (Test-Path -LiteralPath (Join-Path $PSScriptRoot $name)) { throw "Preserved output already exists: $name" }
    }
    $lockStream = [IO.File]::Open($lockPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    $lockText = @{pid=$PID;created_utc=[DateTime]::UtcNow.ToString('o');release_sha256=$releaseHash} | ConvertTo-Json -Compress
    $lockBytes = [Text.Encoding]::UTF8.GetBytes($lockText)
    $lockStream.Write($lockBytes,0,$lockBytes.Length);$lockStream.Flush($true)
    & $python -B (Join-Path $PSScriptRoot 'run_check.py') 1> (Join-Path $PSScriptRoot 'native.stdout.log') 2> (Join-Path $PSScriptRoot 'native.stderr.log')
    $nativeCode = $LASTEXITCODE
    $stdout = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'native.stdout.log')
    $stderrSize = (Get-Item -LiteralPath (Join-Path $PSScriptRoot 'native.stderr.log')).Length
    $marker = $stdout -match '(?m)^CITY_CHECK_COMPLETE\s*$'
    $success = ($nativeCode -eq 0 -and $marker -and $stderrSize -eq 0 -and (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'run_v1/complete.json')))
    $record = @{exit_code=$nativeCode;marker=$marker;empty_stderr=($stderrSize -eq 0);ended_utc=[DateTime]::UtcNow.ToString('o');release_sha256=$releaseHash} | ConvertTo-Json
    $exitPath = Join-Path $PSScriptRoot 'native.exit.json'
    $file = [IO.File]::Open($exitPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try {$bytes=[Text.Encoding]::UTF8.GetBytes($record);$file.Write($bytes,0,$bytes.Length);$file.Flush($true)} finally {$file.Dispose()}
    if (-not $success) {
        $failure = @{reason='Native run did not meet exact exit, marker, stream and artifact conditions';native_exit=$nativeCode;marker=$marker;stderr_bytes=$stderrSize;ended_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json
        $failurePath=Join-Path $PSScriptRoot 'native.failure.json'
        [IO.File]::WriteAllText($failurePath,$failure,[Text.UTF8Encoding]::new($false))
        exit 1
    }
    Write-Output $record
    exit 0
}
catch {
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'native.failure.json'))) {
        $failure = @{reason=$_.Exception.Message;ended_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json
        [IO.File]::WriteAllText((Join-Path $PSScriptRoot 'native.failure.json'),$failure,[Text.UTF8Encoding]::new($false))
    }
    throw
}
finally {
    if ($lockStream) {$lockStream.Dispose()}
    if ($success -and (Test-Path -LiteralPath $lockPath)) {Remove-Item -LiteralPath $lockPath -Force}
}
