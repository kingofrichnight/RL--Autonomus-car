$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
foreach ($name in @('native.stdout.log','native.stderr.log','native.exit.json')) {
    if (Test-Path -LiteralPath (Join-Path $PSScriptRoot $name)) { throw "Preserved output already exists: $name" }
}
& 'C:/Users/hades/OneDrive/Documents/RL--Autonomus-car/.venv/Scripts/python.exe' -B (Join-Path $PSScriptRoot 'run_check.py') 1> (Join-Path $PSScriptRoot 'native.stdout.log') 2> (Join-Path $PSScriptRoot 'native.stderr.log')
$nativeCode = $LASTEXITCODE
$record = @{exit_code=$nativeCode; ended_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json
$exitPath = Join-Path $PSScriptRoot 'native.exit.json'
$file = [IO.File]::Open($exitPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try { $bytes=[Text.Encoding]::UTF8.GetBytes($record); $file.Write($bytes,0,$bytes.Length); $file.Flush($true) } finally {$file.Dispose()}
Write-Output $record
exit $nativeCode
