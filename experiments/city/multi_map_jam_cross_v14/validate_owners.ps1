$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$owners=Get-Content -Raw -LiteralPath (Join-Path $root 'OWNERS.json') | ConvertFrom-Json -Depth 8 -DateKind String
if($owners.status -ne 'captured' -or $owners.identities.Count -lt 2){throw 'Exact owner receipt is incomplete.'}
$checks=@()
foreach($expected in $owners.identities){
    $live=Get-CimInstance Win32_Process -Filter "ProcessId=$($expected.ProcessId)"
    $matches=$false
    if($live){
        $matches=([int]$live.ProcessId -eq [int]$expected.ProcessId -and
            [int]$live.ParentProcessId -eq [int]$expected.ParentProcessId -and
            [string]$live.CreationDate -ceq [string]$expected.CreationDate -and
            [string]$live.ExecutablePath -ceq [string]$expected.ExecutablePath -and
            [string]$live.CommandLine -ceq [string]$expected.CommandLine)
    }
    if($matches){throw "Exact owner still active: PID $($expected.ProcessId)"}
    $checks+=@{pid=[int]$expected.ProcessId;exact_identity_departed=$true;pid_reused=[bool]$live}
}
$exitPath=Join-Path $root 'native.exit.json'
if(-not (Test-Path -LiteralPath $exitPath)){throw 'Native exit receipt is missing; terminal integrity failure.'}
$native=Get-Content -Raw -LiteralPath $exitPath | ConvertFrom-Json -DateKind String
if($native.exit_code -ne 0 -or -not $native.marker -or -not $native.empty_stderr){throw 'Native exit, marker or stream condition failed.'}
if((Get-Item -LiteralPath (Join-Path $root 'native.stderr.log')).Length -ne 0){throw 'Native stderr is not empty.'}
if(-not ((Get-Content -Raw -LiteralPath (Join-Path $root 'native.stdout.log')) -match '(?m)^CITY_CHECK_COMPLETE\s*$')){throw 'Native completion marker is absent.'}
if(Test-Path -LiteralPath (Join-Path $root 'run.lock')){throw 'Owned run lock remains after successful exit.'}
if(Test-Path -LiteralPath (Join-Path $root 'native.failure.json')){throw 'Native failure receipt exists.'}
$report=@{status='pass';checked_utc=[DateTime]::UtcNow.ToString('o');release_sha256=$native.release_sha256;
    native_exit_code=[int]$native.exit_code;marker='CITY_CHECK_COMPLETE';empty_closed_stderr=$true;
    absent_owned_lock=$true;owners=$checks;identity_fields=$owners.identity_fields}
$path=Join-Path $root 'PROCESS_VALIDATION.json'
$file=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try{$json=$report|ConvertTo-Json -Depth 8;$bytes=[Text.Encoding]::UTF8.GetBytes($json);$file.Write($bytes,0,$bytes.Length);$file.Flush($true)}finally{$file.Dispose()}
Write-Output "PROCESS_VALIDATION_PASS $($checks.Count)"
