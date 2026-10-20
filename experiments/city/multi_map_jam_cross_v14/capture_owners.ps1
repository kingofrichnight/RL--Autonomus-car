param([Parameter(Mandatory=$true)][int]$WrapperPid)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$needle=(Join-Path $root 'run_check.py').Replace('/','\')
$seen=@{}
$deadline=[DateTime]::UtcNow.AddSeconds(8)
while([DateTime]::UtcNow -lt $deadline){
    foreach($p in Get-CimInstance Win32_Process){
        $isWrapper=([int]$p.ProcessId -eq $WrapperPid)
        $isWorker=([string]$p.CommandLine).IndexOf($needle,[StringComparison]::OrdinalIgnoreCase) -ge 0
        if($isWrapper -or $isWorker){
            $key="$($p.ProcessId)|$($p.CreationDate)"
            $seen[$key]=@{ProcessId=[int]$p.ProcessId;ParentProcessId=[int]$p.ParentProcessId;
                CreationDate=[string]$p.CreationDate;ExecutablePath=[string]$p.ExecutablePath;
                CommandLine=[string]$p.CommandLine}
        }
    }
    Start-Sleep -Milliseconds 250
}
$identities=@($seen.Values | Sort-Object ProcessId,CreationDate)
if(-not ($identities | Where-Object ProcessId -eq $WrapperPid)){throw 'The exact hidden wrapper identity was not captured.'}
if(-not ($identities | Where-Object CommandLine -like "*$needle*")){throw 'No exact native worker identity was captured.'}
$receipt=@{status='captured';wrapper_pid=$WrapperPid;captured_utc=[DateTime]::UtcNow.ToString('o');
    identities=$identities;identity_fields=@('ProcessId','ParentProcessId','CreationDate','ExecutablePath','CommandLine')}
$path=Join-Path $root 'OWNERS.json'
$file=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
try{$json=$receipt|ConvertTo-Json -Depth 8;$bytes=[Text.Encoding]::UTF8.GetBytes($json);$file.Write($bytes,0,$bytes.Length);$file.Flush($true)}finally{$file.Dispose()}
Write-Output "OWNER_IDENTITIES_CAPTURED $($identities.Count)"
