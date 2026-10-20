"""Readback all delivered recordings, controller receipts and map metadata."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

release=read(ROOT/'release.json')
for p,h in release['inputs'].items():assert sha(Path(p))==h,p
assert read(ROOT/'native.exit.json')['exit_code']==0
assert (ROOT/'native.stderr.log').stat().st_size==0
assert (ROOT/'native.stdout.log').read_text().strip().endswith('CITY_CHECK_COMPLETE')
assert read(ROOT/'PROCESS_VALIDATION.json')['status']=='pass'
complete=read(ROOT/'run_v1/complete.json')
audit=read(ROOT/'run_v1/audit.json');control=read(ROOT/'CONTROL_REVIEW.json')
assist=read(ROOT/'ASSIST_REVIEW.json');delivery=read(ROOT/'DELIVERY_VALIDATION.json')
assert all(r['status']=='pass' for r in (audit,control,assist,delivery))
assert audit['complete_sha256']==assist['complete_sha256']==sha(ROOT/'run_v1/complete.json')
catalog=read(ROOT/'catalog.json')
assert len(catalog['runs'])==30
assert read(ROOT/'TIMEOUT_COMPATIBILITY.json')['status']=='pass'
assert all(r['standstill_limit_seconds']==40 for r in catalog['runs'])
assert len({(r['map_id'],r['summary']['arm'],r['summary']['density']) for r in catalog['runs']})==30
for m in catalog['maps']:
    assert m['route_width']==({'district':8,'highway':7,'jam':6}[m['id']])
    assert m['finish']['width']==m['route_width']
    # Existing traffic lanes followed by four native branch lanes per turn.
    if m['id']!='district':assert m['road_widths'][-4*m['turns']:]==[m['route_width']]*(4*m['turns'])
    else:assert len(m['roads'])==22
checks=[]
verified_sources={}
for run in catalog['runs']:
    stage=run['summary']['stage'];source=Path(run['source_dir']);p=source/'run_v1'/f'{stage}.decisions.jsonl'
    if source not in verified_sources:
        source_complete=read(source/'run_v1/complete.json')
        assert read(source/'run_v1/audit.json')['complete_sha256']==sha(source/'run_v1/complete.json')
        for filename,digest in read(source/'release.json')['inputs'].items():assert sha(Path(filename))==digest
        verified_sources[source]=source_complete
    source_complete=verified_sources[source]
    assert sha(p)==source_complete['files_sha256'][p.name]
    rows=[json.loads(x) for x in p.read_text().splitlines()]
    replay=read(ROOT/run['file']);frames=replay['frames']
    assert len(frames)==len(rows)+1
    for f,row in zip(frames[1:],rows):
        assert f['action']==row['action']
        assert f['assist']==row['assist'] and f['proposed_action']==row['proposed_action']
        assert all(f[k]==v for k,v in row['scene'].items() if k!='action')
    assert frames[-1]['display_complete']==run['summary']['success']
    assert frames[-1]['simulated_seconds']==run['summary']['simulated_seconds']
    checks.append({'stage':stage,'frames':len(frames),'replay_sha256':sha(ROOT/run['file'])})
html=(ROOT/'CITY_REPLAY.html').read_text(encoding='utf-8')
assert 'Five assisted policies' in html and 'assistReason' in html and '__CATALOG__' not in html
files=['release.json','protocol.json','run_v1/complete.json','run_v1/audit.json','CONTROL_REVIEW.json',
       'TIMEOUT_COMPATIBILITY.json','ASSIST_REVIEW.json','DELIVERY_VALIDATION.json','PROCESS_VALIDATION.json','README.md','RESULTS.md','CITY_REPLAY.html','catalog.json']
report={'status':'pass','recordings':30,'checks':checks,'saved_proposals_and_assist_receipts_match':True,
    'road_and_finish_widths_match':True,'files_sha256':{p:sha(ROOT/p) for p in files},
    'scope':'All recorded decisions and delivered frames; browser interaction and visual inspection recorded separately.'}
with (ROOT/'PACKAGE_VALIDATION.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
print('PACKAGE_VALIDATION_PASS',sum(x['frames']-1 for x in checks))
