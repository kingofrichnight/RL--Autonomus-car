"""Independently read back all 30 recordings and verify source-frame identity."""
import hashlib,json
from pathlib import Path
from run_check import ROOT


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


release=read(ROOT/'release.json')
for p,h in release['inputs'].items():assert sha(Path(p))==h,p
assert read(ROOT/'native.exit.json')['exit_code']==0
assert read(ROOT/'native.exit.json')['marker'] is True
assert read(ROOT/'native.exit.json')['empty_stderr'] is True
assert (ROOT/'native.stderr.log').stat().st_size==0
assert read(ROOT/'PROCESS_VALIDATION.json')['status']=='pass'
assert read(ROOT/'PRESERVATION_REVIEW.json')['status']=='pass'
complete=read(ROOT/'run_v1/complete.json');audit=read(ROOT/'run_v1/audit.json')
control=read(ROOT/'CONTROL_REVIEW.json');assist=read(ROOT/'ASSIST_REVIEW.json');delivery=read(ROOT/'DELIVERY_VALIDATION.json')
assert all(x['status']=='pass' for x in (audit,control,assist,delivery))
assert audit['complete_sha256']==assist['complete_sha256']==sha(ROOT/'run_v1/complete.json')
assert read(ROOT/'preflight.json')['status']=='pass'
catalog=read(ROOT/'catalog.json')
assert len(catalog['runs'])==30 and len(catalog['maps'])==3
assert len({(r['map_id'],r['summary']['arm'],r['summary']['density']) for r in catalog['runs']})==30
assert sum(r['source_version']=='v14' for r in catalog['runs'])==10
assert sum(r['source_version']=='v13' for r in catalog['runs'])==20
assert len(complete['episodes'])==10
for m in catalog['maps']:
    assert m['route_width']==({'district':8,'highway':7,'jam':6}[m['id']])
    assert m['finish']['width']==m['route_width']
    if m['id']!='district':assert m['road_widths'][-4*m['turns']:]==[m['route_width']]*(4*m['turns'])
    else:assert len(m['roads'])==22
checks=[];sources={}
for run in catalog['runs']:
    stage=run['summary']['stage'];source=ROOT.parent/run['source_dir'];out=source/'run_v1'
    if source not in sources:
        source_complete=read(out/'complete.json');source_release=read(source/'release.json')
        for filename,digest in source_release['inputs'].items():assert sha(Path(filename))==digest
        assert read(out/'audit.json')['complete_sha256']==sha(out/'complete.json')
        for filename,digest in source_complete['files_sha256'].items():assert sha(out/filename)==digest
        sources[source]=source_complete
    source_complete=sources[source]
    summary=next(x for x in source_complete['episodes'] if x['stage']==stage)
    assert summary==run['summary']
    path=out/f'{stage}.decisions.jsonl';assert sha(path)==source_complete['files_sha256'][path.name]
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    frames=read(ROOT/run['file'])['frames']
    assert len(frames)==len(rows)+1
    assert frames[-1]['display_complete']==summary['success']
    assert frames[-1]['simulated_seconds']==summary['simulated_seconds']
    for frame,row in zip(frames[1:],rows):
        assert frame['action']==row['action'] and frame['assist']==row['assist']
        assert frame['proposed_action']==row['proposed_action']
        assert all(frame[k]==v for k,v in row['scene'].items() if k!='action')
    if run['source_version']=='v14':
        assert summary['map_id']=='jam'
        assert len(summary['crossing_actor_ticks_by_junction'])==3
        assert all(summary['crossing_actor_ticks_by_junction'])
        assert all(frame.get('crossing_by_junction') is not None for frame in frames)
    checks.append({'stage':stage,'frames':len(frames),'replay_sha256':sha(ROOT/run['file'])})
html=(ROOT/'CITY_REPLAY.html').read_text(encoding='utf-8')
assert 'Cross-street NPC' in html and 'crossingInside' in html and '__CATALOG__' not in html
assert 'jam-cross' not in html  # no local workspace path is embedded in the page
files=['release.json','protocol.json','preflight.json','run_v1/complete.json','run_v1/audit.json',
       'CONTROL_REVIEW.json','PRESERVATION_REVIEW.json','ASSIST_REVIEW.json','DELIVERY_VALIDATION.json',
       'PROCESS_VALIDATION.json','README.md','RESULTS.md','CITY_REPLAY.html','catalog.json']
report={'status':'pass','recordings':30,'new_jam_cross_traffic_recordings':10,
    'preserved_city_highway_recordings':20,'checks':checks,'saved_proposals_and_assist_receipts_match':True,
    'all_three_jam_turns_received_cross_traffic_in_each_new_recording':True,
    'road_and_finish_widths_match':True,'files_sha256':{p:sha(ROOT/p) for p in files},
    'scope':'All recorded decisions, crossing counts, source rows and delivered frames; browser visual QA is recorded separately.'}
with (ROOT/'PACKAGE_VALIDATION.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
print('PACKAGE_VALIDATION_PASS',sum(x['frames']-1 for x in checks))
