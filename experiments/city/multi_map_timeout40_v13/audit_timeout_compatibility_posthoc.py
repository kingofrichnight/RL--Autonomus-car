"""Check retained completions and exact pre-timeout prefixes without simulation."""
import json, math
import numpy as np
from city_env import traffic_geometry
from pathlib import Path
from run_check import ROOT, sha, verify, write

verify()
prior=ROOT.parent/'multi_map_city_flow_v12'
catalog=json.loads((prior/'catalog.json').read_text(encoding='utf-8'))
protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
replacements=set(protocol['stage_order'])
checks=[]
for run in catalog['runs']:
    stage=run['summary']['stage']; source=Path(run['source_dir'])/'run_v1'
    complete=json.loads((source/'complete.json').read_text(encoding='utf-8'))
    path=source/(stage+'.decisions.jsonl')
    assert sha(path)==complete['files_sha256'][path.name]
    rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
    if stage in replacements:
        fresh=ROOT/'run_v1'
        assert (source/(stage+'.initial.json')).read_bytes()==(fresh/(stage+'.initial.json')).read_bytes()
        new=[json.loads(x) for x in (fresh/path.name).read_text(encoding='utf-8').splitlines()]
        assert len(new)>len(rows),stage
        for i, old in enumerate(rows):
            expected=json.loads(json.dumps(old))
            if i==len(rows)-1:
                assert expected['truncated'] and expected['scene']['timeout_reason']=='standstill_20s'
                expected['truncated']=False
                expected['scene']['timeout_reason']=None
                # The old terminal step skipped traffic maintenance. Continuing
                # now removes actors that have passed their lane endpoint+10m.
                # Require unchanged physical ticks and exact surviving states.
                before=expected['scene']; after=new[i]['scene']
                removed=set(before['traffic_ids'])-set(after['traffic_ids'])
                assert not (set(after['traffic_ids'])-set(before['traffic_ids'])), 'Unexplained admission at old cutoff'
                for actor_id in removed:
                    actor=before['traffic'][before['traffic_ids'].index(actor_id)]
                    eligible=[]
                    for lane in traffic_geometry('jam'):
                        lng,lat=lane.local_coordinates(np.array(actor[:2]))
                        direction=np.array([math.cos(actor[2]),math.sin(actor[2])])
                        if abs(lat)<1e-6 and np.dot(direction,lane.direction)>1-1e-8:
                            eligible.append(lng>lane.length+10)
                    assert len(eligible)==1 and eligible[0], (stage,actor_id,'Removal is not beyond lane endpoint+10m')
                keep=[j for j,actor_id in enumerate(before['traffic_ids']) if actor_id not in removed]
                for field in ['traffic','traffic_ids','traffic_crashed','traffic_yielding']:
                    before[field]=[before[field][j] for j in keep]
                before['traffic_count']=len(keep)
            assert expected==new[i],(stage,i,'Unexpected pre-timeout trajectory change')
        checks.append({'stage':stage,'kind':'new40second_run','old_decisions':len(rows),
            'new_decisions':len(new),'pre_timeout_physics_and_proposals_exact':True,'old_boundary_removed_out_of_map_actor_ids':sorted(removed)})
    else:
        assert run['summary']['success'] and not run['summary']['timeout']
        initial=json.loads((source/(stage+'.initial.json')).read_text(encoding='utf-8'))
        anchor=initial['ego'][:2]; moved=0; step=0; maximum=0
        for row in rows:
            for tick in row['physics']:
                assert tick['step']==step+1
                step=tick['step'];position=tick['ego'][:2]
                if math.dist(position,anchor)>.5:anchor=position;moved=step
                elapsed=step-moved;maximum=max(maximum,elapsed)
                assert elapsed<300<600,(stage,'Retained trace reaches old timeout')
            assert row['scene']['standstill_seconds']==(step-moved)/15
            assert not row['truncated'] and row['scene']['timeout_reason'] is None
        assert rows[-1]['terminated'] and rows[-1]['scene']['success']
        checks.append({'stage':stage,'kind':'retained_completion','recorded_limit_seconds':20,
            'compatible_limit_seconds':40,'max_stationary_seconds':maximum/15,
            'all_physics_ticks_checked':step,'source_sha256':sha(path)})
assert len(checks)==30 and sum(x['kind']=='retained_completion' for x in checks)==26
write(ROOT/'TIMEOUT_COMPATIBILITY.json',{'status':'pass','standstill_limit_seconds':40,
    'retained_completions':26,'fresh_runs':4,'checks':checks,
    'scope':'Retained traces finish before either deadline. Fresh traces match every pre-cutoff physical tick, observation, proposal and action; at the old cutoff only the timeout flag and independently verified off-map NPC removal may differ. No resumed or padded recordings.', 'posthoc_correction':'The sealed optional compatibility checker assumed identical traffic lists at the old cutoff; it failed on hybrid actor46, which had passed its road endpoint by10.0356m. Continuing the episode correctly invokes traffic maintenance, skipped by the old terminal run. The original checker/error are preserved; this separate read-only checker verifies the exact removal geometry. Native episode and primary audit were successful and unchanged.', 'preserved_checker_sha256':sha(ROOT/'audit_compatibility.py'),'preserved_checker_error_sha256':sha(ROOT/'compatibility.stderr.log')})
print('TIMEOUT_COMPATIBILITY_PASS',len(checks))
