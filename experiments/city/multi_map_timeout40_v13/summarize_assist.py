import json
from collections import Counter
from run_check import ROOT,sha,verify,write
verify()
catalog=json.loads((ROOT/'catalog.json').read_text(encoding='utf-8'))
assert json.loads((ROOT/'TIMEOUT_COMPATIBILITY.json').read_text(encoding='utf-8'))['status']=='pass'
runs=[]
for r in catalog['runs']:
    s=r['summary'];c=r['control_review']
    runs.append({'stage':s['stage'],'source_dir':r['source_dir'],'outcome':c['outcome'],'seconds':s['simulated_seconds'],'recorded_standstill_limit_seconds':r['recorded_standstill_limit_seconds'],'standstill_limit_seconds':40,'assist_override_count':s['assist_interventions'],'decisions':s['decisions']})
report={'status':'pass','runs':runs,'outcome_counts':dict(Counter(r['outcome'] for r in runs)),'complete_sha256':sha(ROOT/'run_v1/complete.json'),'limits':'Four fresh observed-scenario runs;26 retained completions independently compatible with40seconds. No training or safety/general superiority claim.'}
write(ROOT/'ASSIST_REVIEW.json',report)
lines=['# Driving Lab: 40-second standstill timeout','',
'All maps and five assisted policies now allow40 simulated seconds within0.5m of the last movement anchor. Moving farther resets the timer. No overall duration cap. Completion, collision and off-route still end immediately.','',
'Four earlier heavy-jam timeouts were rerun from their initial seeds. The26 successful prior recordings remain unchanged: every physics tick was checked to establish that they finished before either deadline. The original recording deadline is explicit in catalog.json. No resumed or padded recordings.','',
'City junction coordination, geometry, traffic, model weights, forecasts and Progress Assist are unchanged. Longer waiting does not guarantee that a blockage clears.','',
'| Case | Outcome | Seconds | Original recording deadline |','|---|---|---:|---:|']
for r in runs:lines.append(f"| {r['stage']} | {r['outcome']} | {r['seconds']:.1f} | {r['recorded_standstill_limit_seconds']}s |")
lines+=['','The original optional compatibility checker failed because continued simulation removes an NPC beyond the map at the former cutoff. A separate posthoc geometry check verified this expected maintenance difference; both checker versions and the original error are preserved.','[New-run audit](run_v1/audit.json) · [All30 timeout compatibility checks](TIMEOUT_COMPATIBILITY.json) · [Results](RESULTS.md)']
with (ROOT/'README.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print('ASSIST_REVIEW_PASS',report['outcome_counts'])
