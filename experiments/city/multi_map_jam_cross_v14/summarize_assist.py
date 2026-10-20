"""Summarize recorded assisted demonstrations without pooling changed conditions."""
import json
from collections import Counter
from run_check import ROOT,sha,verify,write


def main():
    verify()
    catalog=json.loads((ROOT/'catalog.json').read_text(encoding='utf-8'))
    runs=[]
    for item in catalog['runs']:
        s=item['summary'];c=item['control_review']
        runs.append({'stage':s['stage'],'map_id':s['map_id'],'density':s['density'],'policy':s['arm'],
            'source_version':item['source_version'],'outcome':c['outcome'],'seconds':s['simulated_seconds'],
            'assist_override_count':s['assist_interventions'],'decisions':s['decisions'],
            'peak_crossing_traffic':s.get('peak_crossing_traffic',0),
            'crossing_actor_ticks_by_junction':s.get('crossing_actor_ticks_by_junction',[]),
            'completion':s['success'],'collision':s['collision'],'timeout':s['timeout']})
    assert len(runs)==30
    outcomes=Counter(r['outcome'] for r in runs)
    report={'status':'pass','runs':runs,'outcome_counts':dict(outcomes),
        'new_cross_traffic_runs':sum(r['source_version']=='v14' for r in runs),
        'training_episodes':0,'complete_sha256':sha(ROOT/'run_v1/complete.json'),
        'limits':'One saved model seed and previously observed scenario seed per map. The new jam traffic geometry and ego junction priority are a development demonstration. All five policy labels include Progress Assist; no superiority or safety claim.'}
    write(ROOT/'ASSIST_REVIEW.json',report)
    lines=['# Assisted replay notes','',
        'Every displayed policy is recorded with Progress Assist v1. Saved models and intervention formulas were not retrained or tuned. Saved proposals and applied actions are separate fields. The native route controller supplies steering. This is not a comparison of untouched policies.','',
        'The 20 displayed city/highway rows are unchanged successful demonstrations from v13. The 10 jam rows are new, one-pass development evaluations after adding cross-street NPCs at the three previously empty junctions and jam-only ego priority. These conditions are different and are not pooled with v13. One model seed and one previously observed scenario were used; the results are not success-rate estimates.','',
        '| Map | Traffic | Policy | Outcome | Progress | Peak cross NPCs | Cross NPC ticks at turns 1 / 2 / 3 | Assist overrides |',
        '|---|---|---|---|---:|---:|---|---:|']
    for r in runs:
        ticks=' / '.join(map(str,r['crossing_actor_ticks_by_junction'])) or '—'
        lines.append(f"| {r['map_id']} | {r['density']} | {r['policy']} | {r['outcome']} | {r['seconds']:.1f}s | {r['peak_crossing_traffic']} | {ticks} | {r['assist_override_count']} |")
    lines+=['','NPC junction reservations are a simple simulator aid, not a traffic signal. A positive forecast or lack of recorded collisions does not establish safety. See [native audit](run_v1/audit.json), [cross-traffic preservation review](PRESERVATION_REVIEW.json), [recorded results](RESULTS.md) and [replay](CITY_REPLAY.html).','']
    with (ROOT/'README.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    print('ASSIST_REVIEW_PASS',len(runs),dict(outcomes))


if __name__=='__main__':main()
