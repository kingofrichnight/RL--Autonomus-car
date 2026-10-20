"""Post-exit independent speed-controller and stop-start trace audit.

No learned-policy episodes or parameter search. Reads all saved 15 Hz samples.
"""
from pathlib import Path
import json,math
from run_check import ROOT,verify,sha,write
from audit_check import assert_npc_sweeps_clear


def main():
    verify()
    assert json.loads((ROOT/'native.exit.json').read_text())['exit_code']==0
    assert json.loads((ROOT/'run_v1/audit.json').read_text())['status']=='pass'
    results=[];ticks=0
    for report in json.loads((ROOT/'run_v1/complete.json').read_text())['episodes']:
        stage=report['stage'];path=ROOT/'run_v1'/f'{stage}.decisions.jsonl'
        pre=json.loads((ROOT/'run_v1'/f'{stage}.initial.json').read_text())
        low_positive_seconds=0.;brakes_positive=0;stops=0;switches=0;last_nonidle=None
        verified=0;impact_ticks=0;maximum_error=0.;restart_proofs=[]
        positive_stop_example=None
        for text in path.read_text().splitlines():
            row=json.loads(text)
            import struct
            obs=struct.unpack('<163f',bytes.fromhex(row['observation_hex']))
            target=pre['ego_target_speed']
            if row['action']!=1:
                index=round(pre['ego'][3]/4.5)+(1 if row['action']==2 else -1)
                target=4.5*max(0,min(2,index))
            assert row['scene']['ego_target_speed']==target,(stage,row['index'],'Target overwritten')
            previous_speed=pre['ego'][3]
            for tick in row['physics']:
                ticks+=1
                # Include terminal ego-contact ticks in the independent NPC-pair check.
                assert_npc_sweeps_clear(tick['npc_guard']['before'],tick['npc_guard']['after'])
                speed=tick['ego'][3]
                if not tick['crashed']:
                    expected=previous_speed+(target-previous_speed)/.6/15
                    error=abs(speed-expected);maximum_error=max(maximum_error,error)
                    assert error<1e-9,(stage,row['index'],error,'Unexplained ego speed change')
                    verified+=1
                else:impact_ticks+=1
                previous_speed=speed
            post=row['scene'];dt=post['simulated_seconds']-pre['simulated_seconds']
            if pre['ego'][3]<.5 and obs[112]>0:
                low_positive_seconds+=dt
                if positive_stop_example is None:
                    positive_stop_example={'input_time_s':pre['simulated_seconds'],'action':row['action'],'target_mps':target,'nearest_center_m':pre['nearest_center_m'],'faster_clipped_margin_m':10*obs[112]}
            brakes_positive+=row['action']==0 and target<pre['ego'][3] and obs[112]>0
            stops+=pre['ego'][3]>=.5 and post['ego'][3]<.5
            if row['action'] in (0,2):
                switches+=last_nonidle is not None and last_nonidle!=row['action'];last_nonidle=row['action']
            if pre['ego'][3]<.5 and row['action']==2 and not post['crashed'] and len(restart_proofs)<3:
                assert post['ego'][3]>pre['ego'][3]
                restart_proofs.append({'time_s':pre['simulated_seconds'],'before_mps':pre['ego'][3],'after_mps':post['ego'][3],'target_mps':target})
            pre=post
        results.append({'stage':stage,'outcome':'complete' if report['success'] else 'collision' if report['collision'] else 'off_route' if report['off_route'] else 'timeout',
            'noncollision_ego_ticks_verified':verified,'impact_ticks_excluded_from_acceleration':impact_ticks,'max_speed_reconstruction_error_mps':maximum_error,
            'low_speed_positive_faster_forecast_seconds':low_positive_seconds,'slowing_actions_with_positive_faster_forecast':brakes_positive,
            'stop_entries_below_0_5_mps':stops,'slower_faster_direction_changes':switches,'restart_examples':restart_proofs,'positive_forecast_stop_example':positive_stop_example,
            'source_sha256':sha(path)})
    write(ROOT/'CONTROL_REVIEW.json',{'status':'pass','episodes':len(results),'physics_ticks':ticks,'npc_sweeps_including_ego_collision_ticks':'pass','runs':results,
        'limits':['Positive forecast is a finite 3-second prediction from at most 14 observed actors, not proof the road is safe.',
                  'Low speed means less than 0.5 m/s; durations attach each pre-action diagnosis to the ensuing decision interval.',
                  'All noncollision ego speeds are reconstructed from the saved policy target and native proportional controller; no ego traffic hold detected.',
                  'Three maps, one previously observed scenario each and one model seed. Exploratory engineering demonstrations, not comparative success estimates.']})
    print('CONTROL_REVIEW_PASS',len(results),ticks)
    for r in results:print(r['stage'],r['outcome'],round(r['low_speed_positive_faster_forecast_seconds'],1),r['stop_entries_below_0_5_mps'])

if __name__=='__main__':main()
