"""One exclusive thirty-episode exploratory run. Does not train or overwrite results."""
from pathlib import Path
import gzip
import itertools
import hashlib
import json
import os
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())

def verify():
    release = json.loads((ROOT/'release.json').read_text())
    for name,digest in release['inputs'].items():
        if sha(name) != digest:
            raise RuntimeError('Changed released input: '+name)
    return release

def main():
    release = verify()
    protocol = json.loads((ROOT/'protocol.json').read_text())
    out = ROOT/'run_v1'
    out.mkdir()  # No retry/resume/overwriting a previous run.
    write(out/'started.json',{'pid':os.getpid(),'started_unix':time.time(),'release_sha256':sha(ROOT/'release.json'),'protocol':protocol})
    env = None
    try:
        from city_env import ARCHIVE,REPOSITORY,make_city
        sys.path[:0] = [str(ARCHIVE/'ppo_qrdqn_hybrid_recovery_v2'),str(REPOSITORY)]
        import numpy as np
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(2)
        from hybrid_models import load_pair
        from hybrid_proxy import HybridProxy
        sys.path[:0]=[str(ARCHIVE/'controller_pareto_escape_runtime_v1'),str(ARCHIVE/'controller_clearance_gate_study_v1')]
        from cape_policy_proxy_v1 import CapePolicyProxy
        from cage_policy_proxy_v1 import CagePolicyProxy
        from city_env import MAPS
        # load_pair adds the archived class/vendor paths and verifies saved provenance.
        ppo,qr,models,fingerprints = load_pair(42)
        from policy_fingerprint_v1 import initial_policy_fingerprint
        from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
        from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
        def forbid(*a,**kw): raise RuntimeError('Training is forbidden in this city check')
        OnPolicyAlgorithm.learn = OffPolicyAlgorithm.learn = forbid
        from progress_assist import ProgressAssist,turn_distance
        reports = []
        for stage in protocol['stage_order']:
            map_id,density,arm = stage.split('_')
            traffic = True
            env = make_city(map_id=map_id,traffic=True,duration=None,density=density)
            observation,info = env.reset(seed=MAPS[map_id]['seed'])
            initial = env.unwrapped.scene()
            model = {'ppo':ppo,'qr':qr}.get(arm)
            if arm == 'hybrid025': model = HybridProxy(ppo,qr,env,.25)
            if arm in ('cape','cage'): model={'cape':CapePolicyProxy,'cage':CagePolicyProxy}[arm](qr,env)
            assist=ProgressAssist()
            records, reward_sum, counts = [],0.,[0,0,0]
            before = time.perf_counter()
            with (out/f'{stage}.decisions.jsonl').open('x',encoding='utf-8') as log:
                for index in itertools.count():
                    obshex = observation.astype('<f4',copy=False).tobytes().hex()
                    partition = env.action_entropy_partition().tolist()
                    with torch.inference_mode():
                        action,_ = model.predict(observation,deterministic=True)
                    action = int(action)
                    receipt=assist.decide(action,float(env.unwrapped.vehicle.speed),float(env.unwrapped.vehicle.target_speed),
                        observation[106:115].reshape(3,3).astype(float).tolist(),
                        turn_distance(env.unwrapped.progress,env.unwrapped.route_lanes))
                    proposed=action;action=receipt['selected']
                    observation,reward,terminated,truncated,info = env.step(action)
                    row = {'index':index,'proposed_action':proposed,'assist':receipt,'action':action,'reward':reward,'terminated':terminated,'truncated':truncated,
                           'observation_hex':obshex,'partition':partition,'scene':env.unwrapped.scene(),
                           'physics':env.unwrapped.physics_trace,
                           'input_actor_count':int(np.frombuffer(bytes.fromhex(obshex),dtype='<f4')[:105].reshape(15,7)[1:,0].sum())}
                    if arm == 'hybrid025': row['hybrid'] = model.decisions[-1]
                    if arm in ('cape','cage'):row['escape']=model.decisions[-1]
                    records.append(row)
                    log.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                    reward_sum += reward; counts[action] += 1
                    if (index+1)%100 == 0:
                        log.flush(); os.fsync(log.fileno())
                        print(f'CITY_PROGRESS {stage} {index+1} decisions (no duration cap) turns={info["turns_completed"]}/{MAPS[map_id]["turns"]}',flush=True)
                    if terminated or truncated: break
                log.flush(); os.fsync(log.fileno())
            final = records[-1]['scene']
            report = {'stage':stage,'traffic':traffic,'density':density,'arm':arm,'map_id':map_id,'scenario_seed':MAPS[map_id]['seed'],'model_seed':42,
                'decisions':len(records),'success':final['success'],'collision':final['crashed'],
                'off_route':final['off_route'],'timeout':bool(truncated),'timeout_reason':final['timeout_reason'],'standstill_seconds':final['standstill_seconds'],'turns_completed':final['turns_completed'],
                'route_progress_m':final['route_progress_m'],'route_length_m':final['route_length_m'],
                'route_fraction':(final['route_progress_m']-5)/(final['route_length_m']-5),
                'simulated_seconds':final['simulated_seconds'],'diagnostic_reward':reward_sum,'action_counts':counts,
                'wall_seconds':time.perf_counter()-before,
                'required_turns':MAPS[map_id]['turns'],'interventions':sum(r.get('escape',r.get('hybrid',{})).get('changed',False) for r in records),
                'assist_interventions':sum(r['assist']['changed'] for r in records),
                'initial_traffic':initial['traffic_count'],
                'peak_traffic':max(r['scene']['traffic_count'] for r in records),
                'peak_background_crashes':max(r['scene']['background_crashes'] for r in records),
                'npc_guard_holds_total':final['npc_guard_holds_total'],
                'minimum_center_distance_m':min(r['scene']['nearest_center_m'] for r in records),
                'stopped_decisions':sum(r['scene']['ego'][3]<.5 for r in records)}
            write(out/f'{stage}.summary.json',report)
            write(out/f'{stage}.initial.json',initial)
            reports.append(report)
            print('CITY_EPISODE_COMPLETE '+json.dumps(report),flush=True)
            env.close(); env=None
            verify()
        assert fingerprints == [initial_policy_fingerprint(m) for m in (ppo,qr)], 'Model tensors changed'
        assert all(sha(p) == digest for p,digest in models.items()), 'Model bytes changed'
        verify()
        files = {p.name:sha(p) for p in out.iterdir() if p.is_file()}
        write(out/'complete.json',{'status':'complete','native_return_code':0,'release_sha256':sha(ROOT/'release.json'),
            'models':models,'episodes':reports,'files_sha256':files,'training_interactions':0,'learned_policy_episodes':len(protocol['stage_order'])})
        print('CITY_CHECK_COMPLETE',flush=True)
    except BaseException as exc:
        write(out/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        if env is not None: env.close()

if __name__ == '__main__': main()
