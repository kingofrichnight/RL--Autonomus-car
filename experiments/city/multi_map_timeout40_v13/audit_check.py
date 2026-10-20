"""Read every saved decision, reconstruct route metrics and recheck model actions."""
from pathlib import Path
from collections import Counter
import json
import math
import sys
import numpy as np
from city_env import route_geometry, traffic_geometry, ARCHIVE, MAPS, intersection_geometry
from run_check import verify,sha,write,ROOT

def finish_contact(pose,lane):
    # Independent finite-segment/rectangle edge intersection, inclusive of contact.
    endpoint=lane.position(lane.length,0);h=lane.heading_at(lane.length)
    n=np.array([-math.sin(h),math.cos(h)])
    c,s=math.cos(pose[2]),math.sin(pose[2]);rotation=np.array([[c,s],[-s,c]])
    a=rotation@(endpoint-lane.width/2*n-np.array(pose[:2]));b=rotation@(endpoint+lane.width/2*n-np.array(pose[:2]));d=b-a
    candidates=[a,b]
    for axis,half in [(0,2.5),(1,1.)]:
        if abs(d[axis])>1e-12:
            for edge in [-half,half]:
                t=(edge-a[axis])/d[axis]
                if 0<=t<=1:candidates.append(a+t*d)
    return any(abs(p[0])<=2.5+1e-9 and abs(p[1])<=1.+1e-9 for p in candidates)


def assert_npc_sweeps_clear(before,after):
    if len(before)<2:return
    start=np.array([v[1:3] for v in before]);end=np.array([v[1:3] for v in after]);movement=end-start
    ids,jds=np.triu_indices(len(start),1)
    broad=np.linalg.norm(start[ids]-start[jds],axis=1)<=math.hypot(5,2)+np.linalg.norm(movement[ids],axis=1)+np.linalg.norm(movement[jds],axis=1)
    for i,j in zip(ids[broad],jds[broad]):
        vertices=[];axes=[];padding=[]
        for k in (i,j):
            h=before[k][3];c,s=math.cos(h),math.sin(h);rotation=np.array([[c,-s],[s,c]])
            vertices.append(np.array([[-2.5,-1],[-2.5,1],[2.5,1],[2.5,-1]])@rotation.T+start[k])
            axes.extend([rotation[:,0],rotation[:,1]])
            padding.append(math.hypot(5,2)/2*abs((after[k][3]-h+math.pi)%(2*math.pi)-math.pi))
        lower,upper=0.,1.
        # Require physical body separation; runtime adds 0.2 m extra clearance.
        for axis in axes:
            p1=vertices[0]@axis;p2=vertices[1]@axis;v=float((movement[i]-movement[j])@axis)
            left=float(p2.min()-p1.max()-sum(padding));right=float(p2.max()-p1.min()+sum(padding))
            if abs(v)<1e-12:
                if left>0 or right<0:lower,upper=1.,0.;break
            else:
                bounds=sorted([left/v,right/v]);lower=max(lower,bounds[0]);upper=min(upper,bounds[1])
        assert lower>upper, 'Continuous NPC rectangle separation violation'


def assert_observation_matches_scene(obs,scene,network):
    """Independent physical-distance ordering reconstruction from recorded poses."""
    e=scene['ego'];ep=np.array(e[:2]);ev=e[3]*np.array([math.cos(e[2]),math.sin(e[2])])
    lane=network.get_lane(network.get_closest_lane_index(ep,e[2]))
    es=lane.local_coordinates(ep)[0]
    eligible=[p for p in scene['traffic'] if math.dist(p[:2],e[:2])<200 and lane.local_coordinates(np.array(p[:2]))[0]-es>-10]
    eligible.sort(key=lambda p:math.dist(p[:2],e[:2]))
    expected=[[1,*np.clip(ep/200,-1,1),*np.clip(ev/80,-1,1),math.cos(e[2]),math.sin(e[2])]]
    for x,y,h,v in eligible[:14]:
        expected.append([1,*np.clip((np.array([x,y])-ep)/200,-1,1),
                         *np.clip((v*np.array([math.cos(h),math.sin(h)])-ev)/80,-1,1),math.cos(h),math.sin(h)])
    expected.extend([[0]*7 for _ in range(15-len(expected))])
    np.testing.assert_allclose(obs[:105].reshape(15,7),expected,atol=2e-6,rtol=0)


def main():
    verify()
    out = ROOT/'run_v1'
    complete = json.loads((out/'complete.json').read_text())
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert [s['stage'] for s in complete['episodes']]==protocol['stage_order']
    assert complete['learned_policy_episodes']==len(protocol['stage_order'])==4 and complete['training_interactions']==0
    exit_record = json.loads((ROOT/'native.exit.json').read_text())
    assert exit_record['exit_code'] == 0
    assert (ROOT/'native.stderr.log').stat().st_size == 0
    assert (ROOT/'native.stdout.log').read_text().strip().endswith('CITY_CHECK_COMPLETE')
    assert not (out/'failure.json').exists()
    for name,digest in complete['files_sha256'].items(): assert sha(out/name)==digest,name
    sys.path.insert(0,str(ARCHIVE/'ppo_qrdqn_hybrid_recovery_v2'))
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    from hybrid_models import load_pair
    from hybrid_rule_v2 import select
    ppo,qr,_,_ = load_pair(42)

    total = 0
    sys.path[:0]=[str(ARCHIVE/'controller_pareto_escape_runtime_v1'),str(ARCHIVE/'controller_clearance_gate_study_v1')]
    from cape_pure_v1 import select as cape_select
    from cage_pure_v1 import select as cage_select
    from action_entropy_math_v1 import partition_from_state
    for summary in complete['episodes']:
        _,lanes,turn_exits=route_geometry(summary['map_id'])
        from highway_env.road.road import RoadNetwork
        network=RoadNetwork()
        for j,lane in enumerate(lanes):network.add_lane(f'r{j}',f'r{j+1}',lane)
        for j,lane in enumerate(traffic_geometry(summary['map_id'])):network.add_lane(f't{j}a',f't{j}b',lane)
        for j,lane in enumerate(intersection_geometry(summary['map_id'])):network.add_lane(f'b{j}a',f'b{j}b',lane)
        cumulative=np.cumsum([0]+[lane.length for lane in lanes])
        path = out/(summary['stage']+'.decisions.jsonl')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows)==summary['decisions'] and len(rows)>0
        observations = np.stack([np.frombuffer(bytes.fromhex(r['observation_hex']),dtype='<f4') for r in rows])
        assert observations.shape==(len(rows),163) and np.isfinite(observations).all()
        with torch.inference_mode():
            pobs,_=ppo.policy.obs_to_tensor(observations)
            qobs,_=qr.policy.obs_to_tensor(observations)
            p=ppo.policy.get_distribution(pobs).distribution.probs.cpu().numpy()
            q=qr.policy.quantile_net(qobs).cpu().numpy()
            qr_actions=qr.predict(observations,deterministic=True)[0]
            ppo_actions=ppo.predict(observations,deterministic=True)[0]
        progress, reward, counts = 5.,0.,Counter()
        previous = json.loads((out/(summary['stage']+'.initial.json')).read_text())
        anchor=np.array(previous['ego'][:2]);movement_step=0;physics_step=0;guard_holds=0
        assist_state={"active":False,"clear_count":0}
        for i,row in enumerate(rows):
            assert row['index']==i and row['action'] in (0,1,2)
            assert_observation_matches_scene(observations[i],previous,network)
            scene=row['scene'];old_progress=progress
            assert 1<=len(row['physics'])<=3
            for tick in row['physics']:
                assert tick['step']==physics_step+1
                physics_step=tick['step']
                g=tick['npc_guard'];before=g['before'];after=g['after']
                if tick is row['physics'][0]:assert set(v[0] for v in before)==set(previous['traffic_ids']),'Actor admitted between observation and action'
                assert [v[0] for v in before]==[v[0] for v in after]
                assert len(set(g['held_ids']))==len(g['held_ids'])
                guard_holds+=len(g['held_ids'])
                assert all(v[4]>=0 for v in before+after),'NPC negative speed'
                stops=dict(g['junction_stop_distances'])
                for old,new in zip(before,after):
                    if not new[-1]:
                        forward=(new[1]-old[1])*math.cos(old[3])+(new[2]-old[2])*math.sin(old[3])
                        assert forward>=-1e-8,'NPC reverse translation without impact'
                        if old[0] in stops:
                            assert math.hypot(new[1]-old[1],new[2]-old[2])<=max(0.,stops[old[0]])+1e-8,'NPC crossed reserved stop boundary'
                for a,b in zip(before,after):
                    if a[0] in g['held_ids']:assert a[1:3]==b[1:3]
                if not g['ego_crashed']:
                    assert not any(v[-1] for v in after),'NPC crashed without ego contact'
                    # Independent polygon projection/interval audit, not guard imports.
                    assert_npc_sweeps_clear(before,after)
                assert g['ego_crashed']==tick['crashed']
                pose=np.array(tick['ego'][:2]);heading=tick['ego'][2]
                j0=tick['target_route_index'];assert 0<=j0<len(lanes)
                candidates=[]
                for j in range(max(0,j0-1),min(len(lanes),j0+2)):
                    lane=lanes[j]
                    lng,_=lane.local_coordinates(pose);lng=float(np.clip(lng,0,lane.length))
                    dist=float(np.linalg.norm(pose-lane.position(lng,0)))
                    err=abs((heading-lane.heading_at(lng)+math.pi)%(2*math.pi)-math.pi)
                    candidates.append((dist+err,dist,float(cumulative[j]+lng),lane.width))
                _,dist,current,lane_width=min(candidates);off_route=dist>lane_width/2+1.
                if not off_route:progress=max(progress,current)
                if math.hypot(*(pose-anchor))>.5:anchor=pose.copy();movement_step=physics_step
                turns=sum(progress>=x+2 for x in turn_exits)
                contact=finish_contact(tick['ego'],lanes[-1])
                terminal=tick['crashed'] or off_route or (turns==MAPS[summary['map_id']]['turns'] and contact)
                stuck=physics_step-movement_step>=600 and not terminal
                if tick is not row['physics'][-1]:assert not terminal and not stuck
            assert scene['ego']==tick['ego'] and scene['crashed']==tick['crashed']
            assert scene['simulated_seconds']==physics_step/15
            assert scene['standstill_seconds']==(physics_step-movement_step)/15
            assert scene['last_movement_step']==movement_step
            assert scene['standstill_anchor']==anchor.tolist()
            assert scene['npc_guard_holds_total']==guard_holds
            assert scene['npc_guard_held_ids']==g['held_ids']
            by_id={v[0]:v for v in g['after']}
            for actor_id,actor,crashed in zip(scene['traffic_ids'],scene['traffic'],scene['traffic_crashed']):
                if actor_id in by_id:
                    assert by_id[actor_id][1:5]==actor[:4] and by_id[actor_id][-1]==crashed
                else:
                    assert actor_id in scene['traffic_admissions'] and not crashed
                    assert all(math.dist(actor[:2],p[:2])>=12-1e-8 for aid,p in zip(scene['traffic_ids'],scene['traffic']) if aid!=actor_id)
                    assert math.dist(actor[:2],scene['ego'][:2])>=12-1e-8
            assert set(scene['traffic_ids'])-set(by_id)==set(scene['traffic_admissions'])
            assert scene['approach_remaining_m']==max(0.,40.-progress)
            assert scene['timeout_reason']==('standstill_40s' if stuck else None)
            assert abs(scene['route_progress_m']-progress)<1e-6
            assert scene['turns_completed']==turns and scene['off_route']==off_route
            expected_success=not scene['crashed'] and not off_route and turns==MAPS[summary['map_id']]['turns'] and contact
            assert scene['success']==expected_success
            expected_reward=-10. if scene['crashed'] or off_route else (progress-old_progress)/10-.01+10*expected_success
            assert abs(row['reward']-expected_reward)<1e-7
            assert 0 < scene['simulated_seconds']-previous['simulated_seconds'] <= .20000001
            assert np.linalg.norm(pose-np.array(previous['ego'][:2]))<=2.1
            assert row['terminated']==bool(scene['crashed'] or off_route or expected_success)
            assert row['truncated']==bool(stuck)
            assert scene['traffic_count']==len(scene['traffic'])==len(scene['traffic_ids'])
            assert scene['traffic_count']<=({'district':(44,72),'highway':(40,60),'jam':(32,52)}[summary['map_id']][summary['density']=='heavy'])
            distances=[float(np.linalg.norm(pose-np.array(v[:2]))) for v in scene['traffic']]
            assert abs(min(distances)-scene['nearest_center_m'])<1e-7
            assert scene['background_crashes']==sum(scene['traffic_crashed'])
            assert row['input_actor_count']==int(observations[i,:105].reshape(15,7)[1:,0].sum())<=14
            if i<len(rows)-1:assert not row['terminated'] and not row['truncated']
            target=float(observations[i,105]*9)
            assert partition_from_state(previous['ego'][3],int(round(target/4.5)),target,np.array([0.,4.5,9.])).tolist()==row['partition']
            arm=summary['arm']
            if arm=='ppo':assert row['proposed_action']==int(ppo_actions[i])
            elif arm=='qr':assert row['proposed_action']==int(qr_actions[i])
            elif arm in ('cape','cage'):
                saved=row['escape']
                assert saved['baseline']==int(qr_actions[i])
                selected={'cape':cape_select,'cage':cage_select}[arm](int(qr_actions[i]),row['partition'],observations[i,106:115].reshape(3,3).astype(float).tolist())
                assert saved['selected']==row['proposed_action']==selected
                assert saved['changed']==(selected!=int(qr_actions[i]))
                assert saved['observation_float32_bits']==row['observation_hex']
            else:
                saved=row['hybrid']
                ps=np.frombuffer(bytes.fromhex(saved['probability_bits']),dtype='<f4')
                qs=np.frombuffer(bytes.fromhex(saved['quantile_bits']),dtype='<f4').reshape(3,200)
                # Batch vs single inference need only agree to float32 tolerance; exact saved scalars drive selection.
                np.testing.assert_allclose(ps,p[i],rtol=1e-5,atol=1e-6)
                np.testing.assert_allclose(qs,q[i].T,rtol=1e-5,atol=1e-5)
                assert saved['baseline']==int(qr_actions[i])
                choice=select(ps.astype(float).tolist(),qs.astype(float).tolist(),row['partition'],
                    observations[i,106:115].reshape(3,3).astype(float).tolist(),.25,saved['baseline'])
                assert choice['selected']==row['proposed_action']==saved['selected']
            from audit_assist import check_assist
            assist_state=check_assist(row,previous,observations[i],lanes,assist_state)
            reward+=row['reward'];counts[row['action']]+=1;previous=scene
        final=rows[-1]
        assert final['terminated'] or final['truncated']
        assert summary['success']==previous['success']
        assert summary['collision']==previous['crashed']
        assert summary['off_route']==previous['off_route']
        assert summary['timeout']==final['truncated']
        assert summary['timeout_reason']==previous['timeout_reason']
        assert summary['standstill_seconds']==previous['standstill_seconds']
        assert summary['npc_guard_holds_total']==guard_holds
        assert summary['simulated_seconds']==previous['simulated_seconds']
        assert summary['route_progress_m']==previous['route_progress_m']
        assert summary['turns_completed']==previous['turns_completed']
        assert abs(summary['diagnostic_reward']-reward)<1e-8
        assert summary['action_counts']==[counts[i] for i in range(3)]
        assert summary['interventions']==sum(r.get('escape',r.get('hybrid',{})).get('changed',False) for r in rows)
        assert summary['assist_interventions']==sum(r['assist']['changed'] for r in rows)
        assert summary['peak_traffic']==max(r['scene']['traffic_count'] for r in rows)
        assert summary['peak_background_crashes']==max(r['scene']['background_crashes'] for r in rows)
        assert summary['stopped_decisions']==sum(r['scene']['ego'][3]<.5 for r in rows)
        assert abs(summary['minimum_center_distance_m']-min(r['scene']['nearest_center_m'] for r in rows))<1e-7
        total+=len(rows)
    verify()
    write(out/'audit.json',{'status':'pass','episodes':len(complete['episodes']),'decisions_reconstructed':total,
        'all_observation_actor_rows_independently_reconstructed':True,'no_admission_between_observation_and_action':True,
        'all_npc_speeds_nonnegative':True,'all_nonimpact_npc_steps_forward_or_stationary':True,'model_actions_rechecked':total,'all_route_turns_rewards_and_outcomes_reconstructed':True,'all_physics_standstill_timers_reconstructed':True,
        'npc_swept_separation_checked_on_all_non_ego_collision_physics_ticks':True,
        'complete_sha256':sha(out/'complete.json'),'release_sha256':sha(ROOT/'release.json'),
        'limitation':'Same-machine engineering audit; no independent simulator or large-sample transfer validation.'})
    print('CITY_AUDIT_PASS',total)

if __name__=='__main__':main()
