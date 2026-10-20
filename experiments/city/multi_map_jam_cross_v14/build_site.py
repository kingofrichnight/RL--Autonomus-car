"""Build lazy-loaded replay files only from complete, independently audited runs."""
import importlib.util,json
from pathlib import Path
import numpy as np
from city_env import MAPS,route_geometry,traffic_geometry,intersection_geometry
from run_check import ROOT,sha,verify
from finish_geometry import valid_finish_contact

verify()
control=json.loads((ROOT/'CONTROL_REVIEW.json').read_text())
assert control['status']=='pass' and control['episodes']==10
prior=ROOT.parent/'multi_map_timeout40_v13'
prior_control=json.loads((prior/'CONTROL_REVIEW.json').read_text())
assert prior_control['status']=='pass' and prior_control['episodes']==30
control_by_stage={r['stage']:r for r in prior_control['runs']}
control_by_stage.update({r['stage']:r for r in control['runs']})
replacement_stages={r['stage'] for r in control['runs']}

def observed_actor_ids(observation,scene):
    ego=np.array(scene['ego']);actors=np.array(scene['traffic'],dtype=float)
    if not len(actors):
        assert not observation[:105].reshape(15,7)[1:,0].any()
        return []
    ev=ego[3]*np.array([np.cos(ego[2]),np.sin(ego[2])])
    velocities=actors[:,3,None]*np.column_stack([np.cos(actors[:,2]),np.sin(actors[:,2])])-ev
    features=np.column_stack([np.ones(len(actors)),np.clip((actors[:,:2]-ego[:2])/200,-1,1),
                             np.clip(velocities/80,-1,1),np.cos(actors[:,2]),np.sin(actors[:,2])])
    ids=[]
    for row in observation[:105].reshape(15,7)[1:]:
        if row[0]:
            matched=np.flatnonzero(np.max(np.abs(features-row),axis=1)<2e-6)
            assert len(matched)==1,'Cannot map saved input to one actor'
            ids.append(scene['traffic_ids'][int(matched[0])])
    return ids

catalog={'maps':[],'runs':[],'type':'audited recorded demonstrations, not live inference'}
for map_id in ['district','highway','jam']:
    points,lanes,markers=route_geometry(map_id);roads=traffic_geometry(map_id)+intersection_geometry(map_id);name=MAPS[map_id]['name'];seed=MAPS[map_id]['seed']
    description={'district':'Ten turns at street-grid intersections, with separated corridors and NPC turn-taking at crossings. Traffic cap: 44 medium / 72 heavy.',
        'highway':'Four turns through crossroads with straight-through road alternatives. Traffic cap: 40 medium / 60 heavy.',
        'jam':'Three turns through active crossroads. Each junction has four incoming cross-traffic approaches with NPCs. Medium / heavy traffic caps: 44 / 64.'}[map_id]
    description+=' Start 35 m before map entry. No fixed duration; 40 s within 0.5 m ends the run.'
    limit='Car-shaped NPC coordination permits adjacent-lane passing and holds conflicting movements; abrupt holds are a simulator aid, not realistic braking. Progress Assist may override saved speed actions; it is not a safety guarantee. '
    if map_id=='district':limit+='City NPCs now take turns entering junction conflict boxes; spawning inside those boxes is rejected. This can change initial traffic placement and counts. Ego proposals and assist are unchanged. '
    if map_id=='highway':limit+='Low model-compatible speeds (0, 16.2, 32.4 km/h), not motorway-speed validation.'
    limit+='City v12: 8 m ego corridor, 6 m NPC corridors, traffic centres 8 m from street axis; highway 7 m / 6 m; jam 6 m / 5 m; city turns are on populated grid junctions; highway/jam have native branch roads through turns; lane centres, finish and off-road thresholds use the wider geometry. Retained from v7: all maps use physical-distance input ranking and corrected admission timing. Saved weights are unchanged; assisted actions and reasons are recorded separately. '
    if map_id=='jam':limit+='This version populates the three previously empty drawn junction branches with 12 incoming NPC lanes (four per junction), records cross-traffic IDs/counts and yielding. NPCs coordinate with each other; on jam only, they yield to the ego near/in each junction box. This is a simple kinematic right-of-way aid, not a signal or realistic traffic model. Changed cross-traffic geometry and ego priority make these development recordings unsuitable for pooling with v13. The actor observation remains capped at 14 nearest eligible cars; not every visible car is necessarily observed. Queue speed caps: 12.6 / 6.5 km/h. A legitimate queue can trigger the standstill timeout.'
    route=[lane.position(float(s),0).tolist() for lane in lanes for s in np.linspace(0,lane.length,40)]
    road_paths=[[lane.position(float(s),0).tolist() for s in np.linspace(0,lane.length,25)] for lane in roads]
    all_points=np.array(route+[p for road in road_paths for p in road])
    catalog['maps'].append({'id':map_id,'name':name,'turns':len(markers),'route':route,'corners':[p.tolist() for p in points[1:-1]],
        'roads':road_paths,'road_widths':[float(l.width) for l in roads],'route_width':float(lanes[0].width),'bounds':[*all_points.min(axis=0).tolist(),*all_points.max(axis=0).tolist()],
        'description':description,'limit':limit,'scenario_seed':seed,'entry':MAPS[map_id]['points'][0],
        'finish':{'point':lanes[-1].position(lanes[-1].length,0).tolist(),'heading':float(lanes[-1].heading_at(lanes[-1].length)),'width':float(lanes[-1].width)}})
sources=[(prior,False),(ROOT,True)]
destination=ROOT/'replays';destination.mkdir(exist_ok=True)
checked=[];total=0;source_total=0;finish_checks=[];limitations=[]
for folder,is_new in sources:
    out=folder/'run_v1';complete=json.loads((out/'complete.json').read_text())
    audit_path=out/'audit.json'
    audit=json.loads(audit_path.read_text())
    expected='pass'
    assert audit['status']==expected and audit['complete_sha256']==sha(out/'complete.json')
    assert json.loads((folder/'native.exit.json').read_text())['exit_code']==0
    assert (folder/'native.stderr.log').stat().st_size==0
    for p,h in json.loads((folder/'release.json').read_text())['inputs'].items():assert sha(p)==h
    for p,h in complete['files_sha256'].items():assert sha(out/p)==h
    for summary in complete['episodes']:
        if not is_new and summary['map_id']=='jam':continue
        if is_new and summary['map_id']!='jam':continue
        if not is_new and summary['stage'] in replacement_stages:continue
        if not is_new:
            assert summary['success'] and not summary['timeout'], 'Only successful recordings may be retained'
        stage=summary['stage'];map_id=summary['map_id']
        rows=[json.loads(line) for line in (out/f'{stage}.decisions.jsonl').read_text().splitlines()]
        initial=json.loads((out/f'{stage}.initial.json').read_text())
        frames=[dict(initial,action=None,input_actor_count=None,observed_ids=[],last_tick_acceleration_mps2=None)]
        pre=initial
        for row in rows:
            receipt=row.get('escape',row.get('hybrid',{}))
            obs=np.frombuffer(bytes.fromhex(row['observation_hex']),dtype='<f4')
            last=row['physics'][-1];old=row['physics'][-2]['ego'] if len(row['physics'])>1 else pre['ego']
            acc=None if last['crashed'] else (last['ego'][3]-old[3])*15
            frames.append(dict(row['scene'],action=row['action'],input_actor_count=row['input_actor_count'],
                assist=row['assist'],proposed_action=row['proposed_action'],
                baseline=receipt.get('baseline',row['action'] if summary['arm']=='qr' else None),intervention=receipt.get('changed',False),
                faster_forecast=obs[112:115].astype(float).tolist(),selected_forecast=obs[106+row['action']*3:109+row['action']*3].astype(float).tolist(),
                observed_ids=observed_actor_ids(obs,pre),input_time_s=pre['simulated_seconds'],
                input_world_count=pre['traffic_count'],input_speed_mps=pre['ego'][3],last_tick_acceleration_mps2=acc))
            pre=row['scene']
        finish=next(m['finish'] for m in catalog['maps'] if m['id']==map_id)
        for frame in frames:
            frame['display_complete']=valid_finish_contact(frame,finish['point'],finish['heading'],lane_width=finish['width'])
        contacts=[i for i,f in enumerate(frames) if f['display_complete']]
        assert bool(contacts)==summary['success'],'Finish contact would change recorded outcome: '+stage
        if contacts:frames=frames[:contacts[0]+1]
        display={'end_seconds':frames[-1]['simulated_seconds'],'complete':bool(contacts),
            'completion_rule':'First replay frame with car-body contact at the finite finish line, all turns completed, no collision/off-route.',
            'native_end_seconds':summary['simulated_seconds']}
        finish_checks.append({'stage':stage,**display,'frames_retained':len(frames),'source_frames':len(rows)+1})
        events=[]
        def event(label,i):events.append({'label':f"{label} · {frames[i]['simulated_seconds']:.1f}s",'index':i})
        event('Start',0)
        if map_id=='jam':
            entered=[i for i,f in enumerate(frames) if i>0 and f.get('crossing_in_junction_count',0)>0]
            if entered:event('Cross-street NPC enters junction',entered[0])
            yielding=[i for i,f in enumerate(frames) if i>0 and any(y and c for y,c in zip(f.get('traffic_yielding',[]),f.get('traffic_crossing_flags',[])))]
            if yielding:event('Cross-street NPC yields',yielding[0])
        for label,predicate in [
            ('FASTER with nonpositive forecast',lambda f:f['action']==2 and f['selected_forecast'][0]<=0),
            ('Braking despite positive FASTER forecast',lambda f:f['action']==0 and f['ego_target_speed']<f['input_speed_mps'] and f['faster_forecast'][0]>0),
            ('Stopped with positive FASTER forecast',lambda f:f['ego'][3]<.5 and f['faster_forecast'][0]>0)]:
            indexes=[i for i,f in enumerate(frames[1:],1) if predicate(f)]
            if indexes:event(label,indexes[0])
        for label,predicate in [('First SLOWER command',lambda f:f['action']==0),('Map entry',lambda f:f['approach_remaining_m']==0),('Waiting 5 seconds',lambda f:f['standstill_seconds']>=5),('First stop',lambda f:f['ego'][3]<.5),('First rule intervention',lambda f:f.get('intervention',False)),('First assist override',lambda f:f.get('assist',{}).get('changed',False))]:
            indexes=[i for i,f in enumerate(frames[1:],1) if predicate(f)]
            if indexes:event(label,indexes[0])
        for turn in range(1,frames[-1]['required_turns']+1):
            indexes=[i for i,f in enumerate(frames) if f['turns_completed']>=turn]
            if indexes:event(f'Turn {turn}',indexes[0])
        event('COMPLETE - finish contact' if contacts else 'Final outcome',len(frames)-1)
        path=destination/f"{map_id}_{summary['density']}_{summary['arm']}.json"
        path.write_text(json.dumps({'frames':frames,'events':events},separators=(',',':'),allow_nan=False),encoding='utf-8')
        # Roundtrip every frame; actions override the scene's intentionally-null action field.
        readback=json.loads(path.read_text());assert len(readback['frames'])==len(frames)
        for frame,row in zip(readback['frames'][1:],rows):
            assert all(frame[k]==v for k,v in row['scene'].items() if k!='action')
            assert frame['action']==row['action'] and frame['input_actor_count']==row['input_actor_count']
        catalog['runs'].append({'map_id':map_id,'source_dir':'multi_map_jam_cross_v14' if is_new else 'multi_map_timeout40_v13','source_version':'v14' if is_new else 'v13','recorded_standstill_limit_seconds':40,'standstill_limit_seconds':40,'summary':summary,'display':display,'control_review':control_by_stage[stage],'file':'replays/'+path.name})
        total+=len(frames)-1;source_total+=len(rows);checked.append({'replay':path.name,'sha256':sha(path),'source_log_sha256':sha(out/f'{stage}.decisions.jsonl')})
assert len(catalog['runs'])==30
assert len({(r['map_id'],r['summary']['arm'],r['summary']['density']) for r in catalog['runs']})==30
template=(ROOT/'replay_template.html').read_text(encoding='utf-8');assert template.count('__CATALOG__')==1
(ROOT/'CITY_REPLAY.html').write_text(template.replace('__CATALOG__',json.dumps(catalog,separators=(',',':')).replace('</','<\\/')),encoding='utf-8')
(ROOT/'catalog.json').write_text(json.dumps(catalog,indent=2),encoding='utf-8')
lines=['# Driving Lab: city-grid intersections and Progress Assist v1','',
    'Thirty recorded evaluations: 20 unchanged city/highway recordings retained from v13 and 10 new jammed-map recordings (five policies × two traffic levels) made in v14. The v14 jam map populates all three previously empty intersection approaches with NPC cross traffic. No training. One saved model seed (42), one previously observed jam scenario seed (74042); development demonstrations, not success-rate estimates.','',
    '## Rules','',
    '- v9 highway route lanes are 7 m and traffic lanes 6 m; city 8 m / 6 m and jam 6 m / 5 m. All ten city turns now coincide with the populated street grid. Highway/jam retain v9 branch-road geometry and recordings. Highway branches remain visual only; v14 populates the jam branches with NPC cross traffic. The city has a dedicated central ego corridor; traffic runs in separated outer corridors. Forecast and assistance thresholds are unchanged from v11. Finish contact, off-route checks and rendering use those widths. It retains v7, which corrects actor ranking on every map to physical proximity and moves admission before the next observation. The saved weights are unchanged but input ordering changes: this is a labelled transfer demonstration. The 14-slot cap and legacy rear filter remain; no complete-perception claim.',
    '- New telemetry shows last noncollision-tick acceleration, the selected-action forecast, and outlines actors included in the preceding decision input. Playback defaults to 1x. Progress Assist qualifies after five clear forecasts, retains recovery with hysteresis, caps turn targets, and brakes on predicted conflict. This is a heuristic, not a safety guarantee.',
    '- City and jammed-route NPCs reserve crossings in arrival order, yield outside conflict boxes, check downstream space and remain forward-only; this is a traffic-dynamics aid, not policy training. V14 adds jam-only ego junction priority; ego policy proposals, speed actions and steering remain untouched by reservation code.',
    '- No overall simulation-time or decision-count limit. Completion, collision and off-route remain terminal.',
    '- Replay display: a checkered stripe marks the finite finish line. Playback stops and displays COMPLETE at the first saved replay frame where the car body touches it without collision/off-route and after all required turns. The replay shows 100% completion then. The new native environment checks the same finish contact at each physics tick. Display classifications agree with all 30 displayed native outcomes; centre-progress metrics and native end times remain in the table below and source files.',
    '- Standstill timeout: 40 simulated seconds within 0.5m of the most recent movement anchor. Moving farther resets the timer. Checked at 15Hz, including legitimate queues.',
    '- Each route has a 40m added approach. Spawn is 35m before the original entrance; queue spawns stay beyond it.',
    '- NPC-only coordination rejects conflicting proposed physics movements and holds those NPCs at their preceding clear poses with zero speed. This can stop traffic abruptly or cause queues; it is not a realistic braking model. Native ego collisions are unchanged.',
    '- All original studies and recordings are preserved. These changed-environment results must not replace or be pooled with any earlier demonstration or study results.','']
for m in catalog['maps']:
    lines += [f"## {m['name']}",'',m['description'],'',m['limit'],'',
        '| Policy | Traffic | Outcome | Turns | Progress | Time | Peak cross NPCs | NPC hold-ticks |', '|---|---|---|---:|---:|---:|---:|---:|']
    for run in (r for r in catalog['runs'] if r['map_id']==m['id']):
        v=run['summary'];outcome='Finished' if v['success'] else 'Collision' if v['collision'] else 'Off route' if v['off_route'] else 'Standstill timeout (40s)'
        lines.append(f"| {v['arm']} | {v['density']} | {outcome} | {v['turns_completed']}/{m['turns']} | {v['route_fraction']*100:.1f}% | {v['simulated_seconds']:.2f}s | {v.get('peak_crossing_traffic','—')} | {v['npc_guard_holds_total']} |")
    lines.append('')
lines += ['## Recording provenance','', 'The26 retained successful recordings were originally made with a20second standstill deadline. They never reached that deadline and finish before the new40second condition can matter. The4 earlier timeouts were rerun from the initial seed; no frames were resumed, padded or relabelled. All30 movement clocks are independently reconstructed against40seconds.','', '## Verification and limits','', 'The native trace audit passed. A separate posthoc compatibility checker corrects the original checker assumption that traffic lists stay identical at the former cutoff: continuing the hybrid run removes one NPC beyond its lane endpoint. Exact physics, proposals, actions and retained traffic states are verified. The failed optional checker and its error remain preserved. See TIMEOUT_COMPATIBILITY.json.','',
    f'All {total} displayed decisions and every physics-step standstill timer, route metric and reward were reconstructed. Saved-model/rule actions were checked. NPC continuous rectangle separation was checked on every physics tick without an ego collision. Ego-collision terminal frames remain visible; the NPC guard does not shield the ego. Every replay frame matches its native record.', '',
    'Steering comes from the native route controller. Policies choose speed only using the existing observation interface. This is not CARLA, a new trained policy, realistic motorway validation or evidence of real-world safety. One scenario per map cannot establish general superiority. A run can remain active indefinitely if it keeps moving without reaching a terminal condition.', '',
    f'Release SHA256: `{sha(ROOT/"release.json")}`. No earlier failed audits or scientific outputs were modified. This version logs every physics tick, so its progress reconstruction does not require the preceding demonstration posthoc collision-peak allowance.', '',
    '[Independent audit](run_v1/audit.json) · [Protocol](protocol.json) · [Replay](CITY_REPLAY.html) · [Delivery validation](DELIVERY_VALIDATION.json)']
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(ROOT/'DELIVERY_VALIDATION.json').write_text(json.dumps({'status':'pass','maps':3,'policies':5,'traffic_levels':2,'runs':30,'displayed_decisions':total,
    'new_native_episodes':10,'preserved_recordings':20,'new_training_interactions':0,'all_30_combinations_present':True,
    'release_and_output_hashes':'all match','scientific_validation':'strict_trace_audit_pass',
    'replay_frames_match_source':True,'source_decisions':source_total,'finish_contact_checks':finish_checks,
    'finish_geometry_sha256':sha(ROOT/'finish_geometry.py'),'native_outcome_classifications_unchanged':True,
    'replays':checked,'html_sha256':sha(ROOT/'CITY_REPLAY.html'),'report_sha256':sha(ROOT/'RESULTS.md')},indent=2)+'\n')
print('SITE_VERIFIED',len(catalog['runs']),total)
