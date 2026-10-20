import json
from pathlib import Path
import numpy as np
import pytest
from city_env import CityRouteEnv,ForwardOnlyIDMVehicle,make_city


def reconstruct_bend():
    source=Path(__file__).resolve().parent.parent/'multi_map_junction_v5/run_v1/jam_heavy_qr.decisions.jsonl'
    rows=[json.loads(l) for l in source.read_text().splitlines()]
    scene=next(r['scene'] for r in rows if r['scene']['simulated_seconds']==79.)
    env=CityRouteEnv({'map_id':'jam','traffic':False});env.reset(seed=1)
    e=scene['ego'];env.vehicle.position=np.array(e[:2]);env.vehicle.heading=e[2];env.vehicle.speed=e[3];env.vehicle.on_state_update()
    env.road.vehicles=[env.vehicle]
    for i,p in zip(scene['traffic_ids'],scene['traffic']):
        v=ForwardOnlyIDMVehicle(env.road,p[:2],p[2],p[3]);v.city_id=i;env.road.vehicles.append(v)
    return env


def test_near_lead_car_no_longer_displaced_by_remote_streets():
    env=reconstruct_bend();road=env.road;e=env.vehicle
    road.local_ego_observation=False
    legacy=road.close_objects_to(e,200,count=14,see_behind=False)
    assert 33 not in [v.city_id for v in legacy]
    assert max(np.linalg.norm(v.position-e.position) for v in legacy)>140
    road.local_ego_observation=True
    corrected=road.close_objects_to(e,200,count=14,see_behind=False)
    assert corrected[0].city_id==33
    distances=[np.linalg.norm(v.position-e.position) for v in corrected]
    assert distances==sorted(distances) and len(corrected)==14
    npc=road.vehicles[1]
    before=road.close_objects_to(npc,200,count=14,see_behind=False)
    road.local_ego_observation=False
    assert before==road.close_objects_to(npc,200,count=14,see_behind=False)
    env.close()


def test_independent_auditor_rejects_old_curved_lane_order():
    from audit_check import assert_observation_matches_scene
    env=reconstruct_bend()
    # Scene needs no extra simulation to reconstruct the observation.
    good=env.observation_type.observe().reshape(-1)
    assert_observation_matches_scene(good,env.scene(),env.road.network)
    env.road.local_ego_observation=False
    bad=env.observation_type.observe().reshape(-1)
    with pytest.raises(AssertionError):assert_observation_matches_scene(bad,env.scene(),env.road.network)
    env.close()


@pytest.mark.parametrize('map_id',['district','highway','jam'])
def test_spawn_is_in_observation_before_next_action(map_id):
    env=make_city(map_id=map_id,traffic=False,density='medium');obs,_=env.reset(seed=74042)
    base=env.unwrapped;base.config['traffic']=True
    base.next_spawn=[9999.]*len(base.avenues);base.next_spawn[0]=0.
    obs,*_=env.step(0)
    assert len(base.road.vehicles)==2
    from audit_check import assert_observation_matches_scene
    assert_observation_matches_scene(obs,base.scene(),base.road.network)
    assert not base.physics_trace[-1]['npc_guard']['after'] # admitted after physics
    old_ids=set(base.scene()['traffic_ids'])
    env.step(0)
    assert set(v[0] for v in base.physics_trace[0]['npc_guard']['before'])==old_ids
    env.close()


@pytest.mark.parametrize('map_id',['district','highway','jam'])
def test_every_map_observations_use_identical_proximity_order(map_id):
    from audit_check import assert_observation_matches_scene
    env=make_city(map_id=map_id,density='heavy');obs,_=env.reset(seed=73042)
    base=env.unwrapped
    assert base.road.local_ego_observation
    assert_observation_matches_scene(obs,base.scene(),base.road.network)
    env.close()
