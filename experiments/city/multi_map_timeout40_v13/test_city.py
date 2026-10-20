import json,math
import numpy as np
import pytest
from city_env import CityRouteEnv,MAPS,route_geometry,make_city,QueueVehicle

@pytest.mark.parametrize('map_id',['district','highway','jam'])
def test_geometry_and_clear_route(map_id):
    _,lanes,markers=route_geometry(map_id)
    assert len(markers)==MAPS[map_id]['turns']
    for a,b in zip(lanes,lanes[1:]):
        np.testing.assert_allclose(a.position(a.length,0),b.position(0,0),atol=1e-8)
        assert abs((a.heading_at(a.length)-b.heading_at(0)+math.pi)%(2*math.pi)-math.pi)<1e-8
    env=CityRouteEnv({'map_id':map_id,'traffic':False})
    env.reset(seed=1)
    for _ in range(900):
        _,_,term,trunc,info=env.step(2)
        if term or trunc:break
    assert info['success'] and info['turns_completed']==MAPS[map_id]['turns']
    with pytest.raises(RuntimeError):env.step(2)
    env.close()

@pytest.mark.parametrize('map_id',['district','highway','jam'])
def test_traffic_density_and_determinism(map_id):
    counts=[]
    for density in ['medium','heavy']:
        a=CityRouteEnv({'map_id':map_id,'density':density});b=CityRouteEnv({'map_id':map_id,'density':density})
        a.reset(seed=MAPS[map_id]['seed']);b.reset(seed=MAPS[map_id]['seed'])
        assert a.scene()==b.scene()
        counts.append(len(a.road.vehicles)-1)
        assert counts[-1]<=a.max_traffic
        for i,v in enumerate(a.road.vehicles):
            for w in a.road.vehicles[i+1:]:assert np.linalg.norm(v.position-w.position)>=12
        if map_id=='jam':
            queues=[v for v in a.road.vehicles if isinstance(v,QueueVehicle)]
            assert queues and all(len(v.route)>0 for v in queues)
            for v in queues:
                v.target_speed=9.;v.act();assert v.target_speed<=v.city_cruise_cap
        a.close();b.close()
    assert counts[1]>counts[0]>0,counts

@pytest.mark.parametrize('map_id',['district','highway','jam'])
def test_interface_forecast_json(map_id):
    env=make_city(map_id=map_id)
    obs,_=env.reset(seed=MAPS[map_id]['seed'])
    assert obs.shape==(163,) and obs.dtype==np.float32
    before=env.unwrapped.scene();env.env.env.env.forecast();assert env.unwrapped.scene()==before
    obs,r,t,u,info=env.step(1)
    json.dumps({'scene':env.unwrapped.scene(),'reward':r,'t':t,'u':u},allow_nan=False)
    assert len(env.action_entropy_partition())==3
    env.close()

@pytest.mark.parametrize('map_id',list(MAPS))
def test_stationary_timeout_and_terminal_precedence(map_id):
    env=CityRouteEnv({'map_id':map_id,'traffic':False})
    env.reset(seed=1);env.vehicle.speed=0;env.vehicle.target_speed=0;env.vehicle.speed_index=0
    for i in range(200):
        _,_,term,trunc,info=env.step(0)
        assert not term and trunc==(i==199)
        assert info['standstill_seconds']==(i+1)/5
    assert info['timeout_reason']=='standstill_40s'
    with pytest.raises(RuntimeError):env.step(1)
    env.vehicle.crashed=True;assert env._is_terminated() and not env._is_truncated()
    assert not env.success() and env._reward(0)==-10
    env.reset(seed=1);assert env.last_movement_step==0 and env.scene()['standstill_seconds']==0
    env.close()

def test_no_total_limit_and_movement_reset():
    env=CityRouteEnv({'map_id':'highway','traffic':False});env.reset(seed=1)
    env.steps=15000;env.time=1000;env.last_movement_step=env.steps
    _,_,term,trunc,info=env.step(2)
    assert not term and not trunc and info['simulated_seconds']>1000
    assert info['standstill_seconds']<.2
    env.close()

@pytest.mark.parametrize('map_id',list(MAPS))
def test_radius_jitter_and_reset_boundary(map_id):
    env=CityRouteEnv({'map_id':map_id,'traffic':False});env.reset(seed=1)
    # Engineering stubs isolate the movement clock, without learned policies.
    env.road.act=lambda:None
    env.action_type.act=lambda action:None
    env.vehicle.speed=0
    start=env.vehicle.position.copy()
    def jitter(dt):env.vehicle.position=start+np.array([.2 if env.steps%2 else -.2,0])
    env.road.step=jitter
    for i in range(199):assert not env.step(1)[3]
    # Movement outside the radius on tick 598 resets rather than timing out.
    env.road.step=lambda dt:setattr(env.vehicle,'position',start+np.array([.51,0]))
    assert not env.step(1)[3] and env.last_movement_step==598
    for i in range(199):assert not env.step(1)[3]
    assert env.step(1)[3] and env.steps==1198
    assert env.scene()['standstill_seconds']==40
    env.close()

def test_fixed_duration_rejected():
    with pytest.raises(ValueError):CityRouteEnv({'duration':180})

@pytest.mark.parametrize('map_id',list(MAPS))
def test_clear_approach_and_preserved_turns(map_id):
    env=CityRouteEnv({'map_id':map_id,'density':'heavy'});env.reset(seed=MAPS[map_id]['seed'])
    entry=np.array(MAPS[map_id]['points'][0],dtype=float)
    assert abs(np.linalg.norm(entry-env.vehicle.position)-35)<1e-9
    assert env.scene()['approach_remaining_m']==35
    first=env.route_lanes[0]
    for vehicle in env.road.vehicles[1:]:
        lng,lat=first.local_coordinates(vehicle.position)
        assert not (0<=lng<40 and abs(lat)<6),'NPC in approach'
    assert len(env.turn_exits)==MAPS[map_id]['turns']
    env.close()


@pytest.mark.parametrize('map_id',list(MAPS))
def test_native_finish_exact_body_contact(map_id):
    env=CityRouteEnv({'map_id':map_id,'traffic':False});env.reset(seed=1)
    lane=env.route_lanes[-1];end=lane.position(lane.length,0);heading=lane.heading_at(lane.length)
    direction=np.array([math.cos(heading),math.sin(heading)])
    env.turns_completed=MAPS[map_id]['turns'];env.vehicle.heading=heading
    env.vehicle.position=end-2.501*direction;assert not env.success()
    env.vehicle.position=end-2.5*direction;assert env.success()
    env.vehicle.crashed=True;assert not env.success()
    env.close()
