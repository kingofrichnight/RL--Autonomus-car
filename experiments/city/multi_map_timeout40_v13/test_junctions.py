import numpy as np
import pytest
from city_env import CityRouteEnv, ForwardOnlyIDMVehicle
from audit_check import assert_npc_sweeps_clear


@pytest.mark.parametrize('swap',[False, True])
def test_crossing_pair_both_clear_without_contact_or_reversing(swap):
    env=CityRouteEnv({'map_id':'jam','traffic':False}); env.reset(seed=1)
    road=env.road; env.vehicle.position=np.array([-100.,-100.]); env.vehicle.speed=0.;env.vehicle.target_speed=0.
    a=ForwardOnlyIDMVehicle(road,[28.,0.],0.,3.5,target_lane_index=env.route_indices[0],target_speed=3.5,route=list(env.route_indices),enable_lane_change=False)
    b=ForwardOnlyIDMVehicle(road,[43.,-15.],np.pi/2,4.,target_lane_index=env.avenues[0],target_speed=4.,route=[env.avenues[0]],enable_lane_change=False)
    a.city_id,b.city_id=(1,0) if swap else (0,1)
    road.vehicles=[env.vehicle,a,b]
    yielding=set()
    for _ in range(450):
        road.act(); road.step(1/15)
        yielding.update(v.city_id for v in (a,b) if v.is_yielding)
        assert not a.crashed and not b.crashed and a.speed>=0 and b.speed>=0
        g=road.guard_physics_record; assert_npc_sweeps_clear(g['before'],g['after'])
    assert a.position[0]>60 and b.position[1]>20
    assert yielding
    env.close()


def test_held_npc_stops_at_line_then_restarts_and_ego_is_not_overridden():
    env=CityRouteEnv({'map_id':'jam','traffic':False}); env.reset(seed=1)
    road=env.road
    npc=ForwardOnlyIDMVehicle(road,[37.99,0.],0.,4.,target_lane_index=env.route_indices[0],target_speed=4.,route=list(env.route_indices),enable_lane_change=False)
    npc.city_id=0; npc.junction_stop_distance=.01; npc.action={'steering':0.,'acceleration':-6.}
    npc.step(1/15); assert npc.position[0]<=38.+1e-9 and npc.speed>=0
    npc.junction_stop_distance=None; npc.action={'steering':0.,'acceleration':2.}
    for _ in range(15): npc.step(1/15)
    assert npc.position[0]>38.5
    road.vehicles=[env.vehicle,npc]
    env.vehicle.act('SLOWER'); action=env.vehicle.action.copy();target=env.vehicle.target_speed
    road.junctions.update();road.junctions.brake()
    assert env.vehicle.action==action and env.vehicle.target_speed==target
    assert not hasattr(env.vehicle,'junction_stop_distance')
    env.close()


def test_spawn_never_starts_in_conflict_box_and_exit_blockage_yields():
    env=CityRouteEnv({'map_id':'jam','density':'heavy'});env.reset(seed=74042)
    assert all(not env.road.junctions.contains(v.position) for v in env.road.vehicles[1:])
    lane=env.route_lanes[0]
    a=ForwardOnlyIDMVehicle(env.road,[35.,0.],0.,1.8,target_lane_index=env.route_indices[0],target_speed=1.8)
    b=ForwardOnlyIDMVehicle(env.road,[55.,0.],0.,0.,target_lane_index=env.route_indices[0],target_speed=0.)
    a.city_id=0;b.city_id=1;env.road.vehicles=[env.vehicle,a,b]
    zone=next(z for z in env.road.junctions.zones if np.all(z['lo']<[45.,0.]) and np.all(z['hi']>[45.,0.]))
    assert not env.road.junctions.exit_clear(a,lane,zone['intervals'][env.route_indices[0]])
    env.road.junctions.update();assert a.is_yielding
    env.close()


@pytest.mark.parametrize('density',['medium','heavy'])
def test_crowded_junctions_remain_collision_free_and_live(density):
    env=CityRouteEnv({'map_id':'jam','density':density});env.reset(seed=74042)
    road=env.road;env.vehicle.position=np.array([-100.,-100.]);env.vehicle.speed=0.;env.vehicle.target_speed=0.
    initial={v.city_id:v.position.copy() for v in road.vehicles[1:]}
    for _ in range(900):
        road.act();road.step(1/15)
        g=road.guard_physics_record
        assert not any(v[-1] for v in g['after'])
        assert_npc_sweeps_clear(g['before'],g['after'])
    distances=[np.linalg.norm(v.position-initial[v.city_id]) for v in road.vehicles[1:]]
    assert min(distances)>5., distances
    env.close()
