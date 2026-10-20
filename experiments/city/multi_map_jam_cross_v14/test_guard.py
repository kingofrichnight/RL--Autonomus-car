import numpy as np
import pytest
from city_env import CityRouteEnv,MAPS
from highway_env.vehicle.behavior import IDMVehicle
from traffic_guard import swept_conflicts

def test_swept_crossing_and_clear_parallel():
    assert swept_conflicts([[-5,0],[0,-5]],[[5,0],[0,5]],[0,np.pi/2],[0,np.pi/2])==[(0,1)]
    assert swept_conflicts([[0,0],[0,4]],[[10,0],[10,4]],[0,0],[0,0])==[]

def test_guard_stops_npc_crossing_but_keeps_ego_collision():
    env=CityRouteEnv({'map_id':'jam','traffic':False});env.reset(seed=1)
    road=env.road;env.vehicle.position=np.array([-80.,-80.]);env.vehicle.speed=0
    a=IDMVehicle(road,[-3.,0.],0.,30.);a.city_id=0
    b=IDMVehicle(road,[3.,0.],np.pi,30.);b.city_id=1
    road.vehicles=[env.vehicle,a,b]
    road.step(1/15)
    assert road.guard_total_holds>0 and not a.crashed and not b.crashed
    assert np.linalg.norm(a.position-b.position)>=5.2
    env.vehicle.position=a.position.copy();env.vehicle.speed=0
    road.step(1/15)
    assert env.vehicle.crashed and a.crashed
    env.close()

@pytest.mark.parametrize('map_id',list(MAPS))
def test_guard_traffic_and_json_trace(map_id):
    env=CityRouteEnv({'map_id':map_id,'density':'heavy'});env.reset(seed=MAPS[map_id]['seed'])
    # Engineering constant-action checks, no learned model or outcome selection.
    for _ in range(60):
        _,_,term,trunc,_=env.step(0)
        for tick in env.physics_trace:
            g=tick['npc_guard']
            if not g['ego_crashed']:assert not any(v[-1] for v in g['after'])
        if term or trunc:break
    env.close()


def test_adjacent_opposing_lanes_keep_moving():
    env=CityRouteEnv({'map_id':'district','traffic':False});env.reset(seed=1)
    road=env.road;env.vehicle.position=np.array([-100.,-100.]);env.vehicle.speed=0
    a=IDMVehicle(road,[28.,0.],np.pi/2,5.);a.city_id=0
    b=IDMVehicle(road,[32.,25.],-np.pi/2,5.);b.city_id=1
    road.vehicles=[env.vehicle,a,b]
    for _ in range(90):
        road.step(1/15)
        assert not a.crashed and not b.crashed
    assert road.guard_total_holds==0
    assert a.position[1]>29 and b.position[1]<-4
    env.close()


def test_swept_rectangles_do_not_tunnel_or_ignore_rotation():
    assert swept_conflicts([[-10,0],[10,0]],[[10,0],[-10,0]],[0,0],[0,0])
    assert swept_conflicts([[0,0],[0,3]],[[0,0],[0,3]],[0,0],[np.pi/2,0])
    assert not swept_conflicts([[0,0],[0,4]],[[10,0],[-10,4]],[0,np.pi],[0,np.pi])


def test_independent_audit_rejects_overlap_and_tunneling():
    from audit_check import assert_npc_sweeps_clear
    assert_npc_sweeps_clear([[0,0,0,0],[1,0,4,0]],[[0,10,0,0,False],[1,-10,4,0,False]])
    with pytest.raises(AssertionError):
        assert_npc_sweeps_clear([[0,-10,0,0],[1,10,0,0]],[[0,10,0,0,False],[1,-10,0,0,False]])
    with pytest.raises(AssertionError):
        assert_npc_sweeps_clear([[0,0,0,0],[1,1,0,0]],[[0,0,0,0,False],[1,1,0,0,False]])
