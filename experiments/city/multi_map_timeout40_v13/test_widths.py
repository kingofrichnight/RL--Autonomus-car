"""Width contracts: physical lanes, native termination, independent audit and UI."""
import math
import numpy as np
import pytest
from city_env import MAPS, CityRouteEnv, route_geometry, traffic_geometry
from finish_geometry import touches_finish, valid_finish_contact
from audit_check import finish_contact


@pytest.mark.parametrize('map_id',list(MAPS))
def test_wider_geometry_and_parallel_lane_separation(map_id):
    _,route,_=route_geometry(map_id)
    traffic=traffic_geometry(map_id)
    assert all(l.width==({'district':8,'highway':7,'jam':6}[map_id]) for l in route)
    assert all(l.width==({'district':6,'highway':6,'jam':5}[map_id]) for l in traffic)
    for i,a in enumerate(traffic):
        for b in traffic[i+1:]:
            da=a.end-a.start;db=b.end-b.start
            if abs(da[0]*db[1]-da[1]*db[0])<1e-8:
                delta=b.start-a.start
                separation=abs(da[0]*delta[1]-da[1]*delta[0])/np.linalg.norm(da)
                assert separation>=(a.width+b.width)/2-1e-9


@pytest.mark.parametrize('map_id',list(MAPS))
def test_native_offroute_threshold_tracks_lane_width(map_id):
    env=CityRouteEnv({'map_id':map_id,'traffic':False});env.reset(seed=1)
    lane=env.route_lanes[0]
    # Preserves the existing centre-distance + half-car-width tolerance.
    for lateral,expected in [(lane.width/2+.75,False),(lane.width/2+1,False),(lane.width/2+1.001,True)]:
        env.vehicle.position=lane.position(10,lateral)
        env.vehicle.heading=lane.heading_at(10)
        env._measure_route()
        assert env.off_route==expected
    env.close()


@pytest.mark.parametrize('map_id',list(MAPS))
def test_finish_lateral_boundary_native_and_independent_audit(map_id):
    env=CityRouteEnv({'map_id':map_id,'traffic':False});env.reset(seed=1)
    lane=env.route_lanes[-1];end=lane.position(lane.length,0);h=lane.heading_at(lane.length)
    n=np.array([-math.sin(h),math.cos(h)])
    env.turns_completed=MAPS[map_id]['turns'];env.vehicle.heading=h
    for offset,expected in [(lane.width/2+.75,True),(lane.width/2+1,True),(lane.width/2+1.001,False)]:
        env.vehicle.position=end+offset*n
        pose=[*env.vehicle.position,h]
        assert env.success()==expected
        assert finish_contact(pose,lane)==expected
        assert touches_finish(pose,end,h,lane_width=lane.width)==expected
        scene={'crashed':False,'off_route':False,'turns_completed':1,'required_turns':1,'ego':pose}
        assert valid_finish_contact(scene,end,h,lane_width=lane.width)==expected
    env.close()
