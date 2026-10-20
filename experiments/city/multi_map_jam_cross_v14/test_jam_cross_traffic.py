"""Cross-traffic geometry, entry and priority checks; no learned-policy runs."""
import math
import numpy as np
import pytest
from city_env import MAPS, CityRouteEnv, ForwardOnlyIDMVehicle, jam_crossing_lanes
from highway_env.road.lane import StraightLane


def test_each_jam_turn_has_four_separate_crossing_approaches():
    branches=jam_crossing_lanes()
    assert len(branches)==12
    for x,y in MAPS['jam']['points'][1:-1]:
        here=[]
        for lane in branches:
            if not isinstance(lane,StraightLane):continue
            s,lateral=lane.local_coordinates(np.array([x,y],dtype=float))
            if 0<=s<=lane.length and abs(lateral)<=lane.width/2+1e-9:here.append(lane)
        assert len(here)==4
        assert {tuple(np.round(l.direction,6)) for l in here}=={(1.,0.),(-1.,0.),(0.,1.),(0.,-1.)}


@pytest.mark.parametrize('density',['medium','heavy'])
def test_initial_scene_populates_every_approach_outside_conflict_boxes(density):
    env=CityRouteEnv({'map_id':'jam','density':density,'traffic':True})
    env.reset(seed=MAPS['jam']['seed'])
    try:
        crossing=[v for v in env.road.vehicles if getattr(v,'crossing_traffic',False)]
        assert len(env.crossing_indices)==12
        assert len(crossing)==12
        assert {v.target_lane_index for v in crossing}==env.crossing_indices
        assert all(not env.road.junctions.contains(v.position) for v in crossing)
        assert env.scene()['crossing_traffic_count']==12
        assert env.max_traffic >= len(env.road.vehicles)-1
    finally:env.close()


def test_all_crossing_approaches_participate_in_reservations():
    env=CityRouteEnv({'map_id':'jam','density':'medium','traffic':False})
    env.reset(seed=MAPS['jam']['seed'])
    try:
        reserved={index for zone in env.road.junctions.zones for index in zone['intervals']}
        assert env.crossing_indices <= reserved
        for point in MAPS['jam']['points'][1:-1]:
            matching=[z for z in env.road.junctions.zones if np.linalg.norm((z['lo']+z['hi'])/2-np.array(point))<8]
            assert matching,point
    finally:env.close()


@pytest.mark.parametrize('density',['medium','heavy'])
def test_crossing_npcs_actually_enter_each_junction(density):
    """A drawn lane is insufficient: each turn must see an NPC in its box."""
    env=CityRouteEnv({'map_id':'jam','density':density,'traffic':True})
    env.reset(seed=MAPS['jam']['seed'])
    try:
        road=env.road
        env.vehicle.position=np.array([-100.,-100.]);env.vehicle.speed=0.;env.vehicle.target_speed=0.
        zones=[]
        for point in MAPS['jam']['points'][1:-1]:
            matches=[z for z in road.junctions.zones if np.linalg.norm((z['lo']+z['hi'])/2-np.array(point))<8]
            assert len(matches)==1,point
            zones.append(matches[0])
        entered=set()
        for _ in range(900):
            road.act();road.step(1/15)
            for v in road.vehicles:
                if v is not env.vehicle and getattr(v,'crossing_traffic',False):
                    entered.update(i for i,z in enumerate(zones)
                        if np.all(v.position>=z['lo']) and np.all(v.position<=z['hi']))
            assert not any(v.crashed for v in road.vehicles if v is not env.vehicle)
        assert entered=={0,1,2},(density,entered)
    finally:env.close()


def test_jam_only_ego_priority_is_enabled_and_other_maps_are_unchanged():
    for map_id,expected in [('jam',True),('highway',False),('district',False)]:
        env=CityRouteEnv({'map_id':map_id,'density':'medium','traffic':False})
        env.reset(seed=MAPS[map_id]['seed'])
        try:assert env.road.ego_junction_priority is expected
        finally:env.close()


def test_nearby_ego_gets_priority_and_crossing_npc_yields_without_changing_ego_action():
    env=CityRouteEnv({'map_id':'jam','density':'medium','traffic':False})
    env.reset(seed=MAPS['jam']['seed'])
    try:
        road=env.road;point=MAPS['jam']['points'][1]
        zone=next(z for z in road.junctions.zones if np.linalg.norm((z['lo']+z['hi'])/2-np.array(point))<8)
        # Ego remains on the incoming straight several metres before the
        # controller hands off to its curved turn segment.
        ego_index=env.route_indices[0];ego_lane=road.network.get_lane(ego_index)
        ego_s=114.
        env.vehicle.position=ego_lane.position(ego_s,0);env.vehicle.heading=ego_lane.heading_at(ego_s)
        env.vehicle.speed=4.;env.vehicle.target_speed=4.;env.vehicle.target_lane_index=ego_index
        cross_index=next(i for i in env.avenues[6:] if i in zone['intervals'])
        cross_lane=road.network.get_lane(cross_index);cross_entry=zone['intervals'][cross_index][0]
        npc=ForwardOnlyIDMVehicle(road,cross_lane.position(cross_entry-5,0),cross_lane.heading_at(cross_entry-5),4.5,
            target_lane_index=cross_index,target_speed=4.5,route=[cross_index],enable_lane_change=False)
        npc.city_id=999;npc.crossing_traffic=True;road.vehicles=[env.vehicle,npc]
        env.vehicle.act('IDLE');ego_action=env.vehicle.action.copy();ego_target=env.vehicle.target_speed
        road.junctions.update();road.junctions.brake()
        assert npc.is_yielding and npc.junction_stop_distance is not None
        assert env.vehicle.action==ego_action and env.vehicle.target_speed==ego_target
    finally:env.close()
