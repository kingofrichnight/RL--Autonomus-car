"""Synthetic clearance regression, no learned model or scored episode."""
import math
import sys
import numpy as np
import pytest
from city_env import CityRouteEnv,ForwardOnlyIDMVehicle,REPOSITORY,CITY_TRAFFIC_OFFSET
sys.path.insert(0,str(REPOSITORY))
from safeintent_rl.sensors.predictive import PredictiveSafetyObservation

@pytest.mark.parametrize('direction',[1,-1])
def test_oncoming_clearance_improves_without_ignoring_actor(direction):
    env=CityRouteEnv({'map_id':'district','traffic':False});env.reset(seed=1)
    ego=env.vehicle;start=ego.position.copy()
    npc=ForwardOnlyIDMVehicle(env.road,start+[20,direction*2.5],math.pi,6.)
    npc.city_id=0;env.road.vehicles=[ego,npc]
    wrapper=PredictiveSafetyObservation(env)
    old=wrapper.forecast()
    npc.position=start+np.array([20,direction*CITY_TRAFFIC_OFFSET]);npc.on_state_update()
    new=wrapper.forecast()
    assert npc in env.road.close_objects_to(ego,200,count=14,see_behind=False)
    assert old[2,0]<0 and old[2,1]<1
    assert new[2,0]>.3 and new[2,1]==1
    # A genuine vehicle in the driven lane must still trigger a conflict.
    npc.position=start+np.array([15,0]);npc.on_state_update()
    conflict=wrapper.forecast()
    assert conflict[2,0]<0 and conflict[2,1]<1
    env.close()

def test_outer_traffic_corridors_do_not_overlap_straight_ego_corridor():
    from city_env import CITY_ROUTE_WIDTH,CITY_TRAFFIC_WIDTH
    assert CITY_TRAFFIC_OFFSET-(CITY_ROUTE_WIDTH+CITY_TRAFFIC_WIDTH)/2==1
