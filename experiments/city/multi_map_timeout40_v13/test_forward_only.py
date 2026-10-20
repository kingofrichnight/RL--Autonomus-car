import math
import numpy as np
import pytest
from city_env import CityRouteEnv,ForwardOnlyIDMVehicle,QueueVehicle

@pytest.mark.parametrize('cls',[ForwardOnlyIDMVehicle,QueueVehicle])
@pytest.mark.parametrize('heading',[0.,math.pi/2,math.pi,-math.pi/2])
def test_brake_stops_without_reverse_and_can_restart(cls,heading):
    env=CityRouteEnv({'map_id':'jam','traffic':False});env.reset(seed=1)
    npc=cls(env.road,[100.,100.],heading,.01,target_speed=0,enable_lane_change=False)
    npc.action={'steering':0.,'acceleration':-6.}
    origin=npc.position.copy()
    npc.step(1/15)
    assert npc.speed==0
    stop=npc.position.copy()
    assert np.dot(stop-origin,[math.cos(heading),math.sin(heading)])>=0
    for _ in range(90):npc.step(1/15)
    np.testing.assert_array_equal(npc.position,stop)
    assert npc.speed==0
    npc.action={'steering':0.,'acceleration':2.}
    for _ in range(15):npc.step(1/15)
    assert npc.speed>0 and np.linalg.norm(npc.position-stop)>.5
    env.close()


def test_defensive_negative_speed_and_impact_preserved():
    env=CityRouteEnv({'map_id':'jam','traffic':False});env.reset(seed=1)
    npc=ForwardOnlyIDMVehicle(env.road,[0.,0.],0.,-2.)
    npc.step(1/15);assert npc.speed>=0 and npc.position[0]==0
    npc.impact=np.array([-1.,0.]);npc.step(1/15)
    assert npc.crashed and npc.position[0]==-1 # Physical impact is not erased.
    env.close()


@pytest.mark.parametrize('map_id',['district','highway','jam'])
@pytest.mark.parametrize('density',['medium','heavy'])
def test_all_npcs_forward_over_original_bug_window(density,map_id):
    env=CityRouteEnv({'map_id':map_id,'density':density});env.reset(seed=74042)
    assert all(isinstance(v,ForwardOnlyIDMVehicle) for v in env.road.vehicles[1:])
    for _ in range(60):
        _,_,terminated,truncated,_=env.step(0)
        for tick in env.physics_trace:
            record=tick['npc_guard']
            for old,new in zip(record['before'],record['after']):
                assert old[4]>=0 and new[4]>=0
                if not new[-1]:assert np.dot(np.array(new[1:3])-old[1:3],[math.cos(old[3]),math.sin(old[3])])>=-1e-9
        if terminated or truncated:break
    env.close()
