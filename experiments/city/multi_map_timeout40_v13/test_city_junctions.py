"""City crossing regression fixtures; no learned-policy evaluation or tuning."""
import numpy as np
import pytest
from city_env import CityRouteEnv, ForwardOnlyIDMVehicle
from audit_check import assert_npc_sweeps_clear


def city(density='medium', traffic=False):
    env = CityRouteEnv({'map_id':'district', 'density':density, 'traffic':traffic})
    env.reset(seed=72042)
    return env


def test_city_four_lane_crossings_form_thirty_separate_boxes():
    env = city()
    zones = env.road.junctions.zones
    assert len(zones) == 30
    assert {tuple((z['lo']+z['hi'])/2) for z in zones} == {
        (x,y) for x in [30,90,150,210,270] for y in [30,90,150,210,270,330]}
    for i,z in enumerate(zones):
        assert np.allclose(z['hi']-z['lo'],[26,26])
        assert len(z['intervals']) == 4
        for other in zones[i+1:]:
            assert np.any(z['hi'] < other['lo']) or np.any(other['hi'] < z['lo'])
    env.close()


@pytest.mark.parametrize('swap',[False,True])
def test_eighth_turn_crossing_pair_waits_outside_then_both_clear(swap):
    env=city();road=env.road
    env.vehicle.position=np.array([-100.,-100.]);env.vehicle.speed=0.;env.vehicle.target_speed=0.
    # The same northbound/eastbound lane pair as the observed IDs 68 and 74.
    ni,ei=env.avenues[1],env.avenues[18]
    assert np.allclose(road.network.get_lane(ei).start,[-60,262])
    north=ForwardOnlyIDMVehicle(road,[38.,296.],-np.pi/2,5.,target_lane_index=ni,target_speed=5.,route=[ni],enable_lane_change=False)
    east=ForwardOnlyIDMVehicle(road,[4.,262.],0.,6.,target_lane_index=ei,target_speed=6.,route=[ei],enable_lane_change=False)
    north.city_id,east.city_id=(1,0) if swap else (0,1)
    road.vehicles=[env.vehicle,north,east]
    yields=set()
    for _ in range(450):
        road.act();road.step(1/15)
        yields.update(v.city_id for v in [north,east] if v.is_yielding)
        g=road.guard_physics_record
        assert_npc_sweeps_clear(g['before'],g['after'])
        assert not north.crashed and not east.crashed
        assert north.speed >= 0 and east.speed >= 0
    assert north.position[1] < 245 and east.position[0] > 55
    assert yields
    env.close()


def test_city_admission_downstream_block_and_ego_action_preservation():
    env=city('heavy',True);road=env.road
    assert all(not road.junctions.contains(v.position) for v in road.vehicles[1:])
    index=env.avenues[18];lane=road.network.get_lane(index)
    a=ForwardOnlyIDMVehicle(road,[15.,262.],0.,2.,target_lane_index=index,target_speed=6.)
    b=ForwardOnlyIDMVehicle(road,[45.,262.],0.,0.,target_lane_index=index,target_speed=0.)
    a.city_id=0;b.city_id=1;road.vehicles=[env.vehicle,a,b]
    zone=next(z for z in road.junctions.zones if np.allclose((z['lo']+z['hi'])/2,[30,270]))
    assert not road.junctions.exit_clear(a,lane,zone['intervals'][index])
    env.vehicle.act('SLOWER');action=env.vehicle.action.copy();target=env.vehicle.target_speed
    road.junctions.update();road.junctions.brake()
    assert a.is_yielding and a.junction_stop_distance is not None
    assert env.vehicle.action==action and env.vehicle.target_speed==target
    assert not hasattr(env.vehicle,'junction_stop_distance')
    env.close()


@pytest.mark.parametrize('density',['medium','heavy'])
def test_city_npc_network_advances_without_collisions(density):
    env=city(density,True);road=env.road
    env.vehicle.position=np.array([-100.,-100.]);env.vehicle.speed=0.;env.vehicle.target_speed=0.
    initial={v.city_id:v.position.copy() for v in road.vehicles[1:]}
    for _ in range(900):
        road.act();road.step(1/15)
        g=road.guard_physics_record
        assert not any(v[-1] for v in g['after'])
        assert_npc_sweeps_clear(g['before'],g['after'])
    distances=[np.linalg.norm(v.position-initial[v.city_id]) for v in road.vehicles[1:]]
    assert min(distances)>5.,distances
    env.close()
