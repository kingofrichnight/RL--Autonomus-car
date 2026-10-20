import numpy as np
import pytest
from city_env import MAPS,CityRouteEnv,intersection_geometry

@pytest.mark.parametrize('map_id',['highway','jam'])
def test_every_turn_has_native_straight_through_branches(map_id):
    branches=intersection_geometry(map_id)
    assert len(branches)==4*MAPS[map_id]['turns']
    env=CityRouteEnv({'map_id':map_id,'traffic':False});env.reset(seed=1)
    for i,corner in enumerate(MAPS[map_id]['points'][1:-1]):
        for j in (0,1):
            lane=branches[4*i+j]
            along,lateral=lane.local_coordinates(np.array(corner))
            assert along==25 and lateral==0
            assert env.road.network.get_lane((f'b{4*i+j}a',f'b{4*i+j}b',0)).width==lane.width
    env.close()


def test_city_turns_are_at_populated_cross_streets_not_after_them():
    from city_env import traffic_geometry
    traffic=traffic_geometry('district')
    assert intersection_geometry('district')==[]
    for x,y in MAPS['district']['points'][1:-1]:
        vertical=[l for l in traffic if abs(l.start[0]-l.end[0])<1e-8 and abs(l.start[0]-x)==8]
        horizontal=[l for l in traffic if abs(l.start[1]-l.end[1])<1e-8 and abs(l.start[1]-y)==8]
        assert len(vertical)==len(horizontal)==2
        assert all(min(l.start[1],l.end[1])<y<max(l.start[1],l.end[1]) for l in vertical)
        assert all(min(l.start[0],l.end[0])<x<max(l.start[0],l.end[0]) for l in horizontal)
