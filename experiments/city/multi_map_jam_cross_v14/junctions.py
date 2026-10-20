"""NPC-only, FIFO junction reservations for the city and jam demonstrations.

Keep waiting vehicles outside a conflict box instead of letting the last-tick
collision guard trap two cars nose-to-side. This is a kinematic traffic aid,
not a realistic traffic signal model. Ego actions and motion are never changed.
"""
import numpy as np
from highway_env.road.lane import StraightLane


class JunctionReservations:
    def __init__(self, road, indices, merge_distance=8.):
        self.road = road
        self.zones = []
        lanes = [(i, road.network.get_lane(i)) for i in indices]
        points = []
        for k, (ia, a) in enumerate(lanes):
            if not isinstance(a, StraightLane): continue
            for ib, b in lanes[k+1:]:
                if not isinstance(b, StraightLane): continue
                matrix = np.column_stack([a.direction, -b.direction])
                if abs(np.linalg.det(matrix)) < .5: continue
                sa, sb = np.linalg.solve(matrix, b.start-a.start)
                if 0 <= sa <= a.length and 0 <= sb <= b.length:
                    points.append(a.position(sa, 0))
        # Merge adjacent opposite-direction crossing lanes into one junction.
        groups = []
        for p in points:
            hits = [g for g in groups if any(np.linalg.norm(p-q) <= merge_distance for q in g)]
            merged = [p]
            for g in hits: merged.extend(g)
            groups = [g for g in groups if all(g is not hit for hit in hits)]
            groups.append(merged)
        for group in groups:
            lo = np.min(group, axis=0)-5.
            hi = np.max(group, axis=0)+5.
            intervals = {}
            for index, lane in lanes:
                if isinstance(lane, StraightLane):
                    low, high = 0., lane.length
                    for axis in range(2):
                        if abs(lane.direction[axis]) < 1e-9:
                            if not lo[axis] <= lane.start[axis] <= hi[axis]: low, high = 1., 0.; break
                        else:
                            ends = sorted([(lo[axis]-lane.start[axis])/lane.direction[axis],
                                           (hi[axis]-lane.start[axis])/lane.direction[axis]])
                            low, high = max(low, ends[0]), min(high, ends[1])
                    if high > low: intervals[index] = (low, high)
                else:
                    # Ego routes can use CircularLane turns. Sample their
                    # centreline at <=0.5 m spacing so reservation priority
                    # begins before the vehicle body reaches the conflict box.
                    count=max(2,int(np.ceil(lane.length*2)))
                    stations=np.linspace(0.,lane.length,count+1)
                    inside=[s for s in stations if np.all(lane.position(float(s),0)>=lo)
                            and np.all(lane.position(float(s),0)<=hi)]
                    if inside: intervals[index]=(max(0.,inside[0]-.5),min(lane.length,inside[-1]+.5))
            self.zones.append({'lo':lo, 'hi':hi, 'intervals':intervals, 'owner':None, 'waiting':{}})
        self.tick = 0

    def contains(self, position):
        return any(np.all(position >= z['lo']) and np.all(position <= z['hi']) for z in self.zones)

    def update(self):
        self.tick += 1
        npcs = [v for v in self.road.vehicles if v is not self.road.ego_vehicle]
        for v in npcs: v.junction_stop_distance = None; v.is_yielding = False
        for z in self.zones:
            candidates = {}
            for v in npcs:
                index = v.target_lane_index
                if index not in z['intervals']: continue
                lane = self.road.network.get_lane(index)
                s, lateral = lane.local_coordinates(v.position)
                entry, leave = z['intervals'][index]
                if abs(lateral) < 1.5 and entry-18 <= s < leave:
                    candidates[v.city_id] = (v, entry-s, leave-s, lane)
            z['waiting'] = {i:t for i,t in z['waiting'].items() if i in candidates}
            for i in candidates: z['waiting'].setdefault(i, self.tick)
            if z['owner'] not in candidates: z['owner'] = None
            ego_priority=False
            if getattr(self.road,'ego_junction_priority',False):
                ego=self.road.ego_vehicle
                route=list(getattr(ego,'route',[]) or [])
                if ego.target_lane_index not in route:route.insert(0,ego.target_lane_index)
                for lane_index in route:
                    if lane_index not in z['intervals']:continue
                    lane=self.road.network.get_lane(lane_index)
                    s,lateral=lane.local_coordinates(ego.position)
                    entry,leave=z['intervals'][lane_index]
                    probe=float(np.clip(s,entry,leave))
                    path_distance=float(np.linalg.norm(ego.position-lane.position(probe,0)))
                    # Projection onto the upcoming route segment catches the
                    # final metres before a straight lane hands off to a turn.
                    if s<=leave and abs(lateral)<max(1.5,lane.width/2+1.) and path_distance<=max(5.,2*ego.speed):
                        ego_priority=True;break
                # An ego already inside the box clears first even when its
                # current route segment is curved rather than in intervals.
                ego_priority=ego_priority or bool(np.all(ego.position>=z['lo']) and np.all(ego.position<=z['hi']))
                if ego_priority and z['owner'] in candidates:
                    _,distance,_,_=candidates[z['owner']]
                    if distance>=0:z['owner']=None
            if z['owner'] is None and candidates:
                # Vehicles already inside must clear first. Spawn admission avoids
                # starting inside these boxes, but this also handles lane seams.
                occupants = [i for i,(_,d,_,_) in candidates.items() if d < -1e-6]
                eligible = occupants or ([] if ego_priority else [i for i,(v,d,_,lane) in candidates.items()
                    if d <= max(5., 2*v.speed) and self.exit_clear(v, lane, z['intervals'][v.target_lane_index])])
                if eligible: z['owner'] = min(eligible, key=lambda i:(z['waiting'][i], i))
            for i,(v,d,_,_) in candidates.items():
                if i == z['owner'] or d < -1e-6: continue
                v.junction_stop_distance = min(d, v.junction_stop_distance if v.junction_stop_distance is not None else d)
                v.is_yielding = True

    def exit_clear(self, vehicle, lane, interval):
        entry, leave = interval
        for other in self.road.vehicles:
            if other is vehicle: continue
            other_s, lateral = lane.local_coordinates(other.position)
            vehicle_s,_=lane.local_coordinates(vehicle.position)
            heading=lane.heading_at(float(np.clip(vehicle_s,0.,lane.length)))
            direction=np.array([np.cos(heading),np.sin(heading)])
            alignment = abs(np.dot(direction, [np.cos(other.heading),np.sin(other.heading)]))
            if alignment > .7 and abs(lateral) < 2.6 and entry < other_s < leave+8:
                return False
        return True

    def brake(self):
        for v in self.road.vehicles:
            if v is self.road.ego_vehicle: continue
            d = getattr(v, 'junction_stop_distance', None)
            if d is not None:
                # Brake early. The forward-only integrator enforces the stop line
                # if native discrete braking would otherwise overshoot it.
                safe_speed = np.sqrt(2*3.*max(0., d-.25))
                v.action['acceleration'] = min(v.action['acceleration'], 3.*(safe_speed-v.speed))
