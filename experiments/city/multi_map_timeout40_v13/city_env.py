"""Ten-turn district, multiple crossing avenues and regulated IDM traffic."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import CircularLane, StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.regulation import RegulatedRoad
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle
from finish_geometry import touches_finish

REPOSITORY = Path('C:/Users/hades/OneDrive/Documents/RL--Autonomus-car')
ARCHIVE = Path('C:/Users/hades/.codex/visualizations/2026/09/26/01a0dbf4-9019-7062-b195-1b7281595a2c')


ROUTE_WIDTH = 6.
TRAFFIC_WIDTH = 5.
CITY_ROUTE_WIDTH = 8.
CITY_TRAFFIC_WIDTH = 6.
CITY_TRAFFIC_OFFSET = 8.

MAPS={
    'district': {'name':'City district · 10 turns','points':[[0,30],[90,30],[90,90],[210,90],[210,210],[150,210],[150,150],[30,150],[30,270],[90,270],[90,330],[270,330]],'radius':10.,'turns':10,'seed':72042},
    'highway': {'name':'Highway connectors','points':[[0,90],[135,90],[135,0],[465,0],[465,120],[600,120]],'radius':10.,'turns':4,'seed':73042},
    'jam': {'name':'Jammed three-turn route','points':[[0,0],[90,0],[90,70],[180,70],[180,150]],'radius':10.,'turns':3,'seed':74042},
}

def route_geometry(map_id='highway'):
    if map_id not in MAPS:raise ValueError('Unknown map')
    points=[np.array(p,dtype=float) for p in MAPS[map_id]['points']]
    # Preserve all original turns; extend only the initial straight by 40 metres.
    points[0]=points[0]-40*(points[1]-points[0])/np.linalg.norm(points[1]-points[0])
    lanes, turn_exits = [], []
    cursor = points[0]
    radius = MAPS[map_id]['radius']
    common = dict(width=CITY_ROUTE_WIDTH if map_id=='district' else 7. if map_id=='highway' else ROUTE_WIDTH, line_types=[LineType.CONTINUOUS, LineType.CONTINUOUS], speed_limit=9)
    for i in range(1, len(points)-1):
        before, corner, after = points[i-1:i+2]
        incoming = (corner-before)/np.linalg.norm(corner-before)
        outgoing = (after-corner)/np.linalg.norm(after-corner)
        start, end = corner-radius*incoming, corner+radius*outgoing
        lanes.append(StraightLane(cursor, start, **common))
        center = start + radius*outgoing
        phase = math.atan2(*(start-center)[::-1])
        sign = float(incoming[0]*outgoing[1]-incoming[1]*outgoing[0])
        lanes.append(CircularLane(center, radius, phase, phase+sign*math.pi/2, clockwise=sign>0, **common))
        turn_exits.append(sum(l.length for l in lanes))
        cursor = end
    lanes.append(StraightLane(cursor, points[-1], **common))
    return points, lanes, turn_exits


def traffic_geometry(map_id='highway'):
    if map_id=='district':
        lanes=[]
        for i,x in enumerate([30,90,150,210,270]):
            speed=[5.,6.5,8.,6.5,5.][i]
            lanes += [StraightLane([x-CITY_TRAFFIC_OFFSET,-60],[x-CITY_TRAFFIC_OFFSET,390],width=CITY_TRAFFIC_WIDTH,speed_limit=speed,priority=2),
                      StraightLane([x+CITY_TRAFFIC_OFFSET,390],[x+CITY_TRAFFIC_OFFSET,-60],width=CITY_TRAFFIC_WIDTH,speed_limit=speed,priority=2)]
        for y in [30,90,150,210,270,330]:
            lanes += [StraightLane([-60,y-CITY_TRAFFIC_OFFSET],[330,y-CITY_TRAFFIC_OFFSET],width=CITY_TRAFFIC_WIDTH,speed_limit=6.,priority=1),
                      StraightLane([330,y+CITY_TRAFFIC_OFFSET],[-60,y+CITY_TRAFFIC_OFFSET],width=CITY_TRAFFIC_WIDTH,speed_limit=6.,priority=1)]
        return lanes
    if map_id=='highway':
        return [StraightLane([-50,y],[650,y],width=6.,speed_limit=8,priority=2) for y in [0,7,13]]+[
            StraightLane([650,y],[-50,y],width=6.,speed_limit=8,priority=2) for y in [-11,-17,-23]]
    if map_id!='jam':raise ValueError('Unknown map')
    lanes=[]
    for x in [45,135]:
        lanes += [StraightLane([x-TRAFFIC_WIDTH/2,-50],[x-TRAFFIC_WIDTH/2,200],width=TRAFFIC_WIDTH,speed_limit=4,priority=2),
                  StraightLane([x+TRAFFIC_WIDTH/2,200],[x+TRAFFIC_WIDTH/2,-50],width=TRAFFIC_WIDTH,speed_limit=4,priority=2)]
    lanes += [StraightLane([-40,110-TRAFFIC_WIDTH/2],[230,110-TRAFFIC_WIDTH/2],width=TRAFFIC_WIDTH,speed_limit=3,priority=1),
              StraightLane([230,110+TRAFFIC_WIDTH/2],[-40,110+TRAFFIC_WIDTH/2],width=TRAFFIC_WIDTH,speed_limit=3,priority=1)]
    return lanes

def intersection_geometry(map_id):
    """Native branch lanes continue straight through every route corner.

    No NPCs spawn on these short alternatives. The prescribed navigator still
    selects the turning path; model actions do not choose a navigation route.
    """
    if map_id=='district':return []  # Existing populated grid already crosses every route corner.
    width=7. if map_id=='highway' else ROUTE_WIDTH
    branches=[]
    for x,y in MAPS[map_id]['points'][1:-1]:
        for start,end in [([x-25,y],[x+25,y]),([x,y-25],[x,y+25]),
                          ([x+25,y-width],[x-25,y-width]),([x-width,y+25],[x-width,y-25])]:
            branches.append(StraightLane(start,end,width=width,speed_limit=4.5))
    return branches


class ForwardOnlyIDMVehicle(IDMVehicle):
    """Traffic brakes to a standstill instead of integrating into reverse.

    Clamp before integration defensively and after each native physics step.
    Keep native impact handling and steering; no positions or crashes erased.
    """
    MIN_SPEED = 0.

    def step(self,dt):
        self.speed=max(0.,float(self.speed))
        distance=getattr(self,'junction_stop_distance',None)
        if distance is not None and not self.crashed and self.impact is None:
            self.speed=min(self.speed,max(0.,distance/dt))
        super().step(dt)
        self.speed=max(0.,float(self.speed))


class QueueVehicle(ForwardOnlyIDMVehicle):
    def act(self,action=None):
        # Regulation can restore lane speed; retain the frozen queue cruise cap.
        self.target_speed=min(self.target_speed,getattr(self,'city_cruise_cap',self.target_speed))
        return super().act(action)


class DistrictRoad(RegulatedRoad):
    def close_objects_to(self,vehicle,distance,count=None,see_behind=True,sort=True,vehicles_only=False):
        # The legacy longitudinal ranking aliases far-away streets on a bend.
        # Use physical proximity for the ego's finite observation on every map.
        # Both native kinematics and predictive forecasts call this same method.
        if getattr(self,'local_ego_observation',False) and vehicle is self.ego_vehicle:
            candidates=super().close_objects_to(vehicle,distance,count=None,see_behind=see_behind,sort=False,vehicles_only=vehicles_only)
            if sort: candidates=sorted(candidates,key=lambda other:float(np.linalg.norm(other.position-vehicle.position)))
            return candidates[:count] if count else candidates
        return super().close_objects_to(vehicle,distance,count=count,see_behind=see_behind,sort=sort,vehicles_only=vehicles_only)

    def act(self):
        coordinator=getattr(self,'junctions',None)
        if coordinator is not None: coordinator.update()
        super().act()
        if coordinator is not None: coordinator.brake()

    def step(self,dt):
        from traffic_guard import guarded_step
        guarded_step(self,dt)

    @staticmethod
    def is_conflict_possible(v1,v2,horizon=3,step=.25):
        # Conservative cull only. Actual regulation is the installed native algorithm.
        reach = horizon*(abs(v1.speed)+abs(v2.speed))+2*max(v1.LENGTH,v2.LENGTH)
        if np.linalg.norm(v1.position-v2.position)>reach:
            return False
        return RegulatedRoad.is_conflict_possible(v1,v2,horizon,step)


class CityRouteEnv(AbstractEnv):
    """Continuous district streets, ten turns, car-following and crossing traffic.

    Steering and route choice belong to the existing HighwayEnv route controller.
    RL chooses only SLOWER/IDLE/FASTER; traffic is seeded and clock-scheduled.
    """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'map_id':'highway', 'turns':4, 'traffic':True, 'density':'heavy', 'duration':None,
            'simulation_frequency':15, 'policy_frequency':5,
            'observation':{'type':'Kinematics','vehicles_count':15,
                'features':['presence','x','y','vx','vy','cos_h','sin_h'],
                'absolute':False,'normalize':True,'order':'sorted','clip':True,
                'see_behind':False,'include_obstacles':True,
                'features_range':{'x':[-200,200],'y':[-200,200],'vx':[-80,80],'vy':[-80,80]}},
            'action':{'type':'DiscreteMetaAction','longitudinal':True,'lateral':False,'target_speeds':[0,4.5,9]},
            'screen_width':1000,'screen_height':800,'scaling':3.,
        })
        return config

    def _reset(self):
        if self.config['simulation_frequency'] != 15 or self.config['policy_frequency'] != 5:
            raise ValueError('This transfer interface requires 15 Hz physics and 5 Hz decisions')
        if self.config['duration'] is not None:
            raise ValueError('This version has no fixed duration; use duration=None')
        if self.config['density'] not in ('medium','heavy'):
            raise ValueError('Density must be medium or heavy')
        self.config['turns']=MAPS[self.config['map_id']]['turns']
        self.points, self.route_lanes, self.turn_exits = route_geometry(self.config['map_id'])
        network = RoadNetwork()
        self.route_indices = [(f'r{i}',f'r{i+1}',0) for i in range(len(self.route_lanes))]
        for index,lane in zip(self.route_indices,self.route_lanes):
            network.add_lane(*index[:2], lane)
        self.traffic_lanes = traffic_geometry(self.config['map_id'])
        self.avenues = [(f't{i}a',f't{i}b',0) for i in range(len(self.traffic_lanes))]
        for index,lane in zip(self.avenues,self.traffic_lanes):
            network.add_lane(*index[:2],lane)
        self.intersection_lanes=intersection_geometry(self.config['map_id'])
        for j,lane in enumerate(self.intersection_lanes):
            network.add_lane(f'b{j}a',f'b{j}b',lane)
        self.road = DistrictRoad(network=network,np_random=self.np_random)
        first = self.route_lanes[0]
        ego = self.action_type.vehicle_class(self.road,first.position(5,0),first.heading_at(5),4.5,
            target_lane_index=self.route_indices[0],target_speed=4.5,route=list(self.route_indices))
        self.vehicle = ego
        self.road.ego_vehicle=ego
        self.road.local_ego_observation=True
        self.road.guard_total_holds=0
        self.road.guard_held_ids=[]
        self.road.guard_physics_record=None
        self.road.vehicles = [ego]
        from junctions import JunctionReservations
        # City opposite traffic centres are 16 m apart. Merge their four
        # crossing points into one conflict box, without joining 60 m-spaced
        # intersections. Ego route geometry is deliberately not an NPC lane.
        self.road.junctions=(JunctionReservations(self.road,self.avenues,merge_distance=18.)
            if self.config['map_id']=='district' else
            JunctionReservations(self.road,self.route_indices+self.avenues)
            if self.config['map_id']=='jam' else None)
        self.cumulative = np.cumsum([0.] + [l.length for l in self.route_lanes])
        self.route_length = float(self.cumulative[-1])
        self.progress = self.previous_progress = 5.
        self.turns_completed = 0
        self.off_route = False
        self.ended = False
        self.standstill_anchor = self.vehicle.position.copy()
        self.last_movement_step = 0
        self.physics_trace = []
        self.spawn_period = 18. if self.config['density']=='medium' else 10.
        self.max_traffic = (44 if self.config['density']=='medium' else 72) if self.config['map_id']=='district' else (40 if self.config['density']=='medium' else 60) if self.config['map_id']=='highway' else (32 if self.config['density']=='medium' else 52)
        self.next_spawn = [self.spawn_period+k*.55 for k in range(len(self.avenues))]
        self.traffic_serial = 0
        self.spawn_blocked = 0
        self.traffic_admissions=[]
        if self.config['traffic']:
            spacing = (110 if self.config['density']=='medium' else 65) if self.config['map_id']=='district' else (90 if self.config['density']=='medium' else 45) if self.config['map_id']=='highway' else (80 if self.config['density']=='medium' else 40)
            for index in self.avenues:
                lane = network.get_lane(index)
                phase = float(self.np_random.uniform(0,30))
                for s in np.arange(15+phase,lane.length-8,spacing):
                    self._spawn(index,float(s))
            if self.config['map_id']=='jam':
                spacing=42 if self.config['density']=='medium' else 20
                for distance in np.arange(70,self.route_length-15,spacing):
                    j=int(np.searchsorted(self.cumulative,distance,side='right')-1)
                    self._spawn_queue(j,float(distance-self.cumulative[j]))
                self.avenues.append(self.route_indices[0])
                self.next_spawn.append(self.spawn_period)

    def _spawn_queue(self,j,s):
        lane=self.route_lanes[j];position=lane.position(s,0)
        if self.road.junctions is not None and self.road.junctions.contains(position):
            self.spawn_blocked+=1;return
        if len(self.road.vehicles)-1>=self.max_traffic or any(np.linalg.norm(v.position-position)<12 for v in self.road.vehicles):
            self.spawn_blocked+=1;return
        cap=3.5 if self.config['density']=='medium' else 1.8
        v=QueueVehicle(self.road,position,lane.heading_at(s),cap,target_lane_index=self.route_indices[j],
            target_speed=cap,route=list(self.route_indices[j:]),enable_lane_change=False)
        v.city_cruise_cap=cap;v.city_id=self.traffic_serial;self.traffic_serial+=1
        self.road.vehicles.append(v)


    def _spawn(self,index,s=0.):
        if index in self.route_indices:return self._spawn_queue(self.route_indices.index(index),40. if s==0 else s)
        lane = self.road.network.get_lane(index)
        position = lane.position(s,0)
        if self.config['map_id']=='district':
            approach_s,approach_lateral=self.route_lanes[0].local_coordinates(position)
            if 0<=approach_s<40 and abs(approach_lateral)<6:
                self.spawn_blocked+=1;return
        if self.road.junctions is not None and self.road.junctions.contains(position):
            self.spawn_blocked+=1;return
        # Admission rule never places a car overlapping another car, including ego.
        if len(self.road.vehicles)-1 >= self.max_traffic or any(np.linalg.norm(v.position-position)<12 for v in self.road.vehicles):
            self.spawn_blocked += 1
            return
        vehicle = ForwardOnlyIDMVehicle(self.road,position,lane.heading_at(s),lane.speed_limit,
            target_lane_index=index,target_speed=lane.speed_limit,route=[index],enable_lane_change=False)
        vehicle.city_id = self.traffic_serial
        self.traffic_serial += 1
        self.road.vehicles.append(vehicle)

    def _measure_route(self):
        index = self.route_indices.index(self.vehicle.target_lane_index)
        # Check adjacent pieces too: the native controller changes target shortly before a seam.
        candidates = []
        for i in range(max(0,index-1),min(len(self.route_lanes),index+2)):
            lane = self.route_lanes[i]
            s,lateral = lane.local_coordinates(self.vehicle.position)
            clamped = float(np.clip(s,0,lane.length))
            distance = float(np.linalg.norm(self.vehicle.position-lane.position(clamped,0)))
            heading_error = abs((self.vehicle.heading-lane.heading_at(clamped)+math.pi)%(2*math.pi)-math.pi)
            candidates.append((distance+heading_error, distance, float(self.cumulative[i]+clamped), lane.width))
        _,distance,progress,lane_width = min(candidates)
        self.off_route = distance > lane_width/2 + self.vehicle.WIDTH/2
        if not self.off_route:
            self.progress = max(self.progress,progress)
        self.turns_completed = int(sum(self.progress >= marker+2 for marker in self.turn_exits))

    def step(self,action):
        if self.ended:
            raise RuntimeError('Episode ended; reset before taking another step')
        if not self.action_space.contains(action):
            raise ValueError('Action must be 0=SLOWER, 1=IDLE or 2=FASTER')
        self.previous_progress = self.progress
        self.traffic_admissions=[]
        result = super().step(action)
        self.ended = bool(result[2] or result[3])
        # Admission belongs before the NEXT observation, never between a saved
        # observation and the action chosen from it. Wrappers forecast this same
        # post-step scene; refresh bare-Gym kinematics as well.
        if not self.ended: self._update_traffic()
        return (self.observation_type.observe(),*result[1:])

    def _update_traffic(self):
        before_ids={v.city_id for v in self.road.vehicles if v is not self.vehicle}
        if self.config['traffic']:
            self.road.vehicles = [v for v in self.road.vehicles if v is self.vehicle or
                self.road.network.get_lane(v.target_lane_index).local_coordinates(v.position)[0] <=
                self.road.network.get_lane(v.target_lane_index).length+10]
            for k,index in enumerate(self.avenues):
                if self.time >= self.next_spawn[k]:
                    self._spawn(index)
                    self.next_spawn[k] += self.spawn_period
        self.traffic_admissions=[v.city_id for v in self.road.vehicles if v is not self.vehicle and v.city_id not in before_ids]

    def _simulate(self,action=None):
        # Stop physics immediately on terminal collision/off-route; no post-impact completion.
        self.physics_trace = []
        for frame in range(3):
            if frame == 0:
                self.action_type.act(action)
            self.road.act()
            self.road.step(1/15)
            self.steps += 1
            self._measure_route()
            if float(np.linalg.norm(self.vehicle.position-self.standstill_anchor)) > .5:
                self.standstill_anchor = self.vehicle.position.copy()
                self.last_movement_step = self.steps
            self.physics_trace.append({'step':self.steps,
                'ego':[float(x) for x in (*self.vehicle.position,self.vehicle.heading,self.vehicle.speed)],
                'target_route_index':self.route_indices.index(self.vehicle.target_lane_index),
                'crashed':bool(self.vehicle.crashed),'npc_guard':self.road.guard_physics_record})
            if self._is_terminated() or self._is_truncated():
                break
        self.time = self.steps/15

    def success(self):
        return bool(not self.vehicle.crashed and not self.off_route and
            self.turns_completed == self.config['turns'] and touches_finish(
                [*self.vehicle.position,self.vehicle.heading],self.route_lanes[-1].position(self.route_lanes[-1].length,0),
                self.route_lanes[-1].heading_at(self.route_lanes[-1].length), lane_width=self.route_lanes[-1].width))

    def _reward(self,action):
        # New route-specific diagnostic reward, never represented as the training reward.
        if self.vehicle.crashed or self.off_route:
            return -10.
        return float((self.progress-self.previous_progress)/10 - .01 + (10 if self.success() else 0))

    def _is_terminated(self):
        return bool(self.vehicle.crashed or self.off_route or self.success())

    def _is_truncated(self):
        return bool(self.steps-self.last_movement_step >= 600 and not self._is_terminated())

    def _info(self,obs,action=None):
        return {'speed':float(self.vehicle.speed),'crashed':bool(self.vehicle.crashed),
            'success':self.success(),'off_route':self.off_route,'turns_completed':self.turns_completed,
            'required_turns':self.config['turns'],'route_progress_m':self.progress,
            'route_length_m':self.route_length,'simulated_seconds':self.steps/15,
            'standstill_seconds':(self.steps-self.last_movement_step)/15,
            'standstill_anchor':[float(x) for x in self.standstill_anchor],
            'last_movement_step':self.last_movement_step,
            'approach_remaining_m':max(0.,40.-self.progress),
            'timeout_reason':'standstill_40s' if self._is_truncated() else None,
            'action':None if action is None else int(action)}

    def scene(self):
        traffic=[v for v in self.road.vehicles if v is not self.vehicle]
        nearest=min((float(np.linalg.norm(v.position-self.vehicle.position)) for v in traffic),default=None)
        return {'ego':[float(x) for x in (*self.vehicle.position,self.vehicle.heading,self.vehicle.speed)],
            'traffic':[[float(x) for x in (*v.position,v.heading,v.speed)] for v in traffic],
            'traffic_ids':[int(v.city_id) for v in traffic],
            'traffic_admissions':list(self.traffic_admissions),
            'traffic_crashed':[bool(v.crashed) for v in traffic],
            'traffic_yielding':[bool(getattr(v,'is_yielding',False)) for v in traffic],
            'traffic_count':len(traffic),'nearest_center_m':nearest,
            'background_crashes':sum(bool(v.crashed) for v in traffic),
            'npc_guard_holds_total':self.road.guard_total_holds,
            'npc_guard_held_ids':self.road.guard_held_ids,
            'ego_target_speed':float(self.vehicle.target_speed),
            'map_id':self.config['map_id'],'density':self.config['density'],'spawn_blocked':self.spawn_blocked,
            **self._info(None)}


def make_city(map_id='highway',traffic=True,render_mode=None,duration=None,density='heavy'):
    sys.path[:0]=[str(REPOSITORY),str(ARCHIVE/'action_entropy_runtime_v1')]
    from safeintent_rl.sensors.predictive import PredictiveSafetyObservation
    from partition_metadata_v1 import wrap_base_env
    return wrap_base_env(PredictiveSafetyObservation(CityRouteEnv({'map_id':map_id,'traffic':traffic,'duration':duration,'density':density},render_mode=render_mode)))

if 'SafeIntent-CityWide-v11' not in gym.registry:
    gym.register('SafeIntent-CityWide-v11',entry_point=make_city)
