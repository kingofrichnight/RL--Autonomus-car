"""Frozen demonstration speed assist; does not modify any learned model.

Uses only the existing action forecasts, ego speed/target, and navigator turn
distance. Thresholds are engineering choices, not calibrated safety bounds.
"""
import math

def targets(speed, target):
    i=round(speed/4.5)
    return [4.5*max(0,min(2,i-1)), target, 4.5*max(0,min(2,i+1))]

class ProgressAssist:
    def __init__(self):
        self.active=False
        self.clear_count=0

    def decide(self, proposed, speed, target, forecast, turn_distance):
        flat=[speed,target,turn_distance]+[v for row in forecast for v in row]
        if proposed not in (0,1,2) or len(forecast)!=3 or any(len(r)!=3 for r in forecast) or not all(math.isfinite(v) for v in flat):
            raise ValueError('Invalid assist input')
        cap=4.5 if turn_distance<=15 else 9.
        ts=targets(speed,target)
        before={'active':self.active,'clear_count':self.clear_count}
        margin=.2 if self.active else .3
        eligible=[a for a in range(3) if 0<ts[a]<=cap and forecast[a][0]>=margin and forecast[a][1]>=1.]
        if eligible:
            self.clear_count=min(5,self.clear_count+1)
            if self.clear_count>=5:self.active=True
        else:
            self.active=False;self.clear_count=0
        action=proposed;reason='model'
        if self.active:
            # Prefer retaining a target over sending another increment.
            action=max(eligible,key=lambda a:(ts[a],a==1,a==proposed))
            reason='sustained_progress'
        elif ts[action]>cap:
            action=0;reason='turn_target_cap'
        # The assist never knowingly retains a selected predicted conflict.
        if forecast[action][0]<=0 or forecast[action][1]<1.:
            action=0;reason='forecast_brake';self.active=False;self.clear_count=0
        return {'proposed':int(proposed),'selected':int(action),'changed':action!=proposed,
                'reason':reason,'target_cap_mps':cap,'turn_distance_m':turn_distance,
                'targets_mps':ts,'before':before,
                'after':{'active':self.active,'clear_count':self.clear_count}}

def turn_distance(progress, lanes):
    from highway_env.road.lane import CircularLane
    cumulative=0.
    for lane in lanes:
        if isinstance(lane,CircularLane) and progress<cumulative+lane.length+2:
            return max(0.,cumulative-progress)
        cumulative+=lane.length
    return 1000000.
