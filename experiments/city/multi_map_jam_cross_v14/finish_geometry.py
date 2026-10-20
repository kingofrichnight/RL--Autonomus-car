"""Shared finite finish contact geometry for native termination and replay validation."""
import math

def touches_finish(pose, endpoint, heading, lane_width=5., length=5., width=2.):
    """A finite transverse finish segment intersects the oriented car rectangle.

    Transform the segment into the car's coordinates and clip it to the body.
    Include boundary contact, but not the empty space beside a narrow road.
    """
    x,y,angle=pose[:3]
    normal=(-math.sin(heading),math.cos(heading))
    ends=[(endpoint[0]+sign*lane_width/2*normal[0],endpoint[1]+sign*lane_width/2*normal[1]) for sign in (-1,1)]
    c,s=math.cos(angle),math.sin(angle)
    local=[((a-x)*c+(b-y)*s,-(a-x)*s+(b-y)*c) for a,b in ends]
    lo,hi=0.,1.
    for axis,half in enumerate((length/2,width/2)):
        start=local[0][axis];delta=local[1][axis]-start
        if abs(delta)<1e-12:
            if abs(start)>half+1e-9:return False
        else:
            a=(-half-start)/delta;b=(half-start)/delta
            lo=max(lo,min(a,b));hi=min(hi,max(a,b))
            if lo>hi+1e-9:return False
    return True

def valid_finish_contact(scene, endpoint, heading, lane_width=5.):
    return bool(not scene['crashed'] and not scene['off_route'] and
        scene['turns_completed']==scene['required_turns'] and
        touches_finish(scene['ego'],endpoint,heading,lane_width=lane_width))
