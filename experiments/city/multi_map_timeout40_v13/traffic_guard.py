"""Conservative NPC-only kinematic traffic coordinator; never shields the ego.

After proposing one native physics step, reject conflicting NPC movements before
native contact processing. A rejected NPC remains at its previous pose with zero
speed for that tick. This is an abrupt simulator hold, not realistic braking.
Swept oriented rectangles include a conservative rotation envelope. Parallel lanes
are not blocked by overly wide circular footprints.
The native one-tick collision predictor is checked too. Fixed actor ID priority
breaks ties; holds propagate to approaching NPCs until there are no conflicts.
"""
import math
import numpy as np

def swept_conflicts(before,after,headings,end_headings,length=5.,width=2.,margin=.2):
    """Continuous SAT for translation, expanded to bound per-tick rotation.

    The polygon at the initial orientation plus r*abs(delta_heading) contains
    the rotating body throughout the tick. Intersect all four projection time
    intervals, including t=0 and t=1. A separate native predictor is retained.
    """
    if len(before)<2:return []
    before=np.asarray(before,dtype=float);after=np.asarray(after,dtype=float)
    angles=np.asarray(headings);end_angles=np.asarray(end_headings)
    axes=np.stack([np.column_stack([np.cos(angles),np.sin(angles)]),
                   np.column_stack([-np.sin(angles),np.cos(angles)])],axis=1)
    movement=after-before;r=math.hypot(length,width)/2
    pad=r*np.abs((end_angles-angles+math.pi)%(2*math.pi)-math.pi)
    distance=np.linalg.norm(before[:,None,:]-before[None,:,:],axis=-1)
    reach=2*r+margin+np.linalg.norm(movement,axis=1)[:,None]+np.linalg.norm(movement,axis=1)[None,:]
    candidates=zip(*np.where(np.triu(distance<=reach,1)))
    result=[]
    for i,j in candidates:
        lo,hi=0.,1.
        for axis in np.concatenate([axes[i],axes[j]]):
            size=sum(abs(np.dot(axis,axes[k,0]))*length/2+abs(np.dot(axis,axes[k,1]))*width/2+pad[k] for k in (i,j))+margin
            offset=float(np.dot(before[i]-before[j],axis));delta=float(np.dot(movement[i]-movement[j],axis))
            if abs(delta)<1e-12:
                if abs(offset)>size:lo,hi=1.,0.;break
            else:
                t1=(-size-offset)/delta;t2=(size-offset)/delta
                lo=max(lo,min(t1,t2));hi=min(hi,max(t1,t2))
                if lo>hi:break
        if lo<=hi:result.append((int(i),int(j)))
    return result

def guarded_step(road,dt):
    ego=road.ego_vehicle
    npcs=sorted((v for v in road.vehicles if v is not ego),key=lambda v:v.city_id)
    before=[v.position.copy() for v in npcs]
    headings=[v.heading for v in npcs]
    speeds=[v.speed for v in npcs]
    radii=[math.hypot(v.LENGTH,v.WIDTH)/2 for v in npcs]
    for v in road.vehicles:v.step(dt)
    held=set()
    # An ego-induced impact must never be erased or reclassified as a safe hold.
    if not ego.crashed and not any(v.crashed for v in npcs):
        for _ in range(len(npcs)+1):
            conflicts=swept_conflicts(before,[v.position for v in npcs],headings,[v.heading for v in npcs])
            for i,a in enumerate(npcs):
                for j in range(i+1,len(npcs)):
                    b=npcs[j]
                    if np.linalg.norm(a.position-b.position)>radii[i]+radii[j]+(abs(a.speed)+abs(b.speed))*dt:continue
                    current,future,_=a._is_colliding(b,dt)
                    if current or future:conflicts.append((i,j))
            if not conflicts:break
            added=set()
            for i,j in conflicts:
                if j not in held:added.add(j)
                elif i not in held:added.add(i)
                else:raise RuntimeError('NPC guard found overlapping stationary starting poses')
            for i in added:
                v=npcs[i];v.position=before[i].copy();v.heading=headings[i];v.speed=0.
                v.on_state_update()
            held.update(added)
        else:raise RuntimeError('NPC guard did not converge')
    for i,a in enumerate(road.vehicles):
        for b in road.vehicles[i+1:]:a.handle_collisions(b,dt)
        for obj in road.objects:a.handle_collisions(obj,dt)
    road.guard_held_ids=[npcs[i].city_id for i in sorted(held)]
    road.guard_total_holds+=len(held)
    road.guard_physics_record={'before':[[int(v.city_id),float(p[0]),float(p[1]),float(h),float(speed)] for v,p,h,speed in zip(npcs,before,headings,speeds)],
        'after':[[int(v.city_id),*[float(x) for x in v.position],float(v.heading),float(v.speed),bool(v.crashed)] for v in npcs],
        'junction_stop_distances':[[int(v.city_id),float(v.junction_stop_distance)] for v in npcs if getattr(v,'junction_stop_distance',None) is not None],
        'held_ids':road.guard_held_ids,'ego_crashed':bool(ego.crashed)}
