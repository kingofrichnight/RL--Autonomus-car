"""Independent reconstruction of the documented memory/threshold specification."""
def check_assist(row, previous, observation, lanes, memory):
    from highway_env.road.lane import CircularLane
    v=previous['ego'][3];old=previous['ego_target_speed'];progress=previous['route_progress_m']
    distance=1000000.;length=0.
    for lane in lanes:
        if isinstance(lane,CircularLane) and progress<length+lane.length+2:
            distance=max(0.,length-progress);break
        length+=lane.length
    cap=4.5 if distance<=15 else 9.
    index=round(v/4.5)
    ts=[4.5*max(0,min(2,index-1)),old,4.5*max(0,min(2,index+1))]
    threshold=.2 if memory['active'] else .3
    f=observation[106:115].reshape(3,3)
    permitted=[]
    for a in (0,1,2):
        if ts[a]>0 and ts[a]<=cap and f[a,0]>=threshold and f[a,1]>=1:
            permitted.append(a)
    count=min(5,memory['clear_count']+1) if permitted else 0
    active=bool(permitted) and (memory['active'] or count>=5)
    proposed=row['proposed_action'];selected=proposed;reason='model'
    if active:
        best=max(ts[a] for a in permitted)
        tied=[a for a in permitted if ts[a]==best]
        selected=1 if 1 in tied else proposed if proposed in tied else tied[0]
        reason='sustained_progress'
    elif ts[selected]>cap:
        selected=0;reason='turn_target_cap'
    if f[selected,0]<=0 or f[selected,1]<1:
        selected=0;reason='forecast_brake';active=False;count=0
    after={'active':active,'clear_count':count}
    assert row['assist']=={'proposed':proposed,'selected':selected,'changed':selected!=proposed,
        'reason':reason,'target_cap_mps':cap,'turn_distance_m':distance,'targets_mps':ts,'before':memory,'after':after}
    assert row['action']==selected
    return after
