import pytest
from progress_assist import ProgressAssist

def forecast(m=1.,h=1.):return [[1.,1.,0.],[m,h,.5],[m,h,.8]]

def test_qualification_and_latch_do_not_depend_on_alias():
    a=ProgressAssist()
    for _ in range(4):assert a.decide(0,0,0,forecast(),100)['selected']==0
    assert a.decide(0,0,0,forecast(),100)['selected']==2
    # Next learned brake is overridden, avoiding the old alias-release cycle.
    r=a.decide(0,2.,4.5,forecast(),100)
    assert r['selected']==1 and r['after']['active']

def test_positive_but_lower_clearance_can_qualify():
    a=ProgressAssist()
    for _ in range(5):r=a.decide(0,0,0,forecast(.78),100)
    assert r['selected']==2

def test_hysteresis_and_immediate_loss_of_clearance():
    a=ProgressAssist()
    for _ in range(5):a.decide(0,0,0,forecast(),100)
    assert a.decide(0,2,4.5,forecast(.25),100)['after']['active']
    assert not a.decide(0,2,4.5,forecast(.19),100)['after']['active']
    assert a.clear_count==0

@pytest.mark.parametrize('m,h',[(0,1),(-.1,1),(1,.99)])
def test_forecast_conflict_cancels_acceleration(m,h):
    a=ProgressAssist()
    for _ in range(5):a.decide(2,4.5,4.5,forecast(),100)
    r=a.decide(2,4.5,4.5,forecast(m,h),100)
    assert r['selected']==0 and not r['after']['active']

def test_turn_cap_and_memory_maintains_middle_target():
    a=ProgressAssist()
    assert a.decide(2,9,9,forecast(),15)['selected']==0
    for _ in range(5):r=a.decide(0,4.5,4.5,forecast(),0)
    assert r['selected']==1 and r['targets_mps'][r['selected']]==4.5

def test_reset_is_new_instance_and_invalid_input_rejected():
    a=ProgressAssist()
    for _ in range(5):a.decide(0,0,0,forecast(),100)
    b=ProgressAssist();assert not b.active and b.clear_count==0
    with pytest.raises(ValueError):b.decide(0,float('nan'),0,forecast(),100)

def test_auditor_detects_corrupt_action_and_memory():
    import numpy as np
    from audit_assist import check_assist
    a=ProgressAssist();f=forecast();obs=np.zeros(163);obs[106:115]=np.array(f).ravel()
    pre={'ego':[0,0,0,0],'ego_target_speed':0,'route_progress_m':0}
    receipt=a.decide(0,0,0,f,1000000.)
    row={'assist':receipt,'proposed_action':0,'action':0}
    check_assist(row,pre,obs,[],{'active':False,'clear_count':0})
    row['action']=2
    with pytest.raises(AssertionError):check_assist(row,pre,obs,[],{'active':False,'clear_count':0})
