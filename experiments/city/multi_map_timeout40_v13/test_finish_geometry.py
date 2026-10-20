import math
from finish_geometry import touches_finish,valid_finish_contact

def test_first_front_contact():
    assert not touches_finish([-2.50001,0,0],[0,0],0)
    assert touches_finish([-2.5,0,0],[0,0],0)
    assert touches_finish([-1.8,0,0],[0,0],0)
    assert not touches_finish([2.6,0,0],[0,0],0)

def test_finite_line_and_rotated_route():
    assert not touches_finish([-2.5,3.5001,0],[0,0],0)
    assert touches_finish([-2.5,3.5,0],[0,0],0)
    assert touches_finish([10,17.5,math.pi/2],[10,20],math.pi/2)
    assert not touches_finish([10,17.499,math.pi/2],[10,20],math.pi/2)

def test_terminal_precedence_and_all_turns():
    scene={'ego':[-2.5,0,0],'crashed':False,'off_route':False,'turns_completed':4,'required_turns':4}
    assert valid_finish_contact(scene,[0,0],0)
    assert not valid_finish_contact(dict(scene,crashed=True),[0,0],0)
    assert not valid_finish_contact(dict(scene,off_route=True),[0,0],0)
    assert not valid_finish_contact(dict(scene,turns_completed=3),[0,0],0)
