"""Freeze this new exploratory check before executing any learned-policy episode."""
from pathlib import Path
import importlib.metadata
import json
import sys
from city_env import ARCHIVE,REPOSITORY
from run_check import ROOT,write,sha

def main():
    tests=(ROOT/'engineering-tests.txt').read_text()
    assert '[100%]' in tests and 'FAILED' not in tests
    assert (ROOT/'engineering-tests.stderr.txt').stat().st_size==0
    assert not (ROOT/'run_v1').exists()
    original=ARCHIVE/'ppo_qrdqn_hybrid_recovery_v2'
    sys.path.insert(0,str(original))
    from hybrid_models import model_inputs
    _,_,models=model_inputs(42)
    sources=[ROOT/p for p in ['city_env.py','progress_assist.py','audit_assist.py','test_assist.py','test_oncoming.py','test_intersections.py','test_jam_cross_traffic.py','test_widths.py','test_locality.py','junctions.py','test_junctions.py','test_city_junctions.py','traffic_guard.py','finish_geometry.py','test_finish_geometry.py','run_check.py','preflight.py','audit_check.py','audit_compatibility.py','review_control.py','build_site.py','replay_template.html','validate_package.py','summarize_assist.py','test_city.py','test_guard.py','test_forward_only.py','protocol.json','launch_check.ps1','capture_owners.ps1','validate_owners.ps1','seal.py']]
    sources += [original/p for p in ['hybrid_models.py','hybrid_proxy.py','hybrid_rule_v2.py','hybrid_rule.py']]
    sources += [ARCHIVE/'action_entropy_runtime_v1'/p for p in ['partition_metadata_v1.py','action_entropy_math_v1.py','entropy_ppo_v1.py','policy_fingerprint_v1.py']]
    sources += [REPOSITORY/'scripts/sector_observation_v1.py']
    sources += list((REPOSITORY/'safeintent_rl').rglob('*.py'))
    import highway_env
    sources += list(Path(highway_env.__file__).parent.rglob('*.py'))
    predecessor=ROOT.parent/'city_gym_v3'
    sources += [predecessor/p for p in ['release.json','city_env.py']]
    sources += [p for p in (predecessor/'run_v1').iterdir() if p.is_file()]
    for folder,prefix in [('controller_pareto_escape_runtime_v1','cape'),('controller_clearance_gate_study_v1','cage')]:
        sources += [ARCHIVE/folder/(prefix+suffix) for suffix in ['_pure_v1.py','_observation_adapter_v1.py','_policy_proxy_v1.py']]
    sources += [ROOT.parent/'multi_map_gym_v1'/p for p in ['release.json','city_env.py','protocol.json','run_v1/complete.json']]
    sources += [ROOT.parent/'multi_map_flow_v3'/p for p in ['release.json','protocol.json','run_v1/complete.json','run_v1/audit.json']]
    sources += list((ROOT.parent/'multi_map_flow_v3'/'run_v1').glob('*.jsonl'))
    sources += [ROOT.parent/'multi_map_standstill_v2'/p for p in ['release.json','protocol.json','run_v1/complete.json','run_v1/audit.json']]
    sources += [ROOT.parent/'multi_map_forward_v4'/p for p in ['release.json','protocol.json','run_v1/complete.json','run_v1/audit.json']]
    sources += [ROOT.parent/'multi_map_junction_v5'/p for p in ['release.json','protocol.json','run_v1/complete.json','run_v1/audit.json','run_v1/jam_heavy_qr.decisions.jsonl','aggression_review_v1/REPORT.json']]
    sources += [ROOT.parent/'multi_map_locality_v6'/p for p in ['release.json','protocol.json','run_v1/complete.json','run_v1/audit.json']]
    sources += list((ROOT.parent/'multi_map_locality_v6/run_v1').glob('*.jsonl'))
    sources += [ROOT.parent/'multi_map_consistent_v7'/p for p in ['release.json','protocol.json','run_v1/complete.json','run_v1/audit.json','CONTROL_REVIEW.json']]
    sources += [ROOT/p for p in ['engineering-tests.txt','engineering-tests.stderr.txt']]
    sources += [ROOT.parent/'multi_map_wider_v8'/p for p in ['release.json','run_v1/complete.json','run_v1/audit.json','FINAL_VALIDATION.json']]
    sources += [ROOT.parent/'multi_map_intersections_v9'/p for p in ['release.json','protocol.json','run_v1/complete.json','run_v1/audit.json','CONTROL_REVIEW.json']]
    sources += [ROOT.parent/'multi_map_city_grid_v10'/p for p in ['release.json','protocol.json','run_v1/complete.json','run_v1/audit.json','CONTROL_REVIEW.json','FINAL_VALIDATION.json']]
    sources += [ROOT.parent/'multi_map_city_wide_v11'/p for p in ['release.json','protocol.json','run_v1/complete.json','run_v1/audit.json','CONTROL_REVIEW.json','FINAL_VALIDATION.json','REMAINING_STOP_DIAGNOSTIC.json']]
    sources += [ROOT.parent/'multi_map_city_flow_v12'/p for p in ['release.json','protocol.json','run_v1/complete.json','run_v1/audit.json','CONTROL_REVIEW.json','FINAL_VALIDATION.json','catalog.json','PACKAGE_VALIDATION.json']]
    inputs={str(p):sha(p) for p in sources}
    inputs.update(models)
    release={'name':'Populated jam cross traffic and jam-only ego junction priority','inputs':inputs,
        'engineering_tests_sha256':sha(ROOT/'engineering-tests.txt'),
        'versions':{n:importlib.metadata.version(n) for n in ['gymnasium','highway-env','stable-baselines3','numpy','torch']},
        'episode_budget':10,'training_budget':0,'python':sys.version,
        'legacy_preservation':'All historical releases, models and results read-only; this release binds only its own new check.'}
    write(ROOT/'release.json',release)
    print('CITY_RELEASE',sha(ROOT/'release.json'))

if __name__=='__main__':main()
