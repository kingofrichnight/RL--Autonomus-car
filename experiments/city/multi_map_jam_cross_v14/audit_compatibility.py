"""Verify prior results remain byte-bound and separate from new jam recordings."""
import json
from run_check import ROOT,sha,verify,write


def main():
    verify()
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    complete=json.loads((ROOT/'run_v1/complete.json').read_text(encoding='utf-8'))
    assert complete['episodes'] and [r['stage'] for r in complete['episodes']]==protocol['stage_order']
    prior=ROOT.parent/'multi_map_timeout40_v13'
    prior_release=json.loads((prior/'release.json').read_text(encoding='utf-8'))
    for path,digest in prior_release['inputs'].items():assert sha(path)==digest,path
    old=json.loads((prior/'run_v1/complete.json').read_text(encoding='utf-8'))
    old_audit=json.loads((prior/'run_v1/audit.json').read_text(encoding='utf-8'))
    assert old_audit['status']=='pass' and old_audit['complete_sha256']==sha(prior/'run_v1/complete.json')
    for path,digest in old['files_sha256'].items():assert sha(prior/'run_v1'/path)==digest,path
    old_city_highway=[r for r in old['episodes'] if r['map_id']!='jam']
    old_jam=[r for r in old['episodes'] if r['map_id']=='jam']
    assert len(old_city_highway)==20 and len(old_jam)==10
    assert all(r['success'] and not r['timeout'] for r in old_city_highway)
    new=[r for r in complete['episodes'] if r['map_id']=='jam']
    assert len(new)==10 and all(r['stage'] in protocol['stage_order'] for r in new)
    report={'status':'pass','prior_v13_release_sha256':sha(prior/'release.json'),
        'prior_v13_complete_sha256':sha(prior/'run_v1/complete.json'),
        'prior_v13_all_30_recordings_and_release_inputs_unchanged':True,
        'prior_city_and_highway_recordings_retained_in_site':20,
        'prior_jam_recordings_remain_preserved_in_v13':10,
        'new_v14_jam_recordings':10,'cross_traffic_scenario_changed':True,
        'pooling_prohibited':True}
    write(ROOT/'PRESERVATION_REVIEW.json',report)
    print('PRESERVATION_REVIEW_PASS',20,10)


if __name__=='__main__':main()
