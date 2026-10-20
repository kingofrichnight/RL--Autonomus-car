"""Verify the sealed ten-episode demo release before its single launch."""
import argparse
import json
from pathlib import Path
from run_check import ROOT, sha, verify


def check_outputs():
    for name in ('run_v1', 'native.stdout.log', 'native.stderr.log', 'native.exit.json',
                 'native.failure.json', 'run.lock'):
        assert not (ROOT / name).exists(), f'Output already exists: {name}'


def verify_protocol():
    release=verify()
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    expected=[f'jam_{density}_{arm}' for density in ('medium','heavy')
              for arm in ('ppo','qr','hybrid025','cape','cage')]
    assert protocol['stage_order']==expected
    assert protocol['episode_budget']==len(expected)==10
    assert protocol['training_budget']==0
    assert protocol['saved_training_seed']==42
    assert protocol['scenario_seeds']['jam']==74042
    assert (ROOT/'engineering-tests.txt').is_file()
    assert '[100%]' in (ROOT/'engineering-tests.txt').read_text(encoding='utf-8')
    assert 'FAILED' not in (ROOT/'engineering-tests.txt').read_text(encoding='utf-8')
    assert (ROOT/'engineering-tests.stderr.txt').stat().st_size==0
    check_outputs()
    return release,protocol


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    release,protocol=verify_protocol()
    receipt=ROOT/'preflight.json'
    if args.verify_only:
        assert receipt.is_file(), 'Prepared preflight receipt is missing'
        old=json.loads(receipt.read_text(encoding='utf-8'))
        assert old['status']=='pass' and old['release_sha256']==sha(ROOT/'release.json')
        assert old['stage_order']==protocol['stage_order'] and old['episode_budget']==10
    else:
        payload={'status':'pass','release_sha256':sha(ROOT/'release.json'),
                 'input_count':len(release['inputs']),'stage_order':protocol['stage_order'],
                 'episode_budget':10,'training_budget':0,'scenario_seed':74042}
        with receipt.open('x',encoding='utf-8') as f:
            json.dump(payload,f,indent=2,allow_nan=False);f.write('\n');f.flush()
    print('PREFLIGHT_PASS',sha(ROOT/'release.json'),len(protocol['stage_order']))


if __name__=='__main__':main()
