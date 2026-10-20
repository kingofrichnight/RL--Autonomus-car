"""Frozen model loader and dual-expert inference; no environment construction."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
ORIGINAL = BASE / 'action_entropy_runtime_v1'
QR = BASE / 'ppo_to_qrdqn_recovery_v3'
REPOSITORY = Path('C:/Users/hades/OneDrive/Documents/RL--Autonomus-car')
QR_RELEASE = 'feea77a575ba34c54b9c7047478808ca7ecb62ad4a927a718f424d1892ae27bf'
PPO_RELEASE = 'ac05cbad7519d49820ec6cf72bc601487529d367ecd8c77fc92db21926bc6a1e'
QR_HASHES = {42: '662e77bf92cad2051fd58325886a36e962ba53faa24640fe433171d5c7a1b338',
             43: '215a6db167249a0b379286423079f33facc4b1ed152001042b4b7d31c7a86b3a',
             44: '2ba3ebc5f556ec1b79a0eecba54819fe4cad30d24ee2d5172d7a00967a262c7b'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def model_inputs(seed):
    require(type(seed) is int and seed in QR_HASHES, 'unknown model seed')
    release_path = QR / 'study.release.json'
    require(sha(release_path) == QR_RELEASE, 'QR release differs')
    require(sha(ORIGINAL / 'learning.release.json') == PPO_RELEASE, 'PPO release differs')
    release = json.loads(release_path.read_text())
    ppo_path = ORIGINAL / f'pair_{seed}' / 'none' / 'model.zip'
    qr_path = QR / f'pair_{seed}' / 'qr' / 'model.zip'
    ppo_digest = next(d for p, d in release['inputs'].items() if Path(p).resolve() == ppo_path.resolve())
    expected = {str(ppo_path): ppo_digest, str(qr_path): QR_HASHES[seed]}
    for path, digest in expected.items():
        require(sha(path) == digest, 'model bytes differ: ' + path)
    return ppo_path, qr_path, expected


def load_pair(seed):
    ppo_path, qr_path, expected = model_inputs(seed)
    sys.path[:0] = [str(QR / 'vendor'), str(ORIGINAL), str(REPOSITORY)]
    import numpy as np
    from sb3_contrib import QRDQN
    from entropy_ppo_v1 import ActionEntropyPPO
    from policy_fingerprint_v1 import initial_policy_fingerprint
    ppo = ActionEntropyPPO.load(str(ppo_path), device='cpu')
    qr = QRDQN.load(str(qr_path), device='cpu')
    for model in (ppo, qr):
        require(model.observation_space.shape == (163,) and model.observation_space.dtype == np.dtype('float32')
                and model.action_space.n == 3 and model.num_timesteps == 200704,
                'model interface or interaction count differs')
        require(model.safeintent_training_seed == seed, 'seed lineage differs')
        model.policy.set_training_mode(False)
    require(ppo.safeintent_entropy_arm == 'none', 'PPO entropy arm differs')
    require(qr.safeintent_study_release_sha256 == QR_RELEASE and qr._n_updates == 50151
            and qr.policy.n_quantiles == 200, 'QR lineage differs')
    fingerprints = [initial_policy_fingerprint(m) for m in (ppo, qr)]
    for arm, fp in zip(('ppo', 'qr'), fingerprints):
        cert = json.loads((QR / f'pair_{seed}' / arm / ('evaluate.complete.json' if arm == 'ppo' else 'train.complete.json')).read_text())
        require(fp == cert['loaded_policy' if arm == 'ppo' else 'final_policy'], 'policy fingerprint differs')
    return ppo, qr, expected, fingerprints


def infer(ppo, qr, observation):
    import numpy as np
    import torch
    require(isinstance(observation, np.ndarray) and observation.shape == (163,)
            and observation.dtype == np.dtype('float32') and np.isfinite(observation).all(), 'observation contract')
    with torch.inference_mode():
        obs, _ = ppo.policy.obs_to_tensor(observation)
        probabilities = ppo.policy.get_distribution(obs).distribution.probs.cpu().numpy()
        qobs, _ = qr.policy.obs_to_tensor(observation)
        quantiles = qr.policy.quantile_net(qobs).cpu().numpy()
    require(probabilities.shape == (1, 3) and quantiles.shape == (1, 200, 3), 'expert output shapes')
    return probabilities[0].astype(float).tolist(), quantiles[0].T.astype(float).tolist()
