from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

LABELS = {"cautious": 0, "normal": 1, "aggressive": 2}


class IntentDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        archive = np.load(path)
        self.x = archive["x"].astype(np.float32)
        self.y = archive["y"].astype(np.int64)
        if self.x.ndim != 3 or self.x.shape[-1] != 6:
            raise ValueError("x must have shape [samples, timesteps, 6]")
        if len(self.x) != len(self.y):
            raise ValueError("x and y must contain the same number of samples")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.int64]:
        return self.x[index], self.y[index]


class TrajectoryCollector:
    """Build labeled fixed-length NPC histories directly from HighwayEnv vehicle states."""

    def __init__(self, history_length: int = 10, sample_stride: int = 2) -> None:
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        if sample_stride <= 0:
            raise ValueError("sample_stride must be positive")
        self.history_length = history_length
        self.sample_stride = sample_stride
        self.histories: dict[int, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=history_length)
        )
        self.previous_velocity: dict[int, np.ndarray] = {}
        self.step_index = 0

    def reset(self) -> None:
        self.histories.clear()
        self.previous_velocity.clear()
        self.step_index = 0

    def observe(self, env: Any) -> tuple[list[np.ndarray], list[int]]:
        self.step_index += 1
        samples: list[np.ndarray] = []
        labels: list[int] = []
        base = env.unwrapped
        ego = getattr(base, "vehicle", None)
        road = getattr(base, "road", None)
        if ego is None or road is None:
            return samples, labels

        ego_pos = np.asarray(ego.position, dtype=np.float32)
        ego_vel = np.asarray(ego.velocity, dtype=np.float32)
        active_ids: set[int] = set()
        for vehicle in road.vehicles:
            if vehicle is ego:
                continue
            label_name = getattr(vehicle, "safeintent_driver_label", None)
            if label_name not in LABELS:
                continue

            key = id(vehicle)
            active_ids.add(key)
            velocity = np.asarray(vehicle.velocity, dtype=np.float32)
            previous = self.previous_velocity.get(key, velocity)
            acceleration = float(np.linalg.norm(velocity - previous))
            rel_pos = np.asarray(vehicle.position, dtype=np.float32) - ego_pos
            rel_vel = velocity - ego_vel
            distance = float(np.linalg.norm(rel_pos))
            feature = np.asarray(
                [rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], acceleration, distance],
                dtype=np.float32,
            )
            self.histories[key].append(feature)
            self.previous_velocity[key] = velocity

            if (
                len(self.histories[key]) == self.history_length
                and self.step_index % self.sample_stride == 0
            ):
                samples.append(np.stack(self.histories[key]))
                labels.append(LABELS[label_name])

        for stale in set(self.histories) - active_ids:
            self.histories.pop(stale, None)
            self.previous_velocity.pop(stale, None)
        return samples, labels


def save_dataset(
    path: str | Path,
    samples: list[np.ndarray],
    labels: list[int],
    episode_ids: list[int] | None = None,
    *,
    metadata: dict[str, int | float | str] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(samples, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 3 or x.shape[-1] != 6:
        raise ValueError("samples must have shape [samples, timesteps, 6]")
    if len(x) != len(y):
        raise ValueError("samples and labels must contain the same number of items")
    if len(y) == 0:
        raise ValueError("at least one trajectory sample is required")
    if np.any((y < 0) | (y >= len(LABELS))):
        raise ValueError("labels must use the cautious/normal/aggressive class indices")
    if episode_ids is None:
        episode_ids = list(range(len(samples)))
    groups = np.asarray(episode_ids, dtype=np.int64)
    if len(groups) != len(y):
        raise ValueError("episode_ids must contain one ID per sample")
    np.savez_compressed(
        output,
        x=x,
        y=y,
        episode_ids=groups,
        label_names=np.asarray(list(LABELS)),
        metadata_json=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    )
