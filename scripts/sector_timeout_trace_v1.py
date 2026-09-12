"""Opt-in read-only tracing primitives; no replay or training entry point.

The prospective 111-case runner is NOT released by this module. Historical
factories and observation methods are untouched on disk. Scoped hooks capture
only calls already made by the exact selected environment, including calls on
observation instances recreated at reset. Use only in a single-threaded process.
"""

from __future__ import annotations

import copy
from contextlib import ExitStack
from pathlib import Path
from threading import get_ident
from unittest.mock import patch

import numpy as np
from highway_env.envs.common.observation import KinematicObservation
from highway_env.road.road import Road

from safeintent_rl.envs.intersection import make_intersection_env
from safeintent_rl.sensors.predictive import PredictiveSafetyObservation
from safeintent_rl.sensors.synchronized_predictive import SynchronizedPredictiveObservation
from scripts.run_sector_comparison_v1 import PaddingPredictiveObservation
from scripts.sector_observation_v1 import SectorPredictiveObservation

CONFIG = "configs/intersection_v3_predictive_geometry_v2.yaml"
FEATURES = ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
_ACTIVE_CAPTURE = None


def make_diagnostic_env(arm: str, config_path: str | Path = CONFIG, *, bootstrap_seed=None):
    """Construct each original stack, preserving any inner factory bootstrap.

    The future released replay must explicitly pass bootstrap_seed=40042.
    No bootstrap is requested by default for isolated engineering tests.
    """
    if arm not in ("geometry", "padding", "sector"):
        raise ValueError("Unknown diagnostic arm")
    inner = make_intersection_env(config_path=config_path, seed=bootstrap_seed)
    try:
        if type(inner) is not PredictiveSafetyObservation:
            raise ValueError("Requires the historical predictive configuration")
        if arm == "geometry":
            return inner
        synchronized = SynchronizedPredictiveObservation(inner)
        wrapper = PaddingPredictiveObservation if arm == "padding" else SectorPredictiveObservation
        return wrapper(synchronized)
    except BaseException:
        inner.close()
        raise


def captured_predict(model, observation):
    """Capture the categorical distribution from the one delegated prediction.

    No second inference, sampling or alternate action. Restore the exact original
    attribute presence even on failure; SB3 reuses its distribution object, so
    copy its probabilities immediately. This helper never steps an environment.
    """
    distributions = []
    original = model.policy.get_distribution

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        probabilities = result.distribution.probs.detach().cpu().numpy().copy()
        distributions.append(probabilities)
        return result

    with patch.object(model.policy, "get_distribution", capture):
        result = model.predict(observation, deterministic=True)
    if len(distributions) != 1:
        raise ValueError("Exactly one categorical inference is required")
    probabilities = distributions[0]
    action = np.asarray(result[0])
    if (probabilities.shape != (1, 3) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-6)
            or action.shape != () or not np.issubdtype(action.dtype, np.integer)
            or int(action) != int(probabilities[0].argmax())):
        raise ValueError("Categorical probabilities do not match the deterministic action")
    return result, probabilities[0].tolist()


class ObservationCapture:
    """Copy existing native/forecast events; never refresh observations for tracing.

    IDs are strong-reference, episode-first-sighting identities, with ego zero,
    then road-list order. They are diagnostic-only, NOT cross-arm identities.
    frame() accepts the info returned with its observation; on terminal returns,
    call frame() without another prediction. Returned records are detached copies.
    """

    def __init__(self, env):
        self.env = env
        self.base = env.unwrapped
        wrappers = {PredictiveSafetyObservation: "geometry",
                    PaddingPredictiveObservation: "padding",
                    SectorPredictiveObservation: "sector"}
        if type(env) not in wrappers:
            raise TypeError("Requires an exact historical diagnostic wrapper stack")
        self.arm = wrappers[type(env)]
        self.predictive = env if self.arm == "geometry" else env.env.env
        if (type(self.predictive) is not PredictiveSafetyObservation
                or (self.arm != "geometry"
                    and type(env.env) is not SynchronizedPredictiveObservation)):
            raise TypeError("Historical diagnostic wrapper stack differs")
        self.size = 115 if self.arm == "geometry" else 163
        self._stack = None
        self._thread = None
        self._event = None
        self._birth_ids = {}
        self.events = []

    def _check_thread(self):
        if self._thread != get_ident():
            raise RuntimeError("Tracing requires one owning thread")

    def reset_frame(self, *, new_episode=False):
        self._check_thread()
        if self._event is not None:
            raise RuntimeError("Cannot reset a capture during a native call")
        self.events = []
        if new_episode:
            self._birth_ids = {}

    def _birth(self, actor):
        if actor not in self._birth_ids:
            self._birth_ids[actor] = len(self._birth_ids)
        return self._birth_ids[actor]

    def _actors(self, selected):
        self._birth(self.base.vehicle)
        for actor in [*self.base.road.vehicles, *self.base.road.objects]:
            self._birth(actor)
        return [{"birth_id": self._birth(actor),
                 "position": np.asarray(actor.position).copy().tolist(),
                 "velocity": np.asarray(actor.velocity).copy().tolist(),
                 "heading": float(actor.heading),
                 "length": float(actor.LENGTH), "width": float(actor.WIDTH)}
                for actor in selected]

    def __enter__(self):
        global _ACTIVE_CAPTURE
        if _ACTIVE_CAPTURE is not None or self._stack is not None:
            raise RuntimeError("Overlapping observation captures are forbidden")
        _ACTIVE_CAPTURE = self
        self._thread = get_ident()
        stack = self._stack = ExitStack()
        observe = KinematicObservation.observe
        normalize = KinematicObservation.normalize_obs
        forecast = PredictiveSafetyObservation.forecast
        select = Road.close_objects_to

        def capture_observe(observer):
            if observer.env is not self.base:
                return observe(observer)
            self._check_thread()
            if (observer.features != FEATURES or observer.order != "sorted"
                    or observer.absolute or not observer.normalize or not observer.clip
                    or observer.vehicles_count != 15 or observer.observe_intentions):
                raise ValueError("Historical native observation contract differs")
            event = {"kind": "native", "time": float(self.base.time)}
            return self._call(event, lambda: observe(observer), "normalized_rows")

        def capture_normalize(observer, df):
            if observer.env is not self.base or self._event is None:
                return normalize(observer, df)
            self._check_thread()
            if self._event["kind"] != "native" or "raw_rows" in self._event:
                raise ValueError("Unexpected normalization call sequence")
            raw = df.to_numpy(copy=True)
            ranges = copy.deepcopy(observer.features_range)
            if not ranges or list(df.columns) != FEATURES or not np.isfinite(raw).all():
                raise ValueError("Requires fixed finite native normalization inputs")
            clipped = np.zeros(raw.shape, dtype=bool)
            for index, name in enumerate(FEATURES):
                if name in ranges:
                    low, high = ranges[name]
                    clipped[:, index] = (raw[:, index] < low) | (raw[:, index] > high)
            self._event.update(raw_rows=raw.tolist(), clipped_mask=clipped.tolist(),
                               normalization_ranges=ranges)
            return normalize(observer, df)

        def capture_forecast(wrapper):
            if wrapper is not self.predictive:
                return forecast(wrapper)
            self._check_thread()
            event = {"kind": "forecast", "time": float(self.base.time)}
            return self._call(event, lambda: forecast(wrapper), "values")

        def capture_select(road, *args, **kwargs):
            result = select(road, *args, **kwargs)
            if road is self.base.road and self._event is not None:
                self._check_thread()
                if "selected_birth_ids" in self._event:
                    raise ValueError("Multiple actor selections in a captured call")
                actors = self._actors(result)
                self._event.update(selected_birth_ids=[item["birth_id"] for item in actors],
                                   selected_actors=actors)
            return result

        try:
            stack.enter_context(patch.object(KinematicObservation, "observe", capture_observe))
            stack.enter_context(patch.object(
                KinematicObservation, "normalize_obs", capture_normalize))
            stack.enter_context(patch.object(
                PredictiveSafetyObservation, "forecast", capture_forecast))
            stack.enter_context(patch.object(Road, "close_objects_to", capture_select))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _call(self, event, delegate, key):
        if self._event is not None:
            raise RuntimeError("Unexpected nested native/forecast call")
        self._event = event
        try:
            result = delegate()
            values = np.asarray(result)
            expected = (15, 7) if key == "normalized_rows" else (3, 3)
            if (values.shape != expected or values.dtype != np.float32
                    or not np.isfinite(values).all() or "selected_birth_ids" not in event):
                raise ValueError("Captured native/forecast output differs from contract")
            event[key] = values.copy().tolist()
            if key == "normalized_rows":
                if "raw_rows" not in event:
                    raise ValueError("Native normalization was not captured")
                event["boundary_mask"] = (np.abs(values) == 1).tolist()
            self.events.append(event)
            return result
        finally:
            self._event = None

    def frame(self, observation, info=None):
        self._check_thread()
        values = np.asarray(observation)
        if (values.shape != (self.size,) or values.dtype != np.float32
                or not np.isfinite(values).all()):
            raise ValueError("Invalid returned policy observation")
        native = [item for item in self.events if item["kind"] == "native"]
        predictions = [item for item in self.events if item["kind"] == "forecast"]
        order = [item["kind"] for item in self.events]
        expected_order = ["native", "forecast"]
        if self.arm != "geometry":
            expected_order.append("native")
        if (order != expected_order
                or any(item["time"] != float(self.base.time) for item in self.events)
                or len(native) != (1 if self.arm == "geometry" else 2) or len(predictions) != 1
                or not np.array_equal(values[:105].reshape(15, 7), native[-1]["normalized_rows"])
                or not np.array_equal(values[106:115].reshape(3, 3), predictions[0]["values"])):
            raise ValueError("Capture does not reproduce the returned observation")
        aligned = native[-1]["selected_birth_ids"] == predictions[0]["selected_birth_ids"]
        if self.arm != "geometry" and not aligned:
            raise ValueError("Synchronized native and forecast identities differ")
        ego = self.base.vehicle
        details = info or {}
        progress = details.get("route_progress", 0.0 if self.base.time == 0 else None)
        state = {"time": float(self.base.time), "speed": float(ego.speed),
                 "target_speed": float(ego.target_speed),
                 "position": np.asarray(ego.position).copy().tolist(),
                 "heading": float(ego.heading), "crashed": bool(ego.crashed),
                 "lane_index": list(ego.lane_index), "route_progress": progress,
                 "remaining_route_fraction": None if progress is None else 1 - float(progress)}
        return copy.deepcopy({"arm": self.arm, "policy_observation": values.tolist(),
                              "native_events": native, "forecast_events": predictions,
                              "native_forecast_ids_match": aligned, "state": state,
                              "call_order": order})

    def __exit__(self, exc_type, exc_value, traceback):
        global _ACTIVE_CAPTURE
        try:
            if self._stack is not None:
                self._stack.close()
        finally:
            self._stack = None
            self._thread = None
            self._event = None
            if _ACTIVE_CAPTURE is self:
                _ACTIVE_CAPTURE = None


if __name__ == "__main__":
    raise SystemExit("Tracing primitives only: the 111-episode replay is not released")
