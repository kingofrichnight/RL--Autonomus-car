"""Deterministic local routing; blocked edges are forbidden, never just expensive."""

from dataclasses import dataclass
from heapq import heappop, heappush
from math import isfinite


@dataclass(frozen=True)
class RouteEdge:
    source: str
    target: str
    length_m: float
    speed_mps: float
    risk_cost_s: float = 0.0
    lane_change_cost_s: float = 0.0
    blocked: bool = False

    def __post_init__(self):
        values = (self.length_m, self.speed_mps, self.risk_cost_s, self.lane_change_cost_s)
        if not all(isfinite(v) for v in values):
            raise ValueError("Route costs must be finite")
        if min(values) < 0 or self.speed_mps == 0:
            raise ValueError("Lengths/costs must be nonnegative and speed positive")
        if type(self.blocked) is not bool:
            raise ValueError("blocked must be boolean")

    @property
    def cost_s(self):
        return self.length_m / self.speed_mps + self.risk_cost_s + self.lane_change_cost_s


def shortest_route(edges: list[RouteEdge], source: str, target: str) -> tuple[str, ...] | None:
    """Dijkstra with lexicographic ties; None explicitly means no feasible path."""
    adjacency = {}
    for edge in edges:
        if not edge.blocked:
            adjacency.setdefault(edge.source, []).append(edge)
    queue = [(0.0, (source,))]
    visited = set()
    while queue:
        cost, path = heappop(queue)
        node = path[-1]
        if node in visited:
            continue
        visited.add(node)
        if node == target:
            return path
        for edge in adjacency.get(node, []):
            if edge.target not in visited:
                heappush(queue, (cost + edge.cost_s, path + (edge.target,)))
    return None
