"""Line crossing trackers."""
from collections import deque, defaultdict
from enum import Enum
from typing import Optional, Sequence, List, Tuple, Union, Dict
import random
import math
from collections import deque
import numpy as np
import numba
from numba import jit, njit

from savant_rs.primitives.geometry import (
    PolygonalArea,
    Segment,
    IntersectionKind,
    Point,
)


class Direction(Enum):
    entry = 0
    exit = 1


class Movement(Enum):
    idle = 'idle'
    moving = 'moving'


class TwoLinesCrossingTracker:
    """Determines the direction based on the order in which two lines are crossed.
    This is more reliable method in the case of a line at the frame boundary due to
    the jitter of the detected bounding box.
    """

    def __init__(self, area: PolygonalArea):
        self._area = area
        self._prev_cross_edge_label = {}
        self._track_last_points = defaultdict(lambda: deque(maxlen=2))

    def remove_track(self, track_id: int):
        if track_id in self._track_last_points:
            del self._track_last_points[track_id]

    def add_track_point(self, track_id: int, point: Point):
        self._track_last_points[track_id].append(point)

    def check_tracks(self, track_ids: Sequence[int]) -> List[Optional[Direction]]:
        ret = [None] * len(track_ids)
        check_track_idxs = []
        segments = []
        for i, track_id in enumerate(track_ids):
            track_points = self._track_last_points[track_id]
            if len(track_points) == 2:
                segments.append(Segment(*track_points))
                check_track_idxs.append(i)

        cross_results = self._area.crossed_by_segments(segments)

        for cross_result, track_idx in zip(cross_results, check_track_idxs):
            if cross_result.kind in (IntersectionKind.Inside, IntersectionKind.Outside):
                continue

            track_id = track_ids[track_idx]
            cross_edge_labels = [edge[1] for edge in cross_result.edges]

            if cross_result.kind == IntersectionKind.Enter:
                self._prev_cross_edge_label[track_id] = cross_edge_labels
                continue

            if cross_result.kind == IntersectionKind.Leave:
                if track_id in self._prev_cross_edge_label:
                    cross_edge_labels = (
                        self._prev_cross_edge_label[track_id] + cross_edge_labels
                    )

            cross_edge_labels = list(filter(lambda x: x is not None, cross_edge_labels))

            if cross_edge_labels == ['from', 'to']:
                ret[track_idx] = Direction.entry

            elif cross_edge_labels == ['to', 'from']:
                ret[track_idx] = Direction.exit

        return ret


class IdleObjectTracker:
    def __init__(self, idle_tracker_buffer: int = 900, idle_distance_threshold: float = 20):
        self.history = {}
        self.idle_tracker_buffer = idle_tracker_buffer
        self.idle_distance_threshold = idle_distance_threshold

    def update(self, track_id: int, object_coordinates: Tuple[float, float]):
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.idle_tracker_buffer)
        self.history[track_id].append(object_coordinates)

    def remove_track(self, track_id: int):
        if track_id in self.history:
            del self.history[track_id]

    def check_idle(self, track_ids: Union[int, List[int]]) -> Union[str, List[str]]:
        if isinstance(track_ids, int):
            track_ids = [track_ids]

        results = [None] * len(track_ids)
        for track_idx, track_id in enumerate(track_ids):
            if track_id in self.history:
                object_history = list(self.history[track_id])
                reference_coordinates = object_history[0]
                is_idle = all(
                    self.calculate_distance(reference_coordinates, coordinate) <= self.idle_distance_threshold
                    for coordinate in object_history[1:]
                )
                results[track_idx] = Movement.idle.name if is_idle else Movement.moving.name
            else:
                results[track_idx] = Movement.moving.name

        return results

    def calculate_distance(self, coordinates1: Tuple[float, float], coordinates2: Tuple[float, float]) -> float:
        x1, y1 = coordinates1
        x2, y2 = coordinates2
        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        return distance


class CrowdTracker:
    def __init__(self, crowd_area: List[int]):
        self.crowd_area = np.array(crowd_area, dtype=np.float32).reshape((-1, 2))
        self.people_coordinates = []

    def update(self, point: Tuple[int]):
        self.people_coordinates.append(point)

    def check_crowd(self, threshold: int = 20) -> bool:
        people_coordinates = np.array(self.people_coordinates, dtype=np.float32)
        if people_coordinates.size > 0:
            counts = is_inside_postgis_parallel(
                people_coordinates,
                self.crowd_area
            )
            count = sum(counts)
        else:
            count = 0

        is_crowded = count >= threshold
        self.people_coordinates = []
        return is_crowded


@jit(nopython=True)
def is_inside_postgis(polygon, point):
    length = len(polygon)
    intersections = 0

    dx2 = point[0] - polygon[0][0]
    dy2 = point[1] - polygon[0][1]
    ii = 0
    jj = 1

    while jj < length:
        dx = dx2
        dy = dy2
        dx2 = point[0] - polygon[jj][0]
        dy2 = point[1] - polygon[jj][1]

        F = (dx - dx2) * dy - dx * (dy - dy2)
        if F == 0.0 and dx * dx2 <= 0 and dy * dy2 <= 0:
            return 2

        if (dy >= 0 and dy2 < 0) or (dy2 >= 0 and dy < 0):
            if F > 0:
                intersections += 1
            elif F < 0:
                intersections -= 1

        ii = jj
        jj += 1

    return intersections != 0


@njit(parallel=True)
def is_inside_postgis_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean) 
    for i in numba.prange(ln):
        D[i] = is_inside_postgis(polygon, points[i])
    return D

class RandColorIterator:
    def __init__(self) -> None:
        self.golden_ratio_conjugate = 0.618033988749895
        self.hue = random.random()
        self.saturation = 0.7
        self.value = 0.95

    def __next__(self) -> Tuple[int, int, int, int]:
        self.hue = math.fmod(self.hue + 0.618033988749895, 1)
        return hsv_to_rgb(self.hue, self.saturation, self.value) + (255,)


def hsv_to_rgb(h, s, v) -> Tuple[int, int, int]:
    """HSV values in [0..1]
    returns [r, g, b] values in [0..1]
    """
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q
    return int(255 * r), int(255 * g), int(255 * b)
