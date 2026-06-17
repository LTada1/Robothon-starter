from __future__ import annotations

import math
from collections.abc import Sequence


Point = Sequence[float]


def distance_between_points(a: Point, b: Point) -> float:
    """Return planar distance between two 2D/3D points."""
    return math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))


def calculate_heading(origin: Point, target: Point) -> float:
    """Return yaw angle from origin to target in radians."""
    return math.atan2(float(target[1]) - float(origin[1]), float(target[0]) - float(origin[0]))


def normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_difference(target_angle: float, current_angle: float) -> float:
    """Return shortest signed angular difference target-current."""
    return normalize_angle(target_angle - current_angle)


def point_in_radius(point: Point, center: Point, radius: float) -> bool:
    return distance_between_points(point, center) <= radius


def field_of_view_check(
    observer_position: Point,
    observer_heading: float,
    target_position: Point,
    max_distance: float,
    fov_radians: float,
) -> bool:
    if distance_between_points(observer_position, target_position) > max_distance:
        return False

    target_heading = calculate_heading(observer_position, target_position)
    heading_error = abs(angle_difference(target_heading, observer_heading))
    return heading_error <= fov_radians * 0.5
