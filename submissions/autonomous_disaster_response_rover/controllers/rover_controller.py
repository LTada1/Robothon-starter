from __future__ import annotations

from collections.abc import Mapping, Sequence

try:
    from utils.config import (
        EMERGENCY_DISTANCE,
        MAX_WHEEL_ACCEL,
        MAX_WHEEL_SPEED,
        ROVER_SPEED,
        SAFE_DISTANCE,
        TARGET_REACHED_RADIUS,
        TURN_GAIN,
    )
    from utils.geometry import angle_difference, calculate_heading, distance_between_points
    from utils.mujoco_helpers import set_wheel_velocity
except ImportError:
    from ..utils.config import (
        EMERGENCY_DISTANCE,
        MAX_WHEEL_ACCEL,
        MAX_WHEEL_SPEED,
        ROVER_SPEED,
        SAFE_DISTANCE,
        TARGET_REACHED_RADIUS,
        TURN_GAIN,
    )
    from ..utils.geometry import angle_difference, calculate_heading, distance_between_points
    from ..utils.mujoco_helpers import set_wheel_velocity


Point = Sequence[float]


def compute_heading_error(rover_position: Point, rover_heading: float, target_position: Point) -> float:
    target_heading = calculate_heading(rover_position, target_position)
    return angle_difference(target_heading, rover_heading)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _obstacle_speed_scale(obstacle_distances: Mapping[str, float] | None) -> float:
    if not obstacle_distances:
        return 1.0

    front_distance = float(obstacle_distances.get("front", SAFE_DISTANCE))
    if front_distance <= EMERGENCY_DISTANCE:
        return 0.0
    if front_distance >= SAFE_DISTANCE:
        return 1.0
    return (front_distance - EMERGENCY_DISTANCE) / (SAFE_DISTANCE - EMERGENCY_DISTANCE)


def _avoidance_turn(obstacle_distances: Mapping[str, float] | None) -> float:
    if not obstacle_distances:
        return 0.0

    front = float(obstacle_distances.get("front", SAFE_DISTANCE))
    left = float(obstacle_distances.get("left", SAFE_DISTANCE))
    right = float(obstacle_distances.get("right", SAFE_DISTANCE))

    if front > SAFE_DISTANCE:
        return 0.0

    if front <= EMERGENCY_DISTANCE:
        return -1.0 if left > right else 1.0

    clearance_bias = _clamp((left - right) / max(SAFE_DISTANCE, 0.001), -1.0, 1.0)
    return -clearance_bias


def compute_wheel_commands(
    rover_position: Point,
    rover_heading: float,
    target_position: Point,
    obstacle_distances: Mapping[str, float] | None = None,
) -> tuple[float, float]:
    distance_to_target = distance_between_points(rover_position, target_position)
    if distance_to_target <= TARGET_REACHED_RADIUS:
        return 0.0, 0.0

    heading_error = compute_heading_error(rover_position, rover_heading, target_position)
    speed_scale = _obstacle_speed_scale(obstacle_distances)
    heading_scale = max(0.25, 1.0 - min(abs(heading_error), 1.5) / 1.5)
    base_speed = ROVER_SPEED * speed_scale * heading_scale

    if speed_scale == 0.0:
        avoidance = _avoidance_turn(obstacle_distances)
        turn_command = MAX_WHEEL_SPEED * 0.32 * (1.0 if avoidance >= 0.0 else -1.0)
        return _clamp(turn_command, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED), _clamp(
            -turn_command, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED
        )

    steering = _clamp(TURN_GAIN * heading_error, -MAX_WHEEL_SPEED * 0.65, MAX_WHEEL_SPEED * 0.65)
    steering += MAX_WHEEL_SPEED * 0.20 * _avoidance_turn(obstacle_distances)

    left_speed = base_speed - steering
    right_speed = base_speed + steering

    return (
        _clamp(left_speed, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED),
        _clamp(right_speed, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED),
    )


def apply_control(model, data, left_wheel_velocity: float, right_wheel_velocity: float) -> None:
    current_left = float(data.ctrl[0]) if len(data.ctrl) else 0.0
    current_right = float(data.ctrl[2]) if len(data.ctrl) > 2 else current_left

    target_left = _clamp(left_wheel_velocity, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    target_right = _clamp(right_wheel_velocity, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)

    smooth_left = current_left + _clamp(target_left - current_left, -MAX_WHEEL_ACCEL, MAX_WHEEL_ACCEL)
    smooth_right = current_right + _clamp(target_right - current_right, -MAX_WHEEL_ACCEL, MAX_WHEEL_ACCEL)
    set_wheel_velocity(model, data, smooth_left, smooth_right)
