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


def classify_controller_state(
    rover_position: Point,
    rover_heading: float,
    target_position: Point,
    obstacle_distances: Mapping[str, float] | None = None,
) -> dict:
    distance_to_target = distance_between_points(rover_position, target_position)
    heading_error = compute_heading_error(rover_position, rover_heading, target_position)
    obstacle_distances = obstacle_distances or {}
    front = float(obstacle_distances.get("front", SAFE_DISTANCE * 2.0))
    left = float(obstacle_distances.get("left", SAFE_DISTANCE * 2.0))
    right = float(obstacle_distances.get("right", SAFE_DISTANCE * 2.0))

    if distance_to_target <= TARGET_REACHED_RADIUS:
        state = "TARGET_REACHED"
        avoidance_decision = "stop_at_target"
        speed_scale = 0.0
        steering_bias = 0.0
    elif front <= EMERGENCY_DISTANCE and left <= EMERGENCY_DISTANCE and right <= EMERGENCY_DISTANCE:
        state = "TRAPPED_RECOVERY"
        avoidance_decision = "reverse_turn"
        speed_scale = -0.35
        steering_bias = 1.0
    elif front <= EMERGENCY_DISTANCE:
        state = "BLOCKED"
        turn_left = left >= right
        avoidance_decision = "turn_left_from_block" if turn_left else "turn_right_from_block"
        speed_scale = 0.0
        steering_bias = 1.0 if turn_left else -1.0
    elif front < SAFE_DISTANCE:
        state = "AVOIDING"
        turn_left = left >= right
        avoidance_decision = "veer_left_more_clearance" if turn_left else "veer_right_more_clearance"
        speed_scale = _obstacle_speed_scale(obstacle_distances)
        steering_bias = 0.65 if turn_left else -0.65
    elif min(left, right) < SAFE_DISTANCE * 0.6:
        state = "SIDE_CLEARANCE"
        avoidance_decision = "bias_away_from_left" if left < right else "bias_away_from_right"
        speed_scale = 0.75
        steering_bias = -0.35 if left < right else 0.35
    else:
        state = "TRACKING_TARGET"
        avoidance_decision = "follow_target_heading"
        speed_scale = 1.0
        steering_bias = 0.0

    return {
        "controller_state": state,
        "avoidance_decision": avoidance_decision,
        "heading_error": heading_error,
        "distance_to_target": distance_to_target,
        "front_distance": front,
        "left_distance": left,
        "right_distance": right,
        "speed_scale": speed_scale,
        "steering_bias": steering_bias,
    }


def compute_control_decision(
    rover_position: Point,
    rover_heading: float,
    target_position: Point,
    obstacle_distances: Mapping[str, float] | None = None,
) -> dict:
    decision = classify_controller_state(rover_position, rover_heading, target_position, obstacle_distances)

    if decision["controller_state"] == "TARGET_REACHED":
        left_speed = right_speed = 0.0
    elif decision["controller_state"] == "TRAPPED_RECOVERY":
        left_speed = -MAX_WHEEL_SPEED * 0.45
        right_speed = MAX_WHEEL_SPEED * 0.30
    elif decision["controller_state"] == "BLOCKED":
        turn = MAX_WHEEL_SPEED * 0.36 * (1.0 if decision["steering_bias"] >= 0.0 else -1.0)
        left_speed = -turn
        right_speed = turn
    else:
        heading_error = float(decision["heading_error"])
        heading_scale = max(0.25, 1.0 - min(abs(heading_error), 1.5) / 1.5)
        base_speed = ROVER_SPEED * float(decision["speed_scale"]) * heading_scale
        steering = _clamp(TURN_GAIN * heading_error, -MAX_WHEEL_SPEED * 0.55, MAX_WHEEL_SPEED * 0.55)
        steering += MAX_WHEEL_SPEED * 0.28 * float(decision["steering_bias"])
        left_speed = base_speed - steering
        right_speed = base_speed + steering

    decision["left_wheel_velocity"] = _clamp(left_speed, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    decision["right_wheel_velocity"] = _clamp(right_speed, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    return decision


def compute_wheel_commands(
    rover_position: Point,
    rover_heading: float,
    target_position: Point,
    obstacle_distances: Mapping[str, float] | None = None,
) -> tuple[float, float]:
    decision = compute_control_decision(rover_position, rover_heading, target_position, obstacle_distances)
    return decision["left_wheel_velocity"], decision["right_wheel_velocity"]


def apply_control(model, data, left_wheel_velocity: float, right_wheel_velocity: float) -> None:
    current_left = float(data.ctrl[0]) if len(data.ctrl) else 0.0
    current_right = float(data.ctrl[2]) if len(data.ctrl) > 2 else current_left

    target_left = _clamp(left_wheel_velocity, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    target_right = _clamp(right_wheel_velocity, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)

    smooth_left = current_left + _clamp(target_left - current_left, -MAX_WHEEL_ACCEL, MAX_WHEEL_ACCEL)
    smooth_right = current_right + _clamp(target_right - current_right, -MAX_WHEEL_ACCEL, MAX_WHEEL_ACCEL)
    set_wheel_velocity(model, data, smooth_left, smooth_right)
