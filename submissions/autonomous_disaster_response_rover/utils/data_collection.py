from __future__ import annotations

from typing import Any


def collect_simulated_lidar_readings(obstacle_distances: dict[str, float]) -> dict[str, float]:
    front = float(obstacle_distances.get("front", 5.0))
    left = float(obstacle_distances.get("left", 5.0))
    right = float(obstacle_distances.get("right", 5.0))
    return {
        "lidar_front": round(front, 5),
        "lidar_front_left": round(min(front * 1.15, left), 5),
        "lidar_front_right": round(min(front * 1.15, right), 5),
    }


def collect_proximity_sensor_readings(obstacle_distances: dict[str, float]) -> dict[str, float]:
    return {
        "proximity_left": round(float(obstacle_distances.get("left", 5.0)), 5),
        "proximity_right": round(float(obstacle_distances.get("right", 5.0)), 5),
    }


def collect_victim_detection_events(visible_victims: list[dict[str, Any]]) -> list[str]:
    return [victim["id"] for victim in visible_victims]


def collect_navigation_decision(
    mission_state: str,
    active_target: dict[str, Any] | None,
    obstacle_distances: dict[str, float],
    recovery_action: str | None,
    controller_state: str | None = None,
    avoidance_decision: str | None = None,
) -> str:
    if avoidance_decision:
        return avoidance_decision
    if recovery_action:
        return recovery_action
    if controller_state:
        return controller_state.lower()
    if obstacle_distances.get("front", 5.0) < 1.2:
        return "avoid_obstacle"
    if active_target is None:
        return "hold_position"
    if active_target.get("id") == "extraction_zone":
        return "return_to_extraction"
    if mission_state == "CONFIRM_RESCUE":
        return "confirm_rescue"
    return f"navigate_to_{active_target['id']}"


def collect_step_record(
    *,
    timestamp: float,
    rover_position: list[float],
    rover_heading: float,
    linear_velocity: float,
    wheel_commands: dict[str, float],
    mission_state: str,
    active_target: dict[str, Any] | None,
    visible_victims: list[dict[str, Any]],
    obstacle_distances: dict[str, float],
    hazard_status: bool,
    collision_count: int,
    recovery_action: str | None,
    controller_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    detected_victims = collect_victim_detection_events(visible_victims)
    controller_decision = controller_decision or {}
    navigation_decision = collect_navigation_decision(
        mission_state,
        active_target,
        obstacle_distances,
        recovery_action,
        controller_state=controller_decision.get("controller_state"),
        avoidance_decision=controller_decision.get("avoidance_decision"),
    )
    return {
        "timestamp": round(float(timestamp), 3),
        "rover_x": round(float(rover_position[0]), 5),
        "rover_y": round(float(rover_position[1]), 5),
        "rover_z": round(float(rover_position[2]) if len(rover_position) > 2 else 0.0, 5),
        "rover_heading": round(float(rover_heading), 5),
        "linear_velocity": round(float(linear_velocity), 5),
        "mission_state": mission_state,
        "active_target": active_target["id"] if active_target else "",
        "detected_victims": "|".join(detected_victims),
        **collect_simulated_lidar_readings(obstacle_distances),
        **collect_proximity_sensor_readings(obstacle_distances),
        "hazard_status": bool(hazard_status),
        "collision_count": int(collision_count),
        "left_wheel_command": round(float(wheel_commands.get("left", 0.0)), 5),
        "right_wheel_command": round(float(wheel_commands.get("right", 0.0)), 5),
        "navigation_decision": navigation_decision,
        "controller_state": controller_decision.get("controller_state", ""),
        "avoidance_decision": controller_decision.get("avoidance_decision", ""),
        "heading_error": round(float(controller_decision.get("heading_error", 0.0)), 5),
        "distance_to_target": round(float(controller_decision.get("distance_to_target", 0.0)), 5),
        "speed_scale": round(float(controller_decision.get("speed_scale", 0.0)), 5),
        "steering_bias": round(float(controller_decision.get("steering_bias", 0.0)), 5),
        "recovery_action": recovery_action or "",
    }
