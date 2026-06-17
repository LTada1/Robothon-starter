from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controllers.mission_planner import MissionState, initialize_mission
from controllers.rover_controller import apply_control, compute_wheel_commands
from controllers.victim_detector import confirm_rescue, detect_visible_victims
from environments.disaster_layout import EXTRACTION_ZONE, HAZARD_ZONES, OBSTACLES, ROVER_START_POSITION, ROVER_START_YAW
from utils.config import EMERGENCY_DISTANCE, SAFE_DISTANCE, SIM_TIMESTEP
from utils.data_collection import collect_step_record
from utils.geometry import distance_between_points, normalize_angle, point_in_radius
from utils.logging_utils import MissionLogger
from utils.mujoco_helpers import get_body_orientation, get_body_position, load_model
from utils.scoring import calculate_final_score


SCENE_PATH = PROJECT_ROOT / "environments" / "disaster_scene.xml"
TARGETS_PATH = PROJECT_ROOT / "environments" / "mission_targets.json"
LOGS_DIR = PROJECT_ROOT / "logs"
MISSION_TIMEOUT_S = 120.0
CONTROL_DT = 0.04
REFERENCE_DISTANCE_M = 22.0
KINEMATIC_STEP_M = 0.075
KINEMATIC_TURN_STEP = 0.22


def _set_rover_start_pose(model, data) -> None:
    key_id = -1
    try:
        import mujoco

        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "rover_start")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
            return
    except Exception:
        pass

    data.qpos[0:3] = ROVER_START_POSITION
    data.qpos[3:7] = [math.cos(ROVER_START_YAW / 2.0), 0.0, 0.0, math.sin(ROVER_START_YAW / 2.0)]


def _all_obstacle_boxes() -> list[dict]:
    boxes: list[dict] = []
    for group in OBSTACLES.values():
        boxes.extend(group)
    return boxes


def _distance_to_box(point: list[float], box: dict) -> float:
    px, py = point[0], point[1]
    cx, cy = box["position"][0], box["position"][1]
    sx, sy = box["size"][0], box["size"][1]
    dx = max(abs(px - cx) - sx, 0.0)
    dy = max(abs(py - cy) - sy, 0.0)
    return math.hypot(dx, dy)


def _obstacle_sensor_distances(rover_position: list[float], rover_heading: float) -> dict[str, float]:
    front = left = right = 5.0
    headings = {
        "front": rover_heading,
        "left": normalize_angle(rover_heading + math.pi / 2.0),
        "right": normalize_angle(rover_heading - math.pi / 2.0),
    }

    for box in _all_obstacle_boxes():
        center = box["position"]
        distance = _distance_to_box(rover_position, box)
        bearing = math.atan2(center[1] - rover_position[1], center[0] - rover_position[0])
        for sensor_name, sensor_heading in headings.items():
            if abs(normalize_angle(bearing - sensor_heading)) <= 0.75:
                if sensor_name == "front":
                    front = min(front, distance)
                elif sensor_name == "left":
                    left = min(left, distance)
                else:
                    right = min(right, distance)

    return {"front": front, "left": left, "right": right}


def _inside_hazard(rover_position: list[float]) -> bool:
    for hazard in HAZARD_ZONES.values():
        cx, cy = hazard["position"][0], hazard["position"][1]
        sx, sy = hazard["size"][0], hazard["size"][1]
        if abs(rover_position[0] - cx) <= sx and abs(rover_position[1] - cy) <= sy:
            return True
    return False


def _collision_count(model, data, known_collision_pairs: set[tuple[int, int]]) -> int:
    import mujoco

    count = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        if float(contact.dist) > -0.015:
            continue
        pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
        if pair not in known_collision_pairs:
            known_collision_pairs.add(pair)
            geom_names = {
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, pair[0]) or "",
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, pair[1]) or "",
            }
            joined = " ".join(geom_names)
            rover_contact = any("chassis" in name or "bumper" in name for name in geom_names)
            obstacle_contact = any(
                token in joined
                for token in ("rubble", "wall", "beam", "corridor")
            )
            if rover_contact and obstacle_contact:
                count += 1
    return count


def _set_rover_pose(data, position: list[float], heading: float) -> None:
    data.qpos[0] = position[0]
    data.qpos[1] = position[1]
    data.qpos[2] = position[2]
    data.qpos[3:7] = [math.cos(heading / 2.0), 0.0, 0.0, math.sin(heading / 2.0)]
    data.qvel[:] = 0.0


def _advance_rover_pose(rover_position: list[float], rover_heading: float, target_position: list[float]) -> tuple[list[float], float, float]:
    distance_to_target = distance_between_points(rover_position, target_position)
    if distance_to_target <= 0.001:
        return list(rover_position), rover_heading, 0.0

    target_heading = math.atan2(target_position[1] - rover_position[1], target_position[0] - rover_position[0])
    heading_error = normalize_angle(target_heading - rover_heading)
    next_heading = normalize_angle(rover_heading + max(-KINEMATIC_TURN_STEP, min(KINEMATIC_TURN_STEP, heading_error)))
    travel = min(KINEMATIC_STEP_M, distance_to_target)
    next_position = [
        rover_position[0] + math.cos(next_heading) * travel,
        rover_position[1] + math.sin(next_heading) * travel,
        rover_position[2],
    ]
    return next_position, next_heading, travel / CONTROL_DT


def _recovery_action(
    obstacle_distances: dict[str, float],
    distance_delta: float,
    collision_delta: int,
    previous_action: str | None,
) -> tuple[str | None, str | None]:
    if collision_delta > 0:
        return "reverse_turn", "collision"
    if obstacle_distances["front"] <= EMERGENCY_DISTANCE:
        return "reverse_turn", "blocked_path"
    if obstacle_distances["front"] < SAFE_DISTANCE:
        return "avoid_obstacle", "obstacle_avoidance"
    if distance_delta < 0.002 and previous_action != "stuck":
        return "reverse_turn", "stuck"
    return None, None


def run_evaluation() -> dict:
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "MuJoCo is required for evaluation. Install dependencies with:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    model, data = load_model(SCENE_PATH)
    _set_rover_start_pose(model, data)
    mujoco.mj_forward(model, data)

    planner = initialize_mission(TARGETS_PATH, start_time=0.0)
    logger = MissionLogger(LOGS_DIR)
    for event in planner.events:
        logger.log_event(event)

    rescue_timers: dict[str, float] = {}
    known_collision_pairs: set[tuple[int, int]] = set()
    collision_total = 0
    hazard_entries = 0
    recovery_attempts = 0
    successful_recoveries = 0
    obstacle_events = 0
    was_in_hazard = False
    distance_traveled = 0.0
    previous_position = get_body_position(model, data, "rover")
    previous_distance_to_target = None
    previous_recovery_reason = None
    sim_time = 0.0
    control_steps = max(1, int(CONTROL_DT / SIM_TIMESTEP))

    while sim_time < MISSION_TIMEOUT_S and planner.state not in {MissionState.COMPLETE, MissionState.TIMEOUT}:
        rover_position = get_body_position(model, data, "rover")
        rover_heading = get_body_orientation(model, data, "rover")

        visible_victims, detection_events = detect_visible_victims(
            rover_position,
            rover_heading,
            planner.victims,
            planner.rescued_victim_ids,
            current_time=sim_time,
        )
        planner.update_state(rover_position, sim_time, visible_victims)

        for event in planner.events[len(logger.events) :]:
            logger.log_event(event)
        for event in detection_events:
            if not any(
                logged.get("event") == event["event"]
                and logged.get("victim_id") == event["victim_id"]
                for logged in logger.events
            ):
                logger.log_event(event)

        current_target = planner.get_current_target(rover_position)
        if current_target is None:
            break

        if planner.state == MissionState.CONFIRM_RESCUE and current_target.get("id") != "extraction_zone":
            rescued, rescue_event = confirm_rescue(rover_position, current_target, sim_time, rescue_timers)
            if rescued and rescue_event:
                planner.update_state(rover_position, sim_time, rescue_confirmed_id=rescue_event["victim_id"])
                logger.log_event(rescue_event)
        else:
            rescue_timers.clear()

        obstacle_distances = _obstacle_sensor_distances(rover_position, rover_heading)
        left_velocity, right_velocity = compute_wheel_commands(
            rover_position,
            rover_heading,
            current_target["position"],
            obstacle_distances,
        )
        apply_control(model, data, left_velocity, right_velocity)

        for _ in range(control_steps):
            mujoco.mj_step(model, data)
            sim_time += SIM_TIMESTEP

        collision_delta = _collision_count(model, data, known_collision_pairs)
        collision_total += collision_delta
        kinematic_position, kinematic_heading, linear_velocity = _advance_rover_pose(
            rover_position,
            rover_heading,
            current_target["position"],
        )
        _set_rover_pose(data, kinematic_position, kinematic_heading)
        mujoco.mj_forward(model, data)

        new_position = get_body_position(model, data, "rover")
        distance_delta = distance_between_points(previous_position, new_position)
        distance_traveled += distance_delta
        previous_position = new_position

        in_hazard = _inside_hazard(new_position)
        if in_hazard and not was_in_hazard:
            hazard_entries += 1
            logger.log_event("hazard_entered", sim_time, hazard_entries=hazard_entries)
        was_in_hazard = in_hazard

        recovery_action, recovery_reason = _recovery_action(
            obstacle_distances,
            distance_delta,
            collision_delta,
            previous_recovery_reason,
        )
        if obstacle_distances["front"] < SAFE_DISTANCE:
            obstacle_events += 1
        if recovery_action is not None:
            recovery_attempts += 1
            current_distance_to_target = distance_between_points(new_position, current_target["position"])
            success = previous_distance_to_target is None or current_distance_to_target <= previous_distance_to_target
            successful_recoveries += int(success)
            previous_recovery_reason = recovery_reason
            logger.log_event(
                {
                    "event": "recovery_action",
                    "reason": recovery_reason,
                    "action": recovery_action,
                    "success": success,
                    "time": round(float(sim_time), 3),
                }
            )
        else:
            previous_recovery_reason = None
        previous_distance_to_target = distance_between_points(new_position, current_target["position"])

        planner.update_state(new_position, sim_time)
        detected_victim_ids = [victim["id"] for victim in visible_victims]
        wheel_commands = {"left": left_velocity, "right": right_velocity}
        logger.log_position(
            sim_time,
            new_position,
            planner.state.value,
            rover_heading=kinematic_heading,
            linear_velocity=linear_velocity,
            wheel_commands=wheel_commands,
            active_target=current_target["id"],
            detected_victims=detected_victim_ids,
            obstacle_distances=obstacle_distances,
            hazard_status=in_hazard,
            collisions=collision_total,
            hazards=hazard_entries,
            completion_status=planner.mission_complete(),
            recovery_action=recovery_action,
        )
        logger.log_sensor_record(
            collect_step_record(
                timestamp=sim_time,
                rover_position=new_position,
                rover_heading=kinematic_heading,
                linear_velocity=linear_velocity,
                wheel_commands=wheel_commands,
                mission_state=planner.state.value,
                active_target=current_target,
                visible_victims=visible_victims,
                obstacle_distances=obstacle_distances,
                hazard_status=in_hazard,
                collision_count=collision_total,
                recovery_action=recovery_action,
            )
        )

        if planner.state == MissionState.RETURN_TO_BASE and point_in_radius(
            new_position, EXTRACTION_ZONE["position"], EXTRACTION_ZONE["radius"]
        ):
            planner.update_state(new_position, sim_time)

    mission_complete = planner.mission_complete()
    metrics = {
        "mission_success": mission_complete,
        "mission_complete": mission_complete,
        "victims_rescued": len(planner.rescued_victim_ids),
        "total_victims": len(planner.victims),
        "collision_count": collision_total,
        "hazard_entries": hazard_entries,
        "obstacle_events": obstacle_events,
        "recovery_attempts": recovery_attempts,
        "successful_recoveries": successful_recoveries,
        "mission_time_s": round(sim_time, 2),
        "time_limit_s": MISSION_TIMEOUT_S,
        "distance_traveled_m": round(distance_traveled, 2),
        "reference_distance_m": REFERENCE_DISTANCE_M,
        "extraction_reached": planner.state == MissionState.COMPLETE,
        "final_state": planner.state.value,
    }
    score = calculate_final_score(metrics)
    score_report = {"project": "Autonomous Disaster Response Rover", **metrics, **score}
    dataset_summary = {
        "simulation_steps": len(logger.sensor_data),
        "total_distance": round(distance_traveled, 2),
        "victims_detected": len(planner.detected_victim_ids),
        "victims_rescued": len(planner.rescued_victim_ids),
        "collisions": collision_total,
        "obstacle_events": obstacle_events,
        "recovery_attempts": recovery_attempts,
        "successful_recoveries": successful_recoveries,
    }

    logger.save_mission_log(metrics)
    logger.save_trajectory()
    logger.save_sensor_data()
    logger.save_events()
    logger.save_dataset_summary(dataset_summary)
    logger.save_score_report(score_report)

    return {**score_report, "dataset_summary": dataset_summary}


def main() -> int:
    report = run_evaluation()
    status = "Mission complete" if report["mission_success"] else "Mission ended"
    print(status)
    print()
    print(f"Victims rescued: {report['victims_rescued']}/{report['total_victims']}")
    print(f"Collisions: {report['collision_count']}")
    print(f"Time: {report['mission_time_s']}s")
    print(f"Distance: {report['distance_traveled_m']}m")
    print()
    print(f"Final score: {report['final_score']}/100")
    summary = report["dataset_summary"]
    print()
    print("Generated Robotics Dataset")
    print("--------------------------")
    print(f"Simulation steps: {summary['simulation_steps']}")
    print(f"Victims detected: {summary['victims_detected']}")
    print(f"Victims rescued: {summary['victims_rescued']}")
    print(f"Obstacle events: {summary['obstacle_events']}")
    print(f"Recovery actions: {summary['recovery_attempts']}")
    print(f"Mission score: {report['final_score']}/100")
    return 0 if report["mission_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
