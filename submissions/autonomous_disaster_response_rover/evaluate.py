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
from utils.config import SIM_TIMESTEP
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
    count = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
        if pair not in known_collision_pairs:
            known_collision_pairs.add(pair)
            count += 1
    return count


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
    was_in_hazard = False
    distance_traveled = 0.0
    previous_position = get_body_position(model, data, "rover")
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

        new_position = get_body_position(model, data, "rover")
        distance_traveled += distance_between_points(previous_position, new_position)
        previous_position = new_position

        collision_total += _collision_count(model, data, known_collision_pairs)
        in_hazard = _inside_hazard(new_position)
        if in_hazard and not was_in_hazard:
            hazard_entries += 1
            logger.log_event("hazard_entered", sim_time, hazard_entries=hazard_entries)
        was_in_hazard = in_hazard

        planner.update_state(new_position, sim_time)
        logger.log_position(
            sim_time,
            new_position,
            planner.state.value,
            collisions=collision_total,
            hazards=hazard_entries,
            completion_status=planner.mission_complete(),
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
        "mission_time_s": round(sim_time, 2),
        "time_limit_s": MISSION_TIMEOUT_S,
        "distance_traveled_m": round(distance_traveled, 2),
        "reference_distance_m": REFERENCE_DISTANCE_M,
        "extraction_reached": planner.state == MissionState.COMPLETE,
        "final_state": planner.state.value,
    }
    score = calculate_final_score(metrics)
    score_report = {"project": "Autonomous Disaster Response Rover", **metrics, **score}

    logger.save_mission_log(metrics)
    logger.save_trajectory()
    logger.save_score_report(score_report)

    return score_report


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
    return 0 if report["mission_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
