from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controllers.mission_planner import MissionState, initialize_mission
from controllers.rover_controller import apply_control, compute_wheel_commands
from controllers.victim_detector import confirm_rescue, detect_visible_victims
from evaluate import (
    CONTROL_DT,
    EXTRACTION_ZONE,
    KINEMATIC_STEP_M,
    LOGS_DIR,
    MISSION_TIMEOUT_S,
    REFERENCE_DISTANCE_M,
    SCENE_PATH,
    TARGETS_PATH,
    _advance_rover_pose,
    _collision_count,
    _inside_hazard,
    _obstacle_sensor_distances,
    _recovery_action,
    _set_rover_pose,
    _set_rover_start_pose,
)
from utils.config import SAFE_DISTANCE, SIM_TIMESTEP
from utils.data_collection import collect_step_record
from utils.geometry import distance_between_points, point_in_radius
from utils.logging_utils import MissionLogger
from utils.mujoco_helpers import get_body_orientation, get_body_position, load_model
from utils.render_utils import initialize_viewer
from utils.scoring import calculate_final_score


DEMO_VIDEO_PATH = LOGS_DIR / "demo.mp4"
DEMO_FPS = 10


def _status_banner(new_events: list[dict], state: MissionState, mission_complete: bool) -> str | None:
    for event in reversed(new_events):
        if event.get("event") == "victim_rescued":
            return f"[RESCUED] {event['victim_id']} confirmed"
        if event.get("event") == "victim_detected":
            return f"[DETECTED] {event['victim_id']} detected"
        if event.get("event") == "recovery_action":
            return f"[RECOVERY] {event['action']} ({event['reason']})"
    if mission_complete:
        return "[COMPLETE] Mission finished at extraction zone"
    if state == MissionState.RETURN_TO_BASE:
        return "[RETURN] Heading to extraction zone"
    if state == MissionState.SEARCH:
        return "[SEARCH] Locating victims"
    return None


def _overlay_lines(
    *,
    state: MissionState,
    target: dict | None,
    rescued: int,
    total_victims: int,
    collisions: int,
    mission_time: float,
    current_score: float,
) -> list[str]:
    target_name = target["id"] if target else "none"
    return [
        f"State: {state.value}",
        f"Target: {target_name}",
        f"Rescued: {rescued}/{total_victims}",
        f"Collisions: {collisions}",
        f"Mission time: {mission_time:05.2f}s",
        f"Live score: {current_score:05.2f}/100",
    ]


def _print_event(event: dict) -> None:
    if event.get("event") == "victim_detected":
        print("[DETECTED]")
        print(f"{event['victim_id']} detected")
    elif event.get("event") == "victim_rescued":
        print("[RESCUED]")
        print(f"{event['victim_id']} rescued")
    elif event.get("event") == "recovery_action":
        print("[RECOVERY]")
        print(f"{event['action']} because {event['reason']}")


def run_demo() -> dict:
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "MuJoCo is required for the demo. Install dependencies with:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    model, data = load_model(SCENE_PATH)
    _set_rover_start_pose(model, data)
    mujoco.mj_forward(model, data)

    planner = initialize_mission(TARGETS_PATH, start_time=0.0)
    logger = MissionLogger(LOGS_DIR)
    renderer = initialize_viewer(model, fps=DEMO_FPS)
    for event in planner.events:
        logger.log_event(event)

    print("[START]")
    print("Autonomous Disaster Response Rover mission initialized")
    print("[SEARCH]")
    print("Moving to victim_1")

    rescue_timers: dict[str, float] = {}
    known_collision_pairs: set[tuple[int, int]] = set()
    printed_events = 0
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

    try:
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

            loop_events: list[dict] = []
            for event in planner.events[len(logger.events) :]:
                logger.log_event(event)
                loop_events.append(event)
            for event in detection_events:
                duplicate = any(
                    logged.get("event") == event["event"] and logged.get("victim_id") == event["victim_id"]
                    for logged in logger.events
                )
                if not duplicate:
                    logger.log_event(event)
                    loop_events.append(event)

            current_target = planner.get_current_target(rover_position)
            if current_target is None:
                break

            if planner.state == MissionState.CONFIRM_RESCUE and current_target.get("id") != "extraction_zone":
                rescued, rescue_event = confirm_rescue(rover_position, current_target, sim_time, rescue_timers)
                if rescued and rescue_event:
                    planner.update_state(rover_position, sim_time, rescue_confirmed_id=rescue_event["victim_id"])
                    logger.log_event(rescue_event)
                    loop_events.append(rescue_event)
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
                hazard_event = {"event": "hazard_entered", "time": round(sim_time, 3), "hazard_entries": hazard_entries}
                logger.log_event(hazard_event)
                loop_events.append(hazard_event)
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
                recovery_event = {
                    "event": "recovery_action",
                    "reason": recovery_reason,
                    "action": recovery_action,
                    "success": success,
                    "time": round(sim_time, 3),
                }
                logger.log_event(recovery_event)
                loop_events.append(recovery_event)
            else:
                previous_recovery_reason = None
            previous_distance_to_target = distance_between_points(new_position, current_target["position"])

            planner.update_state(new_position, sim_time)
            if planner.state == MissionState.RETURN_TO_BASE and point_in_radius(
                new_position, EXTRACTION_ZONE["position"], EXTRACTION_ZONE["radius"]
            ):
                planner.update_state(new_position, sim_time)

            detected_victim_ids = [victim["id"] for victim in visible_victims]
            wheel_commands = {"left": left_velocity, "right": right_velocity}
            mission_complete = planner.mission_complete()
            live_metrics = {
                "victims_rescued": len(planner.rescued_victim_ids),
                "total_victims": len(planner.victims),
                "mission_complete": mission_complete,
                "mission_time_s": sim_time,
                "time_limit_s": MISSION_TIMEOUT_S,
                "collision_count": collision_total,
                "distance_traveled_m": distance_traveled,
                "reference_distance_m": REFERENCE_DISTANCE_M,
                "hazard_entries": hazard_entries,
            }
            live_score = calculate_final_score(live_metrics)["final_score"]
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
                completion_status=mission_complete,
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

            for event in logger.events[printed_events:]:
                _print_event(event)
            printed_events = len(logger.events)

            if planner.state == MissionState.RETURN_TO_BASE and "return_printed" not in getattr(run_demo, "_flags", set()):
                print("[RETURN]")
                print("Heading to extraction zone")
                run_demo._flags = {"return_printed"}

            overlay = _overlay_lines(
                state=planner.state,
                target=current_target,
                rescued=len(planner.rescued_victim_ids),
                total_victims=len(planner.victims),
                collisions=collision_total,
                mission_time=sim_time,
                current_score=live_score,
            )
            renderer.capture_split_frame(
                data,
                overlay_lines=overlay,
                banner=_status_banner(loop_events, planner.state, mission_complete),
            )

        # Hold the completed mission for a few seconds so the final score is readable.
        final_metrics = {
            "victims_rescued": len(planner.rescued_victim_ids),
            "total_victims": len(planner.victims),
            "mission_complete": planner.mission_complete(),
            "mission_time_s": sim_time,
            "time_limit_s": MISSION_TIMEOUT_S,
            "collision_count": collision_total,
            "distance_traveled_m": distance_traveled,
            "reference_distance_m": REFERENCE_DISTANCE_M,
            "hazard_entries": hazard_entries,
        }
        final_score = calculate_final_score(final_metrics)["final_score"]
        for _ in range(DEMO_FPS * 3):
            renderer.capture_split_frame(
                data,
                overlay_lines=_overlay_lines(
                    state=planner.state,
                    target={"id": "extraction_zone"},
                    rescued=len(planner.rescued_victim_ids),
                    total_victims=len(planner.victims),
                    collisions=collision_total,
                    mission_time=sim_time,
                    current_score=final_score,
                ),
                banner="[COMPLETE] Mission finished",
            )
        video_path = renderer.write_video(DEMO_VIDEO_PATH)
    finally:
        renderer.cleanup()

    if planner.mission_complete():
        print("[COMPLETE]")
        print("Mission finished")

    metrics = {
        "mission_success": planner.mission_complete(),
        "mission_complete": planner.mission_complete(),
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
        "demo_video": str(video_path),
    }
    score_report = {
        "project": "Autonomous Disaster Response Rover",
        **metrics,
        **calculate_final_score(metrics),
    }
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
    logger.save_post_run_artifacts(
        score_report=score_report,
        dataset_summary=dataset_summary,
        victims=planner.victims,
        extraction_zone=planner.extraction_zone,
    )

    return score_report


def main() -> int:
    report = run_demo()
    print()
    print(f"Victims rescued: {report['victims_rescued']}/{report['total_victims']}")
    print(f"Collisions: {report['collision_count']}")
    print(f"Time: {report['mission_time_s']}s")
    print(f"Distance: {report['distance_traveled_m']}m")
    print()
    print(f"Final score: {report['final_score']}/100")
    print(f"Demo video: {report['demo_video']}")
    return 0 if report["mission_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
