from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controllers.mission_planner import MissionState, initialize_mission
from controllers.rover_controller import apply_control, compute_control_decision
from controllers.victim_detector import confirm_rescue, detect_visible_victims, evaluate_rescue_interaction
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
from utils.config import RESCUE_DEPLOY_COMMAND, RESCUE_RETRACT_COMMAND, SAFE_DISTANCE, SIM_TIMESTEP
from utils.data_collection import collect_step_record
from utils.geometry import distance_between_points, point_in_radius
from utils.logging_utils import MissionLogger
from utils.mujoco_helpers import (
    get_body_orientation,
    get_body_position,
    get_rescue_deployer_state,
    load_model,
    set_rescue_deployer,
)
from utils.render_utils import initialize_viewer
from utils.scoring import calculate_final_score


DEMO_VIDEO_PATH = LOGS_DIR / "demo.mp4"
DEMO_FPS = 10


def _status_banner(new_events: list[dict], state: MissionState, mission_complete: bool) -> str | None:
    for event in reversed(new_events):
        if event.get("event") == "victim_rescued":
            return f"[RESCUED] {event['victim_id']} confirmed"
        if event.get("event") == "rescue_deployment_started":
            return f"[RESCUE] Deploying tool for {event['victim_id']}"
        if event.get("event") == "rescue_interaction_state_changed":
            return f"[{event['to'].upper()}] {event['victim_id']}"
        if event.get("event") == "victim_detected":
            return f"[DETECTED] {event['victim_id']} detected"
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
    distance: float,
    current_score: float,
    controller_state: str = "",
    avoidance_decision: str = "",
    rescue_interaction_state: str = "inactive",
    rescue_tool_deployed: bool = False,
) -> list[str]:
    target_name = target["id"] if target else "none"
    return [
        f"State: {state.value}",
        f"Target: {target_name}",
        f"Rescued: {rescued}/{total_victims}",
        f"Time: {mission_time:05.2f}s",
        f"Distance: {distance:05.2f} m",
        f"Score: {current_score:05.2f}/100",
        f"Collisions: {collisions}",
        f"Controller: {controller_state or 'TRACKING_TARGET'}",
        f"Decision: {avoidance_decision or 'follow_target_heading'}",
        f"Rescue: {rescue_interaction_state}",
        f"Tool: {'deployed' if rescue_tool_deployed else 'retracted'}",
    ]


def _print_event(event: dict) -> None:
    if event.get("event") == "victim_detected":
        print("[DETECTED]")
        print(f"{event['victim_id']} detected")
    elif event.get("event") == "victim_rescued":
        print("[RESCUED]")
        print(f"{event['victim_id']} rescued")
    elif event.get("event") == "rescue_deployment_started":
        print("[RESCUE]")
        print(f"Deploying rescue tool for {event['victim_id']}")


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
    previous_controller_state = None
    previous_rescue_interaction_state = None
    sim_time = 0.0
    control_steps = max(1, int(CONTROL_DT / SIM_TIMESTEP))
    active_banner = "[START] Mission initialized"
    banner_frames_remaining = DEMO_FPS * 3
    return_announced = False
    demo_extraction_radius = 0.35

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
            if len(planner.rescued_victim_ids) == len(planner.victims):
                planner.state = MissionState.RETURN_TO_BASE
                current_target = {"id": "extraction_zone", **planner.extraction_zone}
            if current_target is None:
                break

            rescue_interaction = {"rescue_interaction_state": "inactive"}
            rescue_actuator_command = RESCUE_RETRACT_COMMAND
            victim_confirmation_status = ""
            if current_target.get("id") != "extraction_zone":
                rescue_deployer_state = get_rescue_deployer_state(model, data)
                rescue_interaction = evaluate_rescue_interaction(
                    rover_position,
                    rover_heading,
                    current_target,
                    rescue_deployer_state,
                )
                planner.update_state(
                    rover_position,
                    sim_time,
                    rescue_interaction_state=rescue_interaction["rescue_interaction_state"],
                )
                if rescue_interaction["rescue_interaction_state"] in {"deploying", "confirming"}:
                    rescue_actuator_command = RESCUE_DEPLOY_COMMAND
                if rescue_interaction["rescue_interaction_state"] != previous_rescue_interaction_state:
                    state_event = {
                        "event": "rescue_interaction_state_changed",
                        "time": round(sim_time, 3),
                        "from": previous_rescue_interaction_state or "inactive",
                        "to": rescue_interaction["rescue_interaction_state"],
                        "victim_id": current_target["id"],
                        "aligned_to_victim": rescue_interaction.get("aligned_to_victim", False),
                        "rescue_tool_deployed": rescue_interaction.get("rescue_tool_deployed", False),
                        "rescue_actuator_command": round(float(rescue_actuator_command), 5),
                    }
                    logger.log_event(state_event)
                    loop_events.append(state_event)
                    if rescue_interaction["rescue_interaction_state"] == "deploying":
                        deployment_event = {
                            "event": "rescue_deployment_started",
                            "time": round(sim_time, 3),
                            "victim_id": current_target["id"],
                            "actuator": "rescue_deployer_actuator",
                            "command": round(float(rescue_actuator_command), 5),
                        }
                        logger.log_event(deployment_event)
                        loop_events.append(deployment_event)
                    previous_rescue_interaction_state = rescue_interaction["rescue_interaction_state"]

            set_rescue_deployer(model, data, rescue_actuator_command)

            if planner.state == MissionState.CONFIRMING and current_target.get("id") != "extraction_zone":
                rescued, rescue_event = confirm_rescue(
                    rover_position,
                    current_target,
                    sim_time,
                    rescue_timers,
                    rover_heading=rover_heading,
                    rescue_deployer_state=get_rescue_deployer_state(model, data),
                )
                if rescued and rescue_event:
                    planner.update_state(rover_position, sim_time, rescue_confirmed_id=rescue_event["victim_id"])
                    victim_confirmation_status = "confirmed"
                    deployment_complete_event = {
                        "event": "rescue_deployment_complete",
                        "time": round(sim_time, 3),
                        "victim_id": rescue_event["victim_id"],
                        "actuator": "rescue_deployer_actuator",
                    }
                    logger.log_event(deployment_complete_event)
                    loop_events.append(deployment_complete_event)
                    rescue_event["confirmation_method"] = "aligned_deployed_rescue_tool"
                    logger.log_event(rescue_event)
                    loop_events.append(rescue_event)
                else:
                    victim_confirmation_status = "confirming"
            else:
                rescue_timers.clear()

            obstacle_distances = _obstacle_sensor_distances(rover_position, rover_heading)
            controller_decision = compute_control_decision(
                rover_position,
                rover_heading,
                current_target["position"],
                obstacle_distances,
            )
            if controller_decision["controller_state"] != previous_controller_state:
                state_event = {
                    "event": "controller_state_changed",
                    "time": round(sim_time, 3),
                    "from": previous_controller_state or "NONE",
                    "to": controller_decision["controller_state"],
                    "avoidance_decision": controller_decision["avoidance_decision"],
                    "active_target": current_target["id"],
                    "front_distance": round(float(obstacle_distances.get("front", 0.0)), 5),
                    "left_distance": round(float(obstacle_distances.get("left", 0.0)), 5),
                    "right_distance": round(float(obstacle_distances.get("right", 0.0)), 5),
                    "left_wheel_velocity": round(float(controller_decision["left_wheel_velocity"]), 5),
                    "right_wheel_velocity": round(float(controller_decision["right_wheel_velocity"]), 5),
                }
                logger.log_event(state_event)
                loop_events.append(state_event)
                previous_controller_state = controller_decision["controller_state"]
            left_velocity = controller_decision["left_wheel_velocity"]
            right_velocity = controller_decision["right_wheel_velocity"]
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
            if len(planner.rescued_victim_ids) == len(planner.victims):
                if point_in_radius(new_position, EXTRACTION_ZONE["position"], demo_extraction_radius):
                    planner.state = MissionState.COMPLETE
                    completion_event = {"event": "mission_complete", "time": round(sim_time, 3)}
                    if not any(event.get("event") == "mission_complete" for event in logger.events):
                        logger.log_event(completion_event)
                        loop_events.append(completion_event)
                else:
                    planner.state = MissionState.RETURN_TO_BASE

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
                controller_decision=controller_decision,
                rescue_interaction=rescue_interaction,
                rescue_actuator_command=rescue_actuator_command,
                victim_confirmation_status=victim_confirmation_status,
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
                    controller_decision=controller_decision,
                    rescue_interaction=rescue_interaction,
                    rescue_actuator_command=rescue_actuator_command,
                    victim_confirmation_status=victim_confirmation_status,
                )
            )

            for event in logger.events[printed_events:]:
                _print_event(event)
            printed_events = len(logger.events)

            if planner.state == MissionState.RETURN_TO_BASE and not return_announced:
                print("[RETURN]")
                print("Heading to extraction zone")
                active_banner = "[RETURN] Heading to extraction zone"
                banner_frames_remaining = DEMO_FPS * 3
                return_announced = True

            candidate_banner = _status_banner(loop_events, planner.state, mission_complete)
            if candidate_banner:
                active_banner = candidate_banner
                banner_frames_remaining = DEMO_FPS * 3

            overlay = _overlay_lines(
                state=planner.state,
                target=current_target,
                rescued=len(planner.rescued_victim_ids),
                total_victims=len(planner.victims),
                collisions=collision_total,
                mission_time=sim_time,
                distance=distance_traveled,
                current_score=live_score,
                controller_state=controller_decision.get("controller_state", ""),
                avoidance_decision=controller_decision.get("avoidance_decision", ""),
                rescue_interaction_state=rescue_interaction.get("rescue_interaction_state", "inactive"),
                rescue_tool_deployed=bool(rescue_interaction.get("rescue_tool_deployed", False)),
            )
            renderer.capture_split_frame(
                data,
                overlay_lines=overlay,
                banner=active_banner if banner_frames_remaining > 0 else None,
            )
            banner_frames_remaining = max(0, banner_frames_remaining - 1)

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
                    distance=distance_traveled,
                    current_score=final_score,
                    controller_state="COMPLETE",
                    avoidance_decision="mission_finished",
                    rescue_interaction_state="complete",
                    rescue_tool_deployed=False,
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
