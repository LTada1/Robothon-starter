from __future__ import annotations

from collections.abc import Iterable, Sequence

try:
    from utils.config import RESCUE_ALIGNMENT_TOLERANCE, RESCUE_DEPLOYED_THRESHOLD
    from utils.geometry import angle_difference, calculate_heading, distance_between_points, field_of_view_check
except ImportError:
    from ..utils.config import RESCUE_ALIGNMENT_TOLERANCE, RESCUE_DEPLOYED_THRESHOLD
    from ..utils.geometry import angle_difference, calculate_heading, distance_between_points, field_of_view_check


Point = Sequence[float]
DEFAULT_FOV_RADIANS = 1.75
DEFAULT_CONFIRM_DURATION_S = 1.0
RESCUE_CONTACT_ALIGNMENT_RADIUS = 0.15


def detect_visible_victims(
    rover_position: Point,
    rover_heading: float,
    victims: Iterable[dict],
    rescued_victim_ids: set[str] | None = None,
    current_time: float | None = None,
    fov_radians: float = DEFAULT_FOV_RADIANS,
) -> tuple[list[dict], list[dict]]:
    rescued_victim_ids = rescued_victim_ids or set()
    visible: list[dict] = []
    events: list[dict] = []

    for victim in victims:
        victim_id = victim["id"]
        if victim_id in rescued_victim_ids:
            continue

        if field_of_view_check(
            rover_position,
            rover_heading,
            victim["position"],
            float(victim["detection_radius"]),
            fov_radians,
        ):
            visible.append(victim)
            event = {"event": "victim_detected", "victim_id": victim_id}
            if current_time is not None:
                event["time"] = round(float(current_time), 3)
            events.append(event)

    return visible, events


def get_active_target(
    rover_position: Point,
    rover_heading: float,
    victims: Iterable[dict],
    rescued_victim_ids: set[str] | None = None,
    fov_radians: float = DEFAULT_FOV_RADIANS,
) -> dict | None:
    visible, _ = detect_visible_victims(
        rover_position,
        rover_heading,
        victims,
        rescued_victim_ids=rescued_victim_ids,
        fov_radians=fov_radians,
    )
    if not visible:
        return None

    return min(visible, key=lambda victim: distance_between_points(rover_position, victim["position"]))


def calculate_rescue_alignment_error(rover_position: Point, rover_heading: float, victim: dict) -> float:
    target_heading = calculate_heading(rover_position, victim["position"])
    return angle_difference(target_heading, rover_heading)


def evaluate_rescue_interaction(
    rover_position: Point,
    rover_heading: float,
    victim: dict,
    rescue_deployer_state: float,
) -> dict:
    distance = distance_between_points(rover_position, victim["position"])
    alignment_error = calculate_rescue_alignment_error(rover_position, rover_heading, victim)
    in_rescue_radius = distance <= float(victim["rescue_radius"])
    aligned = distance <= RESCUE_CONTACT_ALIGNMENT_RADIUS or abs(alignment_error) <= RESCUE_ALIGNMENT_TOLERANCE
    deployed = float(rescue_deployer_state) >= RESCUE_DEPLOYED_THRESHOLD

    if not in_rescue_radius:
        state = "approaching"
    elif not aligned:
        state = "aligning"
    elif not deployed:
        state = "deploying"
    else:
        state = "confirming"

    return {
        "rescue_interaction_state": state,
        "distance_to_victim": distance,
        "alignment_error": alignment_error,
        "in_rescue_radius": in_rescue_radius,
        "aligned_to_victim": aligned,
        "rescue_tool_deployed": deployed,
    }


def confirm_rescue(
    rover_position: Point,
    victim: dict,
    current_time: float,
    rescue_start_times: dict[str, float],
    rover_heading: float | None = None,
    rescue_deployer_state: float | None = None,
    confirm_duration_s: float = DEFAULT_CONFIRM_DURATION_S,
) -> tuple[bool, dict | None]:
    victim_id = victim["id"]
    distance = distance_between_points(rover_position, victim["position"])

    if distance > float(victim["rescue_radius"]):
        rescue_start_times.pop(victim_id, None)
        return False, None

    if rover_heading is not None:
        alignment_error = calculate_rescue_alignment_error(rover_position, rover_heading, victim)
        if distance > RESCUE_CONTACT_ALIGNMENT_RADIUS and abs(alignment_error) > RESCUE_ALIGNMENT_TOLERANCE:
            rescue_start_times.pop(victim_id, None)
            return False, None

    if rescue_deployer_state is not None and float(rescue_deployer_state) < RESCUE_DEPLOYED_THRESHOLD:
        rescue_start_times.pop(victim_id, None)
        return False, None

    rescue_start_times.setdefault(victim_id, float(current_time))
    dwell_time = float(current_time) - rescue_start_times[victim_id]

    if dwell_time >= confirm_duration_s:
        event = {
            "event": "victim_rescued",
            "victim_id": victim_id,
            "time": round(float(current_time), 3),
        }
        rescue_start_times.pop(victim_id, None)
        return True, event

    return False, None
