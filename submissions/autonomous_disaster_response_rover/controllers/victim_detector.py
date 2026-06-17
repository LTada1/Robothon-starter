from __future__ import annotations

from collections.abc import Iterable, Sequence

try:
    from utils.geometry import distance_between_points, field_of_view_check
except ImportError:
    from ..utils.geometry import distance_between_points, field_of_view_check


Point = Sequence[float]
DEFAULT_FOV_RADIANS = 1.75
DEFAULT_CONFIRM_DURATION_S = 1.0


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


def confirm_rescue(
    rover_position: Point,
    victim: dict,
    current_time: float,
    rescue_start_times: dict[str, float],
    confirm_duration_s: float = DEFAULT_CONFIRM_DURATION_S,
) -> tuple[bool, dict | None]:
    victim_id = victim["id"]
    distance = distance_between_points(rover_position, victim["position"])

    if distance > float(victim["rescue_radius"]):
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
