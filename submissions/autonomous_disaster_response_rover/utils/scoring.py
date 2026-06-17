from __future__ import annotations


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def calculate_victim_score(victims_rescued: int, total_victims: int) -> float:
    if total_victims <= 0:
        return 0.0
    return 40.0 * _clamp(victims_rescued / total_victims, 0.0, 1.0)


def calculate_completion_score(mission_complete: bool) -> float:
    return 20.0 if mission_complete else 0.0


def calculate_time_score(mission_time_s: float, time_limit_s: float) -> float:
    if time_limit_s <= 0:
        return 0.0
    ratio = _clamp(mission_time_s / time_limit_s, 0.0, 1.0)
    return 15.0 * (1.0 - ratio)


def calculate_collision_score(collision_count: int, max_collisions: int = 10) -> float:
    if max_collisions <= 0:
        return 0.0
    penalty = _clamp(collision_count / max_collisions, 0.0, 1.0)
    return 10.0 * (1.0 - penalty)


def calculate_distance_score(distance_traveled_m: float, reference_distance_m: float) -> float:
    if reference_distance_m <= 0:
        return 0.0
    efficiency = _clamp(reference_distance_m / max(distance_traveled_m, reference_distance_m), 0.0, 1.0)
    return 10.0 * efficiency


def calculate_hazard_score(hazard_entries: int, max_hazard_entries: int = 5) -> float:
    if max_hazard_entries <= 0:
        return 0.0
    penalty = _clamp(hazard_entries / max_hazard_entries, 0.0, 1.0)
    return 5.0 * (1.0 - penalty)


def calculate_final_score(metrics: dict) -> dict:
    breakdown = {
        "victims": calculate_victim_score(
            int(metrics.get("victims_rescued", 0)),
            int(metrics.get("total_victims", 0)),
        ),
        "completion": calculate_completion_score(bool(metrics.get("mission_complete", False))),
        "time": calculate_time_score(
            float(metrics.get("mission_time_s", 0.0)),
            float(metrics.get("time_limit_s", 1.0)),
        ),
        "collision": calculate_collision_score(int(metrics.get("collision_count", 0))),
        "distance": calculate_distance_score(
            float(metrics.get("distance_traveled_m", 0.0)),
            float(metrics.get("reference_distance_m", 1.0)),
        ),
        "hazard": calculate_hazard_score(int(metrics.get("hazard_entries", 0))),
    }
    final_score = round(sum(breakdown.values()), 2)
    return {
        "final_score": final_score,
        "breakdown": {key: round(value, 2) for key, value in breakdown.items()},
    }
