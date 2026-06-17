from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from utils.geometry import distance_between_points, point_in_radius
except ImportError:
    from ..utils.geometry import distance_between_points, point_in_radius


class MissionState(str, Enum):
    START = "START"
    SEARCH = "SEARCH"
    APPROACH_TARGET = "APPROACH_TARGET"
    CONFIRM_RESCUE = "CONFIRM_RESCUE"
    NEXT_TARGET = "NEXT_TARGET"
    RETURN_TO_BASE = "RETURN_TO_BASE"
    COMPLETE = "COMPLETE"
    TIMEOUT = "TIMEOUT"


DEFAULT_TARGETS_PATH = Path(__file__).resolve().parents[1] / "environments" / "mission_targets.json"


class MissionPlanner:
    def __init__(self, mission_timeout_s: float = 120.0) -> None:
        self.mission_timeout_s = mission_timeout_s
        self.state = MissionState.START
        self.victims: list[dict[str, Any]] = []
        self.extraction_zone: dict[str, Any] = {}
        self.rescued_victim_ids: set[str] = set()
        self.detected_victim_ids: set[str] = set()
        self.current_target_id: str | None = None
        self.events: list[dict[str, Any]] = []
        self.start_time: float = 0.0

    def initialize_mission(self, targets_path: str | Path = DEFAULT_TARGETS_PATH, start_time: float = 0.0) -> None:
        payload = json.loads(Path(targets_path).read_text(encoding="utf-8"))
        self.victims = payload["victims"]
        self.extraction_zone = payload["extraction_zone"]
        self.rescued_victim_ids.clear()
        self.detected_victim_ids.clear()
        self.current_target_id = None
        self.events = [{"event": "mission_started", "time": round(float(start_time), 3)}]
        self.start_time = float(start_time)
        self.state = MissionState.SEARCH

    def _mission_elapsed(self, current_time: float) -> float:
        return float(current_time) - self.start_time

    def _unrescued_victims(self) -> list[dict[str, Any]]:
        return [victim for victim in self.victims if victim["id"] not in self.rescued_victim_ids]

    def _victim_by_id(self, victim_id: str | None) -> dict[str, Any] | None:
        if victim_id is None:
            return None
        for victim in self.victims:
            if victim["id"] == victim_id:
                return victim
        return None

    def select_nearest_unrescued_victim(self, rover_position: list[float] | tuple[float, ...]) -> dict[str, Any] | None:
        unrescued = self._unrescued_victims()
        if not unrescued:
            return None
        return min(unrescued, key=lambda victim: distance_between_points(rover_position, victim["position"]))

    def update_state(
        self,
        rover_position: list[float] | tuple[float, ...],
        current_time: float,
        visible_victims: list[dict[str, Any]] | None = None,
        rescue_confirmed_id: str | None = None,
    ) -> MissionState:
        if self.state in {MissionState.COMPLETE, MissionState.TIMEOUT}:
            return self.state

        if self._mission_elapsed(current_time) >= self.mission_timeout_s:
            self.state = MissionState.TIMEOUT
            self.events.append({"event": "mission_timeout", "time": round(float(current_time), 3)})
            return self.state

        for victim in visible_victims or []:
            victim_id = victim["id"]
            if victim_id not in self.detected_victim_ids and victim_id not in self.rescued_victim_ids:
                self.detected_victim_ids.add(victim_id)
                self.events.append(
                    {
                        "event": "victim_detected",
                        "victim_id": victim_id,
                        "time": round(float(current_time), 3),
                    }
                )

        if rescue_confirmed_id is not None:
            self.mark_target_rescued(rescue_confirmed_id, current_time)
            self.state = MissionState.NEXT_TARGET

        if len(self.rescued_victim_ids) == len(self.victims):
            self.current_target_id = None
            extraction_position = self.extraction_zone["position"]
            extraction_radius = float(self.extraction_zone["radius"])
            if point_in_radius(rover_position, extraction_position, extraction_radius):
                self.state = MissionState.COMPLETE
                self.events.append({"event": "mission_complete", "time": round(float(current_time), 3)})
            else:
                self.state = MissionState.RETURN_TO_BASE
            return self.state

        if self.state in {MissionState.SEARCH, MissionState.NEXT_TARGET, MissionState.START}:
            target = self.select_nearest_unrescued_victim(rover_position)
            self.current_target_id = target["id"] if target else None
            self.state = MissionState.APPROACH_TARGET if target else MissionState.RETURN_TO_BASE

        current_target = self._victim_by_id(self.current_target_id)
        if current_target and point_in_radius(rover_position, current_target["position"], float(current_target["rescue_radius"])):
            self.state = MissionState.CONFIRM_RESCUE
        elif current_target:
            self.state = MissionState.APPROACH_TARGET

        return self.state

    def get_current_target(self, rover_position: list[float] | tuple[float, ...] | None = None) -> dict[str, Any] | None:
        if self.state == MissionState.RETURN_TO_BASE:
            return {"id": "extraction_zone", **self.extraction_zone}

        target = self._victim_by_id(self.current_target_id)
        if target is None and rover_position is not None:
            target = self.select_nearest_unrescued_victim(rover_position)
            self.current_target_id = target["id"] if target else None
        return target

    def mark_target_rescued(self, victim_id: str, current_time: float) -> None:
        if victim_id in self.rescued_victim_ids:
            return
        self.rescued_victim_ids.add(victim_id)
        self.events.append(
            {
                "event": "victim_rescued",
                "victim_id": victim_id,
                "time": round(float(current_time), 3),
            }
        )
        if victim_id == self.current_target_id:
            self.current_target_id = None

    def mission_complete(self) -> bool:
        return self.state == MissionState.COMPLETE


def initialize_mission(targets_path: str | Path = DEFAULT_TARGETS_PATH, start_time: float = 0.0) -> MissionPlanner:
    planner = MissionPlanner()
    planner.initialize_mission(targets_path=targets_path, start_time=start_time)
    return planner


def update_state(
    planner: MissionPlanner,
    rover_position: list[float] | tuple[float, ...],
    current_time: float,
    visible_victims: list[dict[str, Any]] | None = None,
    rescue_confirmed_id: str | None = None,
) -> MissionState:
    return planner.update_state(rover_position, current_time, visible_victims, rescue_confirmed_id)


def get_current_target(
    planner: MissionPlanner,
    rover_position: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any] | None:
    return planner.get_current_target(rover_position)


def mark_target_rescued(planner: MissionPlanner, victim_id: str, current_time: float) -> None:
    planner.mark_target_rescued(victim_id, current_time)


def mission_complete(planner: MissionPlanner) -> bool:
    return planner.mission_complete()
