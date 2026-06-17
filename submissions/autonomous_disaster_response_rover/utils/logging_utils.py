from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class MissionLogger:
    def __init__(self, logs_dir: str | Path) -> None:
        self.logs_dir = Path(logs_dir)
        self.events: list[dict[str, Any]] = []
        self.trajectory: list[dict[str, Any]] = []
        self.sensor_data: list[dict[str, Any]] = []

    def log_event(self, event: dict[str, Any] | str, timestamp: float | None = None, **details: Any) -> None:
        if isinstance(event, dict):
            payload = dict(event)
        else:
            payload = {"event": event}
            if timestamp is not None:
                payload["time"] = round(float(timestamp), 3)
        payload.update(details)
        self.events.append(payload)

    def log_position(
        self,
        timestamp: float,
        rover_position: list[float] | tuple[float, ...],
        mission_state: str,
        rover_heading: float | None = None,
        linear_velocity: float | None = None,
        wheel_commands: dict[str, float] | None = None,
        active_target: str | None = None,
        detected_victims: list[str] | None = None,
        obstacle_distances: dict[str, float] | None = None,
        hazard_status: bool = False,
        collisions: int = 0,
        hazards: int = 0,
        completion_status: bool = False,
        recovery_action: str | None = None,
    ) -> None:
        self.trajectory.append(
            {
                "timestamp": round(float(timestamp), 3),
                "x": round(float(rover_position[0]), 5),
                "y": round(float(rover_position[1]), 5),
                "z": round(float(rover_position[2]) if len(rover_position) > 2 else 0.0, 5),
                "heading": round(float(rover_heading), 5) if rover_heading is not None else "",
                "linear_velocity": round(float(linear_velocity), 5) if linear_velocity is not None else "",
                "mission_state": mission_state,
                "active_target": active_target or "",
                "detected_victims": "|".join(detected_victims or []),
                "left_wheel_command": round(float(wheel_commands.get("left", 0.0)), 5) if wheel_commands else "",
                "right_wheel_command": round(float(wheel_commands.get("right", 0.0)), 5) if wheel_commands else "",
                "front_obstacle_distance": round(float(obstacle_distances.get("front", 0.0)), 5) if obstacle_distances else "",
                "left_obstacle_distance": round(float(obstacle_distances.get("left", 0.0)), 5) if obstacle_distances else "",
                "right_obstacle_distance": round(float(obstacle_distances.get("right", 0.0)), 5) if obstacle_distances else "",
                "hazard_status": bool(hazard_status),
                "collisions": int(collisions),
                "hazards": int(hazards),
                "completion_status": bool(completion_status),
                "recovery_action": recovery_action or "",
            }
        )

    def log_sensor_record(self, record: dict[str, Any]) -> None:
        self.sensor_data.append(dict(record))

    def save_mission_log(self, metrics: dict[str, Any]) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / "mission_log.json"
        path.write_text(
            json.dumps({"events": self.events, "metrics": metrics}, indent=2),
            encoding="utf-8",
        )
        return path

    def save_trajectory(self) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / "trajectory.csv"
        fieldnames = [
            "timestamp",
            "x",
            "y",
            "z",
            "heading",
            "linear_velocity",
            "mission_state",
            "active_target",
            "detected_victims",
            "left_wheel_command",
            "right_wheel_command",
            "front_obstacle_distance",
            "left_obstacle_distance",
            "right_obstacle_distance",
            "hazard_status",
            "collisions",
            "hazards",
            "completion_status",
            "recovery_action",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.trajectory)
        return path

    def save_sensor_data(self) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / "sensor_data.csv"
        fieldnames = [
            "timestamp",
            "rover_x",
            "rover_y",
            "rover_z",
            "rover_heading",
            "linear_velocity",
            "mission_state",
            "active_target",
            "detected_victims",
            "lidar_front",
            "lidar_front_left",
            "lidar_front_right",
            "proximity_left",
            "proximity_right",
            "hazard_status",
            "collision_count",
            "left_wheel_command",
            "right_wheel_command",
            "navigation_decision",
            "recovery_action",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.sensor_data)
        return path

    def save_events(self) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / "events.json"
        path.write_text(json.dumps(self.events, indent=2), encoding="utf-8")
        return path

    def save_dataset_summary(self, summary: dict[str, Any]) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / "dataset_summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return path

    def save_score_report(self, score_report: dict[str, Any]) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / "score_report.json"
        path.write_text(json.dumps(score_report, indent=2), encoding="utf-8")
        return path


_DEFAULT_LOGGER: MissionLogger | None = None


def _logger(logs_dir: str | Path = "logs") -> MissionLogger:
    global _DEFAULT_LOGGER
    if _DEFAULT_LOGGER is None or _DEFAULT_LOGGER.logs_dir != Path(logs_dir):
        _DEFAULT_LOGGER = MissionLogger(logs_dir)
    return _DEFAULT_LOGGER


def log_event(event: dict[str, Any] | str, timestamp: float | None = None, logs_dir: str | Path = "logs", **details: Any) -> None:
    _logger(logs_dir).log_event(event, timestamp, **details)


def log_position(
    timestamp: float,
    rover_position: list[float] | tuple[float, ...],
    mission_state: str,
    collisions: int = 0,
    hazards: int = 0,
    completion_status: bool = False,
    logs_dir: str | Path = "logs",
) -> None:
    _logger(logs_dir).log_position(
        timestamp,
        rover_position,
        mission_state,
        collisions=collisions,
        hazards=hazards,
        completion_status=completion_status,
    )


def save_score_report(score_report: dict[str, Any], logs_dir: str | Path = "logs") -> Path:
    return _logger(logs_dir).save_score_report(score_report)
