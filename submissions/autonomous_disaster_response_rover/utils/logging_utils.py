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
        collisions: int = 0,
        hazards: int = 0,
        completion_status: bool = False,
    ) -> None:
        self.trajectory.append(
            {
                "timestamp": round(float(timestamp), 3),
                "x": round(float(rover_position[0]), 5),
                "y": round(float(rover_position[1]), 5),
                "z": round(float(rover_position[2]) if len(rover_position) > 2 else 0.0, 5),
                "mission_state": mission_state,
                "collisions": int(collisions),
                "hazards": int(hazards),
                "completion_status": bool(completion_status),
            }
        )

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
        fieldnames = ["timestamp", "x", "y", "z", "mission_state", "collisions", "hazards", "completion_status"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.trajectory)
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
    _logger(logs_dir).log_position(timestamp, rover_position, mission_state, collisions, hazards, completion_status)


def save_score_report(score_report: dict[str, Any], logs_dir: str | Path = "logs") -> Path:
    return _logger(logs_dir).save_score_report(score_report)
