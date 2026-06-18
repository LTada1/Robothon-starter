from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


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

    def save_mission_summary(self, summary: dict[str, Any]) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / "mission_summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return path

    def save_trajectory_plot(self, victims: list[dict[str, Any]], extraction_zone: dict[str, Any]) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / "trajectory.png"
        width, height = 1200, 850
        margin = 80
        image = Image.new("RGB", (width, height), (246, 248, 250))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        points = [(float(row["x"]), float(row["y"])) for row in self.trajectory]
        victim_points = [(float(victim["position"][0]), float(victim["position"][1])) for victim in victims]
        extraction = (float(extraction_zone["position"][0]), float(extraction_zone["position"][1]))
        all_points = points + victim_points + [extraction]
        if not all_points:
            all_points = [(0.0, 0.0)]

        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        min_x, max_x = min(xs) - 0.8, max(xs) + 0.8
        min_y, max_y = min(ys) - 0.8, max(ys) + 0.8
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)

        def project(point: tuple[float, float]) -> tuple[int, int]:
            x = margin + int((point[0] - min_x) / span_x * (width - 2 * margin))
            y = height - margin - int((point[1] - min_y) / span_y * (height - 2 * margin))
            return x, y

        draw.rectangle((margin, margin, width - margin, height - margin), outline=(190, 196, 203), width=2)
        draw.text((margin, 28), "Autonomous Disaster Response Rover - Mission Trajectory", fill=(20, 28, 36), font=font)

        for index in range(6):
            gx = margin + int(index * (width - 2 * margin) / 5)
            gy = margin + int(index * (height - 2 * margin) / 5)
            draw.line((gx, margin, gx, height - margin), fill=(226, 230, 234))
            draw.line((margin, gy, width - margin, gy), fill=(226, 230, 234))

        if len(points) > 1:
            projected = [project(point) for point in points]
            draw.line(projected, fill=(31, 111, 235), width=5)
            for point in projected[:: max(1, len(projected) // 24)]:
                draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=(31, 111, 235))

        if points:
            start = project(points[0])
            end = project(points[-1])
            draw.ellipse((start[0] - 10, start[1] - 10, start[0] + 10, start[1] + 10), fill=(9, 105, 218))
            draw.text((start[0] + 12, start[1] - 6), "start", fill=(9, 105, 218), font=font)
            draw.ellipse((end[0] - 10, end[1] - 10, end[0] + 10, end[1] + 10), fill=(27, 120, 55))
            draw.text((end[0] + 12, end[1] - 6), "finish", fill=(27, 120, 55), font=font)

        extraction_xy = project(extraction)
        radius_px = 24
        draw.ellipse(
            (
                extraction_xy[0] - radius_px,
                extraction_xy[1] - radius_px,
                extraction_xy[0] + radius_px,
                extraction_xy[1] + radius_px,
            ),
            outline=(27, 120, 55),
            width=4,
        )
        draw.text((extraction_xy[0] + 28, extraction_xy[1] - 8), "extraction", fill=(27, 120, 55), font=font)

        for victim in victims:
            px, py = project((float(victim["position"][0]), float(victim["position"][1])))
            draw.rectangle((px - 9, py - 9, px + 9, py + 9), fill=(251, 188, 5), outline=(120, 82, 0))
            draw.text((px + 12, py - 8), victim["id"], fill=(92, 64, 0), font=font)

        legend_x = width - margin - 270
        legend_y = margin + 20
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 250, legend_y + 110), radius=8, fill=(255, 255, 255), outline=(202, 208, 214))
        draw.line((legend_x + 18, legend_y + 28, legend_x + 68, legend_y + 28), fill=(31, 111, 235), width=5)
        draw.text((legend_x + 78, legend_y + 20), "rover path", fill=(20, 28, 36), font=font)
        draw.rectangle((legend_x + 25, legend_y + 50, legend_x + 41, legend_y + 66), fill=(251, 188, 5), outline=(120, 82, 0))
        draw.text((legend_x + 78, legend_y + 50), "victim target", fill=(20, 28, 36), font=font)
        draw.ellipse((legend_x + 24, legend_y + 78, legend_x + 44, legend_y + 98), outline=(27, 120, 55), width=3)
        draw.text((legend_x + 78, legend_y + 80), "extraction zone", fill=(20, 28, 36), font=font)

        image.save(path)
        return path

    def save_post_run_artifacts(
        self,
        *,
        score_report: dict[str, Any],
        dataset_summary: dict[str, Any],
        victims: list[dict[str, Any]],
        extraction_zone: dict[str, Any],
    ) -> tuple[Path, Path]:
        trajectory_path = self.save_trajectory_plot(victims, extraction_zone)
        mission_summary = {
            "project": score_report.get("project", "Autonomous Disaster Response Rover"),
            "mission_success": score_report.get("mission_success", False),
            "final_score": score_report.get("final_score", 0),
            "victims_rescued": score_report.get("victims_rescued", 0),
            "total_victims": score_report.get("total_victims", len(victims)),
            "mission_time_s": score_report.get("mission_time_s", 0),
            "distance_traveled_m": score_report.get("distance_traveled_m", 0),
            "collision_count": score_report.get("collision_count", 0),
            "hazard_entries": score_report.get("hazard_entries", 0),
            "extraction_reached": score_report.get("extraction_reached", False),
            "dataset_summary": dataset_summary,
            "artifacts": {
                "trajectory_png": str(trajectory_path),
                "trajectory_csv": str(self.logs_dir / "trajectory.csv"),
                "score_report": str(self.logs_dir / "score_report.json"),
            },
        }
        summary_path = self.save_mission_summary(mission_summary)
        return trajectory_path, summary_path

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
