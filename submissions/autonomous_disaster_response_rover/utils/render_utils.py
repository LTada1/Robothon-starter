from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _require_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is required for rendering. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return mujoco


def _require_imageio() -> Any:
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise RuntimeError(
            "imageio[ffmpeg] is required for video output. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return iio


class DemoRenderer:
    def __init__(
        self,
        model: Any,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        overview_camera_name: str = "mission_overview_camera",
    ) -> None:
        self.mujoco = _require_mujoco()
        self.model = model
        self.width = width
        self.height = height
        self.fps = fps
        self.renderer = self.mujoco.Renderer(model, width=width, height=height)
        self.follow_camera = self.mujoco.MjvCamera()
        self.overview_camera_name = overview_camera_name
        self.frames: list[np.ndarray] = []
        self.font = ImageFont.load_default()

    def setup_camera(self) -> None:
        self.follow_camera.type = self.mujoco.mjtCamera.mjCAMERA_FREE
        self.follow_camera.distance = 3.0
        self.follow_camera.azimuth = 135.0
        self.follow_camera.elevation = -28.0

    def _update_follow_camera(self, data: Any) -> None:
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "rover")
        if body_id >= 0:
            self.follow_camera.lookat[:] = data.xpos[body_id]
            self.follow_camera.lookat[2] = 0.25

    def capture_frame(self, data: Any, view: str = "follow") -> np.ndarray:
        if view == "overview":
            self.renderer.update_scene(data, camera=self.overview_camera_name)
        else:
            self._update_follow_camera(data)
            self.renderer.update_scene(data, camera=self.follow_camera)

        frame = self.renderer.render().copy()
        self.frames.append(frame)
        return frame

    def _overlay_text(
        self,
        frame: np.ndarray,
        overlay_lines: list[str] | None = None,
        banner: str | None = None,
    ) -> np.ndarray:
        if not overlay_lines and not banner:
            return frame

        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image, "RGBA")

        if overlay_lines:
            line_height = 24
            panel_width = 360
            panel_height = 26 + line_height * len(overlay_lines)
            draw.rounded_rectangle((18, 18, 18 + panel_width, 18 + panel_height), radius=8, fill=(8, 12, 16, 190))
            draw.text((34, 30), "Autonomous Disaster Response Rover", fill=(255, 214, 128, 255), font=self.font)
            y = 56
            for line in overlay_lines:
                draw.text((34, y), line, fill=(238, 244, 248, 255), font=self.font)
                y += line_height

        if banner:
            text_box = draw.textbbox((0, 0), banner, font=self.font)
            text_width = text_box[2] - text_box[0]
            x0 = max(18, (self.width - text_width) // 2 - 32)
            x1 = min(self.width - 18, (self.width + text_width) // 2 + 32)
            y0 = self.height - 82
            y1 = self.height - 30
            draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=(5, 28, 20, 210))
            draw.text(((self.width - text_width) // 2, y0 + 18), banner, fill=(150, 255, 190, 255), font=self.font)

        return np.asarray(image)

    def capture_split_frame(
        self,
        data: Any,
        overlay_lines: list[str] | None = None,
        banner: str | None = None,
    ) -> np.ndarray:
        self._update_follow_camera(data)
        self.renderer.update_scene(data, camera=self.follow_camera)
        follow = self.renderer.render().copy()

        self.renderer.update_scene(data, camera=self.overview_camera_name)
        overview = self.renderer.render().copy()

        split = np.concatenate([follow[:, : self.width // 2], overview[:, self.width // 2 :]], axis=1)
        split = self._overlay_text(split, overlay_lines=overlay_lines, banner=banner)
        self.frames.append(split)
        return split

    def write_video(self, output_path: str | Path) -> Path:
        if not self.frames:
            raise ValueError("No frames captured; cannot write demo video.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        iio = _require_imageio()
        try:
            iio.imwrite(output_path, np.asarray(self.frames), fps=self.fps, codec="libx264")
            return output_path
        except Exception:
            fallback = output_path.with_suffix(".gif")
            iio.imwrite(fallback, np.asarray(self.frames), fps=self.fps)
            return fallback

    def cleanup(self) -> None:
        close = getattr(self.renderer, "close", None)
        if callable(close):
            close()


def initialize_viewer(model: Any, width: int = 1280, height: int = 720, fps: int = 30) -> DemoRenderer:
    renderer = DemoRenderer(model, width=width, height=height, fps=fps)
    renderer.setup_camera()
    return renderer


def setup_camera(renderer: DemoRenderer) -> None:
    renderer.setup_camera()


def capture_frame(renderer: DemoRenderer, data: Any, view: str = "follow") -> np.ndarray:
    return renderer.capture_frame(data, view=view)


def write_video(renderer: DemoRenderer, output_path: str | Path) -> Path:
    return renderer.write_video(output_path)


def cleanup(renderer: DemoRenderer) -> None:
    renderer.cleanup()
