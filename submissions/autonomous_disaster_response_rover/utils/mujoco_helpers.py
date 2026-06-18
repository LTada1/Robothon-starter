from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def _require_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is required for simulation helpers. Install dependencies with "
            "`python -m pip install -r requirements.txt` from the repository root."
        ) from exc
    return mujoco


def load_model(xml_path: str | Path) -> tuple[Any, Any]:
    mujoco = _require_mujoco()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def get_body_position(model: Any, data: Any, body_name: str) -> list[float]:
    mujoco = _require_mujoco()
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name}")
    return data.xpos[body_id].copy().tolist()


def get_body_orientation(model: Any, data: Any, body_name: str) -> float:
    mujoco = _require_mujoco()
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name}")

    matrix = data.xmat[body_id].reshape(3, 3)
    return math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))


def _actuator_id(model: Any, name: str) -> int:
    mujoco = _require_mujoco()
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id < 0:
        raise ValueError(f"Actuator not found: {name}")
    return int(actuator_id)


def set_wheel_velocity(model: Any, data: Any, left_velocity: float, right_velocity: float) -> None:
    left_actuators = ("left_front_wheel_velocity", "left_rear_wheel_velocity")
    right_actuators = ("right_front_wheel_velocity", "right_rear_wheel_velocity")

    for name in left_actuators:
        data.ctrl[_actuator_id(model, name)] = left_velocity
    for name in right_actuators:
        data.ctrl[_actuator_id(model, name)] = right_velocity


def set_rescue_deployer(model: Any, data: Any, command: float) -> None:
    actuator = _actuator_id(model, "rescue_deployer_actuator")
    low, high = float(model.actuator_ctrlrange[actuator][0]), float(model.actuator_ctrlrange[actuator][1])
    data.ctrl[actuator] = max(low, min(high, float(command)))


def get_rescue_deployer_state(model: Any, data: Any) -> float:
    mujoco = _require_mujoco()
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rescue_deployer_joint")
    if joint_id < 0:
        raise ValueError("Joint not found: rescue_deployer_joint")
    qpos_index = int(model.jnt_qposadr[joint_id])
    return float(data.qpos[qpos_index])


def read_sensor_values(model: Any, data: Any) -> dict[str, list[float] | float]:
    values: dict[str, list[float] | float] = {}
    cursor = 0

    for sensor_id in range(model.nsensor):
        name = model.sensor(sensor_id).name
        dim = int(model.sensor_dim[sensor_id])
        raw = data.sensordata[cursor : cursor + dim].copy().tolist()
        values[name] = raw[0] if dim == 1 else raw
        cursor += dim

    return values
