from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = [
    "euler_to_quaternion",
    "euler_to_rotation",
    "geodesic_distance",
    "hat_map",
    "logm_so3",
    "quaternion_to_euler",
    "quaternion_to_rotation",
    "rotation_to_euler",
    "rotation_to_quaternion",
    "vee_map",
    "ypr_to_quaternion",
    "ypr_to_rotation",
    "yrp_to_quaternion",
    "yrp_to_rotation",
    "quaternion_to_ypr",
    "quaternion_to_yrp",
    "rotation_to_ypr",
    "rotation_to_yrp",
]


def _angle(value: float, degrees: bool) -> float:
    return float(np.deg2rad(value) if degrees else value)


def _maybe_degrees(values: tuple[float, float, float], degrees: bool) -> tuple[float, float, float]:
    if not degrees:
        return values
    return tuple(float(np.rad2deg(value)) for value in values)


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("quaternion norm must be non-zero")
    return q / norm


def _axis_rotation_x(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _axis_rotation_y(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _axis_rotation_z(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def hat_map(vector: np.ndarray) -> np.ndarray:
    """Map a 3-vector to the skew-symmetric matrix that represents cross product."""
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def vee_map(matrix: np.ndarray) -> np.ndarray:
    """Map a skew-symmetric matrix back to its 3-vector representation."""
    matrix = np.asarray(matrix, dtype=float)
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


def ypr_to_rotation(yaw: float, pitch: float, roll: float, degrees: bool = False) -> np.ndarray:
    """Convert yaw-pitch-roll angles to an SO(3) matrix using Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    y = _angle(yaw, degrees)
    p = _angle(pitch, degrees)
    r = _angle(roll, degrees)
    return _axis_rotation_z(y) @ _axis_rotation_y(p) @ _axis_rotation_x(r)


def rotation_to_ypr(matrix: np.ndarray, degrees: bool = False) -> tuple[float, float, float]:
    """Convert an SO(3) matrix to yaw-pitch-roll angles for Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
    cos_pitch = np.cos(pitch)
    if abs(cos_pitch) > 1e-9:
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    else:
        yaw = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
        roll = 0.0
    return _maybe_degrees((yaw, pitch, roll), degrees)


def yrp_to_rotation(yaw: float, roll: float, pitch: float, degrees: bool = False) -> np.ndarray:
    """Convert yaw-roll-pitch angles to an SO(3) matrix using Rz(yaw) @ Rx(roll) @ Ry(pitch)."""
    y = _angle(yaw, degrees)
    r = _angle(roll, degrees)
    p = _angle(pitch, degrees)
    return _axis_rotation_z(y) @ _axis_rotation_x(r) @ _axis_rotation_y(p)


def rotation_to_yrp(matrix: np.ndarray, degrees: bool = False) -> tuple[float, float, float]:
    """Convert an SO(3) matrix to yaw-roll-pitch angles for Rz(yaw) @ Rx(roll) @ Ry(pitch)."""
    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    roll = float(np.arcsin(np.clip(rotation[2, 1], -1.0, 1.0)))
    cos_roll = np.cos(roll)
    if abs(cos_roll) > 1e-9:
        yaw = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
        pitch = float(np.arctan2(-rotation[2, 0], rotation[2, 2]))
    else:
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        pitch = 0.0
    return _maybe_degrees((yaw, roll, pitch), degrees)


def rotation_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert an SO(3) matrix to a unit quaternion in [w, x, y, z] order."""
    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    return _normalize_quaternion(np.array([w, x, y, z]))


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    """Convert a quaternion in [w, x, y, z] order to an SO(3) matrix."""
    w, x, y, z = _normalize_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def ypr_to_quaternion(yaw: float, pitch: float, roll: float, degrees: bool = False) -> np.ndarray:
    """Convert yaw-pitch-roll angles to a unit quaternion in [w, x, y, z] order."""
    return rotation_to_quaternion(ypr_to_rotation(yaw, pitch, roll, degrees=degrees))


def quaternion_to_ypr(quaternion: np.ndarray, degrees: bool = False) -> tuple[float, float, float]:
    """Convert a quaternion in [w, x, y, z] order to yaw-pitch-roll angles."""
    return rotation_to_ypr(quaternion_to_rotation(quaternion), degrees=degrees)


def yrp_to_quaternion(yaw: float, roll: float, pitch: float, degrees: bool = False) -> np.ndarray:
    """Convert yaw-roll-pitch angles to a unit quaternion in [w, x, y, z] order."""
    return rotation_to_quaternion(yrp_to_rotation(yaw, roll, pitch, degrees=degrees))


def quaternion_to_yrp(quaternion: np.ndarray, degrees: bool = False) -> tuple[float, float, float]:
    """Convert a quaternion in [w, x, y, z] order to yaw-roll-pitch angles."""
    return rotation_to_yrp(quaternion_to_rotation(quaternion), degrees=degrees)


def logm_so3(matrix: np.ndarray) -> np.ndarray:
    """Compute the SO(3) logarithm as a rotation vector in axis-angle form."""
    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    cos_theta = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        return vee_map(0.5 * (rotation - rotation.T))
    if np.pi - theta < 1e-7:
        q = rotation_to_quaternion(rotation)
        axis = q[1:]
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return np.zeros(3)
        return theta * axis / norm
    return theta / (2.0 * np.sin(theta)) * vee_map(rotation - rotation.T)


def geodesic_distance(a: np.ndarray, b: np.ndarray | None = None) -> float:
    """Compute SO(3) geodesic distance between rotations, unit [rad]."""
    rel = (
        np.asarray(a, dtype=float)
        if b is None
        else np.asarray(a, dtype=float).T @ np.asarray(b, dtype=float)
    )
    return float(np.linalg.norm(logm_so3(rel)))


def euler_to_rotation(
    first: float,
    second: float,
    third: float,
    degrees: bool = False,
    order: Literal["ypr", "yrp"] = "ypr",
) -> np.ndarray:
    """Convert Euler angles to SO(3); order selects ypr or yrp argument semantics."""
    if order == "ypr":
        return ypr_to_rotation(first, second, third, degrees=degrees)
    return yrp_to_rotation(first, second, third, degrees=degrees)


def rotation_to_euler(
    matrix: np.ndarray, degrees: bool = False, order: Literal["ypr", "yrp"] = "ypr"
) -> tuple[float, float, float]:
    """Convert SO(3) to Euler angles; order selects ypr or yrp return semantics."""
    if order == "ypr":
        return rotation_to_ypr(matrix, degrees=degrees)
    return rotation_to_yrp(matrix, degrees=degrees)


def euler_to_quaternion(
    first: float,
    second: float,
    third: float,
    degrees: bool = False,
    order: Literal["ypr", "yrp"] = "ypr",
) -> np.ndarray:
    """Convert Euler angles to quaternion; order selects ypr or yrp argument semantics."""
    return rotation_to_quaternion(
        euler_to_rotation(first, second, third, degrees=degrees, order=order)
    )


def quaternion_to_euler(
    quaternion: np.ndarray, degrees: bool = False, order: Literal["ypr", "yrp"] = "ypr"
) -> tuple[float, float, float]:
    """Convert quaternion to Euler angles; order selects ypr or yrp return semantics."""
    return rotation_to_euler(quaternion_to_rotation(quaternion), degrees=degrees, order=order)
