import numpy as np

from plant_analysis.utils.so3_utils import (
    euler_to_quaternion,
    euler_to_rotation,
    geodesic_distance,
    hat_map,
    logm_so3,
    quaternion_to_euler,
    quaternion_to_rotation,
    quaternion_to_ypr,
    quaternion_to_yrp,
    rotation_to_quaternion,
    rotation_to_ypr,
    rotation_to_yrp,
    vee_map,
    ypr_to_rotation,
    yrp_to_rotation,
)


def test_hat_and_vee_are_inverse():
    vector = np.array([1.0, 2.0, 3.0])
    assert np.allclose(vee_map(hat_map(vector)), vector)


def test_rotation_log_zero_for_identity():
    assert np.allclose(logm_so3(np.eye(3)), np.zeros(3))


def test_geodesic_distance_positive_for_rotation():
    rotation = ypr_to_rotation(0, 10, 0, degrees=True)
    assert geodesic_distance(np.eye(3), rotation) > 0


def test_ypr_to_rotation_uses_rz_ry_rx_order():
    angles = (35.0, 12.0, -7.0)
    yaw, pitch, roll = np.deg2rad(angles)
    rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    ry = np.array(
        [[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0.0, np.cos(pitch)]]
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(roll), -np.sin(roll)], [0.0, np.sin(roll), np.cos(roll)]]
    )
    assert np.allclose(ypr_to_rotation(*angles, degrees=True), rz @ ry @ rx)


def test_ypr_to_rotation_and_back_are_inverse():
    angles = (35.0, 12.0, -7.0)
    rotation = ypr_to_rotation(*angles, degrees=True)
    recovered = rotation_to_ypr(rotation, degrees=True)
    assert np.allclose(recovered, angles, atol=1e-9)


def test_yrp_to_rotation_and_back_are_inverse():
    angles = (35.0, -7.0, 12.0)
    rotation = yrp_to_rotation(*angles, degrees=True)
    recovered = rotation_to_yrp(rotation, degrees=True)
    assert np.allclose(recovered, angles, atol=1e-9)


def test_quaternion_and_rotation_are_inverse():
    rotation = ypr_to_rotation(0.3, 0.2, -0.1)
    quaternion = rotation_to_quaternion(rotation)
    assert np.allclose(quaternion_to_rotation(quaternion), rotation)
    assert np.isclose(np.linalg.norm(quaternion), 1.0)


def test_quaternion_to_ypr_and_yrp_match_rotation_extractors():
    ypr_angles = (0.25, 0.15, -0.35)
    ypr_quaternion = rotation_to_quaternion(ypr_to_rotation(*ypr_angles))
    assert np.allclose(quaternion_to_ypr(ypr_quaternion), ypr_angles)

    yrp_angles = (0.25, -0.35, 0.15)
    yrp_quaternion = rotation_to_quaternion(yrp_to_rotation(*yrp_angles))
    assert np.allclose(quaternion_to_yrp(yrp_quaternion), yrp_angles)


def test_euler_helpers_default_to_ypr_and_support_yrp():
    rotation = euler_to_rotation(1.0, 2.0, 3.0, degrees=True)
    quaternion = euler_to_quaternion(1.0, 2.0, 3.0, degrees=True)
    assert np.allclose(quaternion_to_rotation(quaternion), rotation)
    assert np.allclose(quaternion_to_euler(quaternion, degrees=True), (1.0, 2.0, 3.0))

    yrp_quaternion = euler_to_quaternion(1.0, 3.0, 2.0, degrees=True, order="yrp")
    assert np.allclose(
        quaternion_to_euler(yrp_quaternion, degrees=True, order="yrp"), (1.0, 3.0, 2.0)
    )
