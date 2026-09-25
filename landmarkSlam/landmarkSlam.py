"""Landmark SLAM toy problem with GTSAM.

A robot observes a cube (8 landmarks) from 5 poses. Noisy ICP gives odometry,
noisy depth gives landmark observations. Joint optimization refines both.

Run:  python landmarkSlam.py            # headless (no Open3D windows)
      python landmarkSlam.py --show     # with Open3D visualisation
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import gtsam

HERE = Path(__file__).resolve().parent

# g2o information matrices chosen to match the injected sensor noise:
# ICP odometry is about 2 m uncertain, while landmark observations are about
# 0.15 m uncertain. Landmark vertices are points, so their information matrix
# is 3x3 (six values in g2o's upper-triangular format), not a 6x6 SE(3) matrix.
ODOM_INFO = "0.2 0 0 0 0 0 0.2 0 0 0 0 0.2 0 0 0 0.2 0 0 0.2 0 0.2"
LANDMARK_INFO = "44 0 0 44 0 44"


def getVertices():
    """Cube corners + Open3D spheres for visualisation."""
    import open3d as o3d

    points = [
        [0, 8, 8],
        [0, 0, 8],
        [0, 0, 0],
        [0, 8, 0],
        [8, 8, 8],
        [8, 0, 8],
        [8, 0, 0],
        [8, 8, 0],
    ]
    vertices = []
    for ele in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
        sphere.paint_uniform_color([0.9, 0.2, 0])
        trans = np.identity(4)
        trans[0, 3], trans[1, 3], trans[2, 3] = ele
        sphere.transform(trans)
        vertices.append(sphere)
    return vertices, points


def getCloud(cube, color):
    """Point-cloud spheres for one cube observation."""
    import open3d as o3d

    vertices = []
    for ele in cube:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        sphere.paint_uniform_color(color)
        trans = np.identity(4)
        trans[0, 3], trans[1, 3], trans[2, 3] = ele
        sphere.transform(trans)
        vertices.append(sphere)
    return vertices


def getFrames():
    """Five ground-truth robot poses (x, y, z, yaw_deg)."""
    import open3d as o3d

    poses = [
        [-12, 0, 0, 0],
        [-10, -4, 0, 30],
        [-8, -8, 0, 60],
        [-4, -12, 0, 75],
        [0, -16, 0, 80],
    ]
    frames = []
    for pose in poses:
        T = np.identity(4)
        T[0, 3], T[1, 3], T[2, 3] = pose[0], pose[1], pose[2]
        T[0:3, 0:3] = R.from_euler("z", pose[3], degrees=True).as_matrix()
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.2)
        frame.transform(T)
        frames.append(frame)
    return frames, poses


def visualizeData(vertices, frames):
    import open3d as o3d

    o3d.visualization.draw_geometries(vertices + frames)


def pose_to_matrix(pose):
    """[x, y, z, yaw_deg] -> 4x4 transform."""
    T = np.identity(4)
    T[0, 3], T[1, 3], T[2, 3] = pose[0], pose[1], pose[2]
    T[0:3, 0:3] = R.from_euler("z", pose[3], degrees=True).as_matrix()
    return T


def getLocalCubes(points, poses):
    """Cube corners expressed in each robot frame."""
    points = np.asarray(points, dtype=float)
    cubes = np.zeros((len(poses), len(points), 3))
    for i, pose in enumerate(poses):
        T = pose_to_matrix(pose)
        T_inv = np.linalg.inv(T)
        for j, pt in enumerate(points):
            pt_h = np.append(pt, 1.0)
            cubes[i, j] = (T_inv @ pt_h)[:3]
    return cubes


def addNoiseCubes(cubes, noise=0.15, seed=42):
    """Additive Gaussian noise on local landmark observations."""
    rng = np.random.default_rng(seed)
    return cubes + rng.normal(0.0, noise, cubes.shape)


def draw_registration_result(source, target, transformation):
    import open3d as o3d

    src = copy.deepcopy(source)
    tgt = copy.deepcopy(target)
    src.paint_uniform_color([1, 0.706, 0])
    tgt.paint_uniform_color([0, 0.651, 0.929])
    src.transform(transformation)
    o3d.visualization.draw_geometries([src, tgt])


def registerCubes(trans, cubes, show=True):
    """Re-project all local cubes into the first frame (for visual checks)."""
    import open3d as o3d

    clouds = [
        getCloud(cubes[0], [0.9, 0.2, 0]),
        getCloud(cubes[1], [0, 0.2, 0.9]),
        getCloud(cubes[2], [0.2, 0.9, 0]),
        getCloud(cubes[3], [0.5, 0, 0.95]),
        getCloud(cubes[4], [0.9, 0.45, 0]),
    ]
    t01, t12, t23, t34 = trans[0], trans[1], trans[2], trans[3]
    t02 = t01 @ t12
    t03 = t02 @ t23
    t04 = t03 @ t34
    for mesh in clouds[1]:
        mesh.transform(t01)
    for mesh in clouds[2]:
        mesh.transform(t02)
    for mesh in clouds[3]:
        mesh.transform(t03)
    for mesh in clouds[4]:
        mesh.transform(t04)
    if show:
        o3d.visualization.draw_geometries(sum(clouds, []))


def icpTransformations(cubes, show=False):
    """Relative odometry (2 w.r.t. 1, ...) via point-to-point ICP with known data association."""
    import open3d as o3d

    pcds = []
    for cube in cubes:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(cube))
        pcds.append(pcd)

    corr = o3d.utility.Vector2iVector(np.array([(i, i) for i in range(cubes.shape[1])]))
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    pairs = [
        p2p.compute_transformation(pcds[1], pcds[0], corr),
        p2p.compute_transformation(pcds[2], pcds[1], corr),
        p2p.compute_transformation(pcds[3], pcds[2], corr),
        p2p.compute_transformation(pcds[4], pcds[3], corr),
    ]
    if show:
        draw_registration_result(pcds[1], pcds[0], pairs[0])
    return np.array(pairs)


# --- g2o I/O (kept g2o-text for teaching; optimisation itself uses GTSAM) ---


def _quat_of(T):
    return R.from_matrix(T[0:3, 0:3]).as_quat()  # x, y, z, w


def writeRobotPose(trans, g2o):
    start = [-12, 0, 0, 0]
    Tw_1 = pose_to_matrix(start)
    poses_w = [Tw_1]
    for T in trans:
        poses_w.append(poses_w[-1] @ T)
    for i, Tw in enumerate(poses_w):
        qx, qy, qz, qw = _quat_of(Tw)
        g2o.write(
            f"VERTEX_SE3:QUAT {i + 1} {Tw[0, 3]} {Tw[1, 3]} {Tw[2, 3]} "
            f"{qx} {qy} {qz} {qw}\n"
        )


def writeOdom(trans, g2o):
    for i, T in enumerate(trans):
        qx, qy, qz, qw = _quat_of(T)
        g2o.write(
            f"EDGE_SE3:QUAT {i + 1} {i + 2} {T[0, 3]} {T[1, 3]} {T[2, 3]} "
            f"{qx} {qy} {qz} {qw} {ODOM_INFO}\n"
        )


def writeCubeVertices(cubes, g2o):
    Tw_1 = pose_to_matrix([-12, 0, 0, 0])
    for i, pt in enumerate(np.asarray(cubes[0])):
        pt_w = (Tw_1 @ np.append(pt, 1.0))[:3]
        g2o.write(f"VERTEX_TRACKXYZ {i + 6} {pt_w[0]} {pt_w[1]} {pt_w[2]}\n")


def writeLandmarkEdge(cubes, g2o):
    # EDGE_SE3_TRACKXYZ pose_id point_id offset_id x y z info(3x3 upper triangle)
    for i, cube in enumerate(cubes):
        for j, (x, y, z) in enumerate(cube):
            g2o.write(
                f"EDGE_SE3_TRACKXYZ {i + 1} {j + 6} 0 {x} {y} {z} {LANDMARK_INFO}\n"
            )


def writeG2o(trans, cubes, path="noise.g2o"):
    with open(path, "w") as g2o:
        # EDGE_SE3_TRACKXYZ refers to a zero sensor offset (parameter id 0).
        g2o.write("PARAMS_SE3OFFSET 0 0 0 0 0 0 0 1\n")
        g2o.write("# Robot poses\n\n")
        writeRobotPose(trans, g2o)
        g2o.write("\n# Cube vertices (points, not poses)\n\n")
        writeCubeVertices(cubes, g2o)
        g2o.write("\n# Odometry edges\n\n")
        writeOdom(trans, g2o)
        g2o.write("\n# Landmark edges\n\n")
        writeLandmarkEdge(cubes, g2o)
        g2o.write("\nFIX 1\n")
    return str(path)


def optimize(noise_path="noise.g2o", out_path="opt_gtsam.g2o"):
    """Optimise a g2o file with GTSAM's Levenberg-Marquardt.

    ``FIX 1`` is understood by the g2o command-line tools, but
    :func:`gtsam.readG2o` does not turn it into a factor. Add a small prior
    explicitly so the 3D graph has a fixed world frame, then write ``FIX 1``
    back into the result for g2o-compatible tools.
    """
    graph, values = gtsam.readG2o(str(noise_path), is3D=True)
    first_pose = values.atPose3(1)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6))
    graph.add(gtsam.PriorFactorPose3(1, first_pose, prior_noise))

    result = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
    print(f"initial error: {graph.error(values):.2f} -> final: {graph.error(result):.2f}")
    gtsam.writeG2o(graph, result, str(out_path))
    # gtsam.writeG2o writes vertices and factors but not the explicit prior.
    with open(out_path, "a") as g2o:
        g2o.write("\nFIX 1\n")
    return str(out_path)


def readG2o(fileName):
    """Optimised robot poses (first 5 SE3 vertices) -> (5, 4, 4) array."""
    poses = []
    with open(fileName) as f:
        for line in f:
            if "VERTEX_SE3:QUAT" not in line:
                continue
            parts = line.split()
            _, ind, x, y, z, qx, qy, qz, qw = parts[:9]
            if int(ind) > 5:
                continue
            T = np.identity(4)
            T[0, 3], T[1, 3], T[2, 3] = float(x), float(y), float(z)
            T[0:3, 0:3] = R.from_quat([float(qx), float(qy), float(qz), float(qw)]).as_matrix()
            poses.append(T)
    return np.asarray(poses)


def readLandmarks(fileName):
    """Optimised landmark vertices from ``VERTEX_TRACKXYZ`` lines."""
    landmarks = []
    with open(fileName) as f:
        for line in f:
            if line.startswith("VERTEX_TRACKXYZ"):
                parts = line.split()
                landmarks.append([float(value) for value in parts[2:5]])
    return np.asarray(landmarks)


def getRelativeEdge(poses):
    return np.array(
        [
            np.linalg.inv(poses[0]) @ poses[1],
            np.linalg.inv(poses[1]) @ poses[2],
            np.linalg.inv(poses[2]) @ poses[3],
            np.linalg.inv(poses[3]) @ poses[4],
        ]
    )


def main(show=False, out_dir=None):
    out_dir = Path(out_dir) if out_dir else HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    vertices, points = getVertices()
    frames, poses = getFrames()
    if show:
        visualizeData(vertices, frames)

    gt_cubes = getLocalCubes(points, poses)
    noisy_high = addNoiseCubes(gt_cubes, noise=1.8, seed=42)  # for ICP odometry
    noisy_low = addNoiseCubes(gt_cubes, noise=0.15, seed=1)  # for landmarks

    trans = icpTransformations(noisy_high, show=show)
    if show:
        registerCubes(trans, noisy_low, show=True)

    noise_path = writeG2o(trans, noisy_low, path=out_dir / "noise.g2o")
    opt_path = optimize(noise_path, out_path=out_dir / "opt_gtsam.g2o")

    opt_poses = readG2o(opt_path)
    print(f"Optimised {len(opt_poses)} poses -> {opt_path}")
    if show:
        registerCubes(getRelativeEdge(opt_poses), noisy_low, show=True)
    return opt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landmark SLAM toy problem")
    parser.add_argument("--show", action="store_true", help="Open3D visualisation")
    parser.add_argument("--out-dir", default=None, help="Where to write .g2o files")
    args = parser.parse_args()
    has_display = (
        os.name == "nt"
        or sys.platform == "darwin"
        or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    )
    if args.show and not has_display:
        print("Warning: no display detected, running headless.")
        args.show = False
    main(show=args.show, out_dir=args.out_dir)
