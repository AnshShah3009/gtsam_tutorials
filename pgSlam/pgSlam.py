"""Pose-graph SLAM toy problem with GTSAM.

A robot drives an oval trajectory. Wheel odometry drifts, loop closures
correct it. This module contains the (deliberately small) helpers used by
``pgSlam.ipynb``.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from gtsam import (
    BetweenFactorPose2,
    LevenbergMarquardtOptimizer,
    LevenbergMarquardtParams,
    NonlinearFactorGraph,
    Pose2,
    PriorFactorPose2,
    Values,
)
from gtsam.noiseModel import Diagonal

# Noise models (sigmas). Loop closures are slightly more confident than odometry,
# mirroring the old g2o info matrices (~500 for odom, ~700 for loop closures,
# since sigma ~= 1 / sqrt(info)).
PRIOR_SIGMAS = np.array([1e-6, 1e-6, 1e-8])
ODOM_SIGMAS = np.array([0.05, 0.05, 0.05])
LOOP_SIGMAS = np.array([0.04, 0.04, 0.04])


def getTheta(X, Y):
    """Heading from the direction of the neighbouring points, one per pose."""

    def heading(x0, y0, x1, y1):
        dx, dy = x1 - x0, y1 - y0
        if np.isclose(dx, 0.0, atol=1e-12) and np.isclose(dy, 0.0, atol=1e-12):
            return 0.0
        if np.isclose(dx, 0.0, atol=1e-12):
            return math.pi / 2 if dy > 0 else 3 * math.pi / 2
        angle = math.atan2(dy, dx)
        return angle if angle >= 0 else angle + 2 * math.pi

    n = len(X)
    theta = [heading(X[0], Y[0], X[1], Y[1])]
    theta.extend(heading(X[i - 1], Y[i - 1], X[i + 1], Y[i + 1]) for i in range(1, n - 1))
    theta.append(heading(X[-2], Y[-2], X[-1], Y[-1]))
    return np.array(theta)


def genTraj():
    """Ground-truth oval: straights + three U-turns, 120 poses."""
    # Forward I
    num = 20
    x_st, y_st, leng = -5.0, -8.0, 9.0
    step = leng / num
    X1 = np.zeros(num)
    Y1 = np.zeros(num)
    X1[0], Y1[0] = x_st, y_st
    for i in range(1, num):
        X1[i] = X1[i - 1] + step
        Y1[i] = y_st

    # U-turn I
    rad, num = 2.5, 20
    x_cen, y_cen = X1[-1], Y1[-1] + rad
    thetas = np.linspace(-math.pi / 2, math.pi / 2, num)
    X2 = x_cen + rad * np.cos(thetas)
    Y2 = y_cen + rad * np.sin(thetas)

    # Backward I
    num, leng = 20, 10.0
    step = leng / num
    X3 = np.zeros(num)
    Y3 = np.zeros(num)
    X3[0], Y3[0] = X2[-1], Y2[-1]
    for i in range(1, num):
        X3[i] = X3[i - 1] - step
        Y3[i] = Y3[0]

    # U-turn II
    rad, num = 2.6, 20
    x_cen, y_cen = X3[-1], Y3[-1] - rad
    thetas = np.linspace(math.pi / 2, 3 * math.pi / 2, num)
    X4 = x_cen + rad * np.cos(thetas)
    Y4 = y_cen + rad * np.sin(thetas)

    # Forward II
    num, leng = 20, 11.0
    step = leng / num
    X5 = np.zeros(num)
    Y5 = np.zeros(num)
    X5[0], Y5[0] = X4[-1], Y4[-1]
    for i in range(1, num):
        X5[i] = X5[i - 1] + step
        Y5[i] = Y5[0]

    # U-turn III
    rad, num = 2.7, 20
    x_cen, y_cen = X5[-1], Y5[-1] + rad
    thetas = np.linspace(-math.pi / 2, math.pi / 2, num)
    X6 = x_cen + rad * np.cos(thetas)
    Y6 = y_cen + rad * np.sin(thetas)

    X = np.concatenate([X1, X2, X3, X4, X5, X6])
    Y = np.concatenate([Y1, Y2, Y3, Y4, Y5, Y6])
    return X, Y, getTheta(X, Y)


def relative_pose2(x1, y1, t1, x2, y2, t2):
    """Relative pose of 2 w.r.t. 1 as a GTSAM Pose2."""
    return Pose2(x1, y1, t1).between(Pose2(x2, y2, t2))


def addNoise(X, Y, THETA, pos_std=0.03, theta_std=0.03, seed=4):
    """Integrate noisy odometry to produce a drifted trajectory.

    Noise is added to each relative motion (not to absolute poses),
    which is why drift accumulates. The first 5 poses are kept noise-free
    so the beginning of the trajectory is easy to compare with the prior.
    """
    rng = np.random.default_rng(seed)
    xN = np.zeros(len(X))
    yN = np.zeros(len(Y))
    tN = np.zeros(len(THETA))
    xN[0], yN[0], tN[0] = X[0], Y[0], THETA[0]

    for i in range(1, len(X)):
        rel = relative_pose2(X[i - 1], Y[i - 1], THETA[i - 1], X[i], Y[i], THETA[i])
        if i < 5:
            dx_n, dy_n, dt_n = rel.x(), rel.y(), rel.theta()
        else:
            dx_n = rel.x() + rng.normal(0.0, pos_std)
            dy_n = rel.y() + rng.normal(0.0, pos_std)
            dt_n = rel.theta() + rng.normal(0.0, theta_std)
        pred = Pose2(xN[i - 1], yN[i - 1], tN[i - 1]).compose(Pose2(dx_n, dy_n, dt_n))
        xN[i], yN[i], tN[i] = pred.x(), pred.y(), pred.theta()

    return xN, yN, tN


def loop_pairs(n_poses=120):
    """Loop pairs used by the tutorial: 0..38 paired with 80..118."""
    if n_poses < 120:
        raise ValueError("The tutorial trajectory needs at least 120 poses")
    return [(i, i + 80) for i in range(0, 40, 2)]


def build_pose2_graph(xN, yN, tN, X_gt, Y_gt, T_gt):
    """Prior + noisy odometry + (near-perfect) loop closures."""
    graph = NonlinearFactorGraph()
    initial = Values()

    prior_noise = Diagonal.Sigmas(PRIOR_SIGMAS)
    odom_noise = Diagonal.Sigmas(ODOM_SIGMAS)
    loop_noise = Diagonal.Sigmas(LOOP_SIGMAS)

    graph.add(PriorFactorPose2(0, Pose2(xN[0], yN[0], tN[0]), prior_noise))
    initial.insert(0, Pose2(xN[0], yN[0], tN[0]))

    for i in range(1, len(xN)):
        rel = relative_pose2(xN[i - 1], yN[i - 1], tN[i - 1], xN[i], yN[i], tN[i])
        graph.add(BetweenFactorPose2(i - 1, i, rel, odom_noise))
        initial.insert(i, Pose2(xN[i], yN[i], tN[i]))

    # In a real system these come from place recognition + ICP.
    # Here we simulate them from ground truth.
    for a, b in loop_pairs(len(xN)):
        rel = relative_pose2(X_gt[a], Y_gt[a], T_gt[a], X_gt[b], Y_gt[b], T_gt[b])
        graph.add(BetweenFactorPose2(a, b, rel, loop_noise))

    return graph, initial


def optimize_graph(graph, initial, verbose=False):
    """Levenberg-Marquardt optimization."""
    params = LevenbergMarquardtParams()
    params.setVerbosity("TERMINATION" if verbose else "SILENT")
    return LevenbergMarquardtOptimizer(graph, initial, params).optimize()


def extract_poses(values):
    """Values -> (X, Y, THETA) arrays ordered by key."""
    keys = sorted(values.keys())
    X = np.array([values.atPose2(k).x() for k in keys])
    Y = np.array([values.atPose2(k).y() for k in keys])
    T = np.array([values.atPose2(k).theta() for k in keys])
    return X, Y, T


# --- Plotting (one helper, thin wrappers keep old notebooks working) ---


def _plot_one(ax, X, Y, THETA, marker, color, label, arrow_scale=0.25):
    ax.plot(X, Y, marker, label=label, markersize=4)
    for x, y, t in zip(X, Y, THETA):
        ax.plot(
            [x, x + arrow_scale * math.cos(t)],
            [y, y + arrow_scale * math.sin(t)],
            color + "->",
            linewidth=0.8,
        )


def plot_trajectories(trajs, title="", save=None):
    """Plot one or more (X, Y, THETA, label, color) trajectories."""
    _, ax = plt.subplots()
    for X, Y, T, label, color in trajs:
        _plot_one(ax, X, Y, T, color + "o", color, label)
    ax.set_aspect("equal")
    ax.legend()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150)
    else:
        plt.show()
    plt.close()


def draw(X, Y, THETA):
    plot_trajectories([(X, Y, THETA, "Trajectory", "m")])


def drawTwo(X1, Y1, T1, X2, Y2, T2):
    plot_trajectories(
        [
            (X1, Y1, T1, "Ground truth", "r"),
            (X2, Y2, T2, "Optimized", "b"),
        ]
    )


def drawThree(X1, Y1, T1, X2, Y2, T2, X3, Y3, T3):
    plot_trajectories(
        [
            (X1, Y1, T1, "Ground truth", "r"),
            (X2, Y2, T2, "Optimized", "b"),
            (X3, Y3, T3, "Noisy", "g"),
        ]
    )


if __name__ == "__main__":
    X, Y, T = genTraj()
    xN, yN, tN = addNoise(X, Y, T, seed=4)

    graph, initial = build_pose2_graph(xN, yN, tN, X, Y, T)
    print(f"Factors: {graph.size()}, initial error: {graph.error(initial):.1f}")
    result = optimize_graph(graph, initial)
    print(f"Final error: {graph.error(result):.1f}")

    xOpt, yOpt, tOpt = extract_poses(result)
    drawThree(X, Y, T, xOpt, yOpt, tOpt, xN, yN, tN)
