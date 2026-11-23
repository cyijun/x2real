"""Seed 42 Arm-6A calibration trajectory for the physical grasp, not a competition policy.

Coordinates are from the cloth reset trace transformed to the robot base frame.
Independent arm/cloth measurements in MANA_CLOTH_TRACE check acquisition.
"""

import numpy as np
from runtime import geometry as G


def run(robot):
    near = np.array([[0.321, 0.137], [0.314, -0.163]])
    far = np.array([[0.621, 0.131], [0.614, -0.169]])

    def pair(xy, z, opening, phase, pitch=80):
        rpy = G.tilted(pitch)
        commands = [dict(tcp=[*point, z], rpy=rpy, grip=opening) for point in xy]
        return robot.move(L=commands[0], R=commands[1], speed=0.26, label=phase)

    # Insert the spread jaws just inside the corners. Close across the towel edges.
    grip_xy = near.copy()
    grip_xy[:, 0] += 0.012
    grip_xy[:, 1] += [-0.008, 0.008]
    pair(grip_xy, 0.065, 0.42, "approach near corners")
    pair(grip_xy, 0.001, 0.42, "lower fingers")
    pair(grip_xy, 0.001, 0, "pinch both corners")
    pair(grip_xy + [0.032, 0], 0.095, 0, "lift near edge")
    pair((near + far) / 2, 0.14, 0, "carry over midline")
    pair(far - [0.035, 0], 0.06, 0, "unroll toward far edge", 60)
    pair(far - [0.01, 0], 0.013, 0, "lay down", 55)
    pair(far - [0.01, 0], 0.018, 0.45, "release", 55)
    pair(far - [0.04, 0], 0.09, 0.45, "retract", 55)
    robot.wait(2)
    # Keep the left arm clear of the right-side basket. Pick up the two
    # overlapping layers near the middle of their shared edge.
    robot.go_home(arms=("L",), grip=0.4)
    right = G.tilted(62)

    def move_right(x, y, z, grip, label):
        robot.move_tcp("R", [x, y, z], rpy=right, grip=grip, speed=0.28, label=label)

    edge = far.mean(axis=0)
    move_right(edge[0] - 0.015, edge[1], 0.07, 0.45, "approach folded edge")
    move_right(edge[0] - 0.015, edge[1], 0.01, 0.45, "lower for regrasp")
    move_right(edge[0] - 0.015, edge[1], 0.01, 0, "regrasp folded towel")
    move_right(edge[0] - 0.06, edge[1], 0.20, 0, "lift folded towel")
    move_right(0.54, -0.33, 0.24, 0, "carry over basket")
    move_right(0.60, -0.34, 0.085, 0, "lower into basket")
    move_right(0.60, -0.34, 0.085, 0.65, "release into basket")
    move_right(0.52, -0.32, 0.22, 0.65, "clear basket")
    robot.wait(2)
