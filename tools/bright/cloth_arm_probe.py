"""Scripted Arm-6A fold-towel trajectory for the cloth env (seed 42 layout); a validation script,
not a competition policy.

Corner coordinates are the seed-42 towel corners in the robot base frame (from the cloth reset
trace, MANA_CLOTH_TRACE).  The trajectory reproduces the fold that works for the bare PhysX cloth
(tools/bright/cloth_physx_fold_probe.py): both near corners travel on a half circle (peak 12 cm)
and land 1 cm *beyond* the far corners before the grippers open, which lays the top layer flat
instead of leaving a loop.  Then the right gripper picks the two layers at the middle of the
shared edge and drops the folded towel into the basket.

Targets are given for the fingertip midpoint (the point the cloth grasp attaches to), which sits
1.7 cm ahead of the harness TCP along the finger axis.
"""

import math

import numpy as np
from runtime import geometry as G

TIP = 0.017                     # fingertip midpoint ahead of the harness TCP [m]
CLOTH_Z = 0.014                 # towel mid-plane in the base frame (tablecloth top + rest offset)
NEAR = np.array([[0.321, 0.137], [0.314, -0.163]])    # [left, right] near corners (base xy)
FAR = np.array([[0.621, 0.131], [0.614, -0.169]])
BASKET = np.array([0.606, -0.348])                    # basket centre (base xy), rim at z ~0.096
LIFT = 0.10                     # arc height of the near corners
PITCH_GRASP, PITCH_END = 80, 30   # gripper pitch at the grasp / at the far edge (flatter = more reach)
END_TIP_Z = 0.035               # fingertip height when the top layer is let go
OVERSHOOT = 0.01                # land beyond the far corners


def tcp_for_tip(tip, pitch_deg):
    p = math.radians(pitch_deg)
    return np.asarray(tip, float) - TIP * np.array([math.cos(p), 0.0, -math.sin(p)])


def run(robot):
    def both(tips, grip, pitch, label, speed=0.15):
        cmds = [dict(tcp=tcp_for_tip(t, pitch).tolist(), rpy=G.tilted(pitch), grip=grip) for t in tips]
        return robot.move(L=cmds[0], R=cmds[1], speed=speed, label=label)

    def right(tip, grip, pitch, label, speed=0.15):
        return robot.move_tcp("R", tcp_for_tip(tip, pitch).tolist(), rpy=G.tilted(pitch), grip=grip,
                              speed=speed, label=label)

    g = NEAR + np.array([[0.012, -0.010], [0.012, 0.010]])     # just inside each corner
    robot.begin_step("grasp_corners")
    both([[*p, CLOTH_Z + 0.07] for p in g], 0.45, 80, "approach near corners", speed=0.2)
    both([[*p, CLOTH_Z + 0.004] for p in g], 0.45, 80, "lower to the towel", speed=0.08)
    both([[*p, CLOTH_Z + 0.004] for p in g], 0.0, 80, "pinch both corners")
    robot.wait(1, label="hold pinch")
    robot.end_step(True)

    robot.begin_step("fold")
    R = (FAR[:, 0].mean() - NEAR[:, 0].mean() + OVERSHOOT) / 2
    for deg in (30, 60, 90, 120, 150, 180):
        th = math.radians(deg)
        pitch = PITCH_GRASP + (PITCH_END - PITCH_GRASP) * deg / 180
        tips = [[p[0] + R * (1 - math.cos(th)), p[1], CLOTH_Z + 0.004 + LIFT * math.sin(th) + (END_TIP_Z - CLOTH_Z - 0.004) * deg / 180]
                for p in g]
        both(tips, 0.0, pitch, f"fold arc {deg}", speed=0.12)
    robot.wait(1, label="hold before release")
    end = [[p[0] + 2 * R, p[1]] for p in g]
    both([[*p, END_TIP_Z] for p in end], 0.5, PITCH_END, "release")
    robot.wait(1, label="let go")
    both([[p[0] - 0.03, p[1], 0.10] for p in end], 0.5, PITCH_END, "retract up", speed=0.08)
    robot.wait(3, label="settle folded towel")
    robot.end_step(True)

    robot.begin_step("pick_folded")
    robot.go_home(arms=("L",), grip=0.4)
    # shared (far) edge, 3 cm inside it and 3 cm right of the middle (easier reach for the right arm);
    # the fingertips sit on the top layer so both layers are within the capture radius
    edge = FAR.mean(axis=0) + np.array([-0.02, -0.03])
    PR = 45
    right([*edge, CLOTH_Z + 0.07], 0.45, PR, "approach folded edge", speed=0.2)
    right([*edge, CLOTH_Z + 0.008], 0.45, PR, "lower onto the folded edge", speed=0.08)
    right([*edge, CLOTH_Z + 0.008], 0.0, PR, "pinch both layers")
    robot.wait(1, label="hold")
    right([*edge, 0.28], 0.0, PR, "lift folded towel", speed=0.10)
    robot.end_step(True)

    robot.begin_step("into_basket")
    right([*BASKET, 0.28], 0.0, PR, "carry over basket", speed=0.12)
    right([*BASKET, 0.20], 0.0, PR, "lower into basket", speed=0.08)
    right([*BASKET, 0.20], 0.65, PR, "release into basket")
    robot.wait(1, label="drop")
    right([BASKET[0] - 0.10, BASKET[1] + 0.05, 0.28], 0.65, PR, "clear basket", speed=0.15)
    robot.wait(3, label="settle in basket")
    robot.end_step(True)
    while robot.time_left() > 2.0:
        robot.wait(1, label="idle")
