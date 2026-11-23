# Validation script for the 6-step cloth fold-towel task (harness program, seed-42 oracle layout).
# Other seeds: tools/bright/cloth_seed_run.sh <seed> <this file> (layout from a 3 s episode reset trace).
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
NEAR = np.array([[0.3207, 0.1374], [0.3143, -0.1626]])
FAR = np.array([[0.6207, 0.1310], [0.6143, -0.1690]])
BASKET = np.array([0.6362, -0.3544])                    # basket centre (base xy), quat [0.0, 0.0, 0.70711, 0.70711]
LIFT = 0.10                     # arc height of the near corners
PITCH_GRASP, PITCH_END = 80, 30   # gripper pitch at the grasp / at the far edge (flatter = more reach)
END_TIP_Z = 0.035               # fingertip height when the top layer is let go
OVERSHOOT = 0.01                # land beyond the far corners
FOLD_SLIP = 0.077                # measured drag of the towel during the fold (seed 42, oracle trace)
CREASE_IN = 0.015                # pinch this far inside the fold line
CREASE_DY = -0.01               # pinch slightly right of the fold-line middle (right-arm reach)
CARRY_Z = 0.32                  # fingertip height while carrying (towel hangs ~20 cm)
TOUCH_Z = 0.22                  # fingertip height where the hanging towel reaches the basket floor
PLACE_TIP_Z = 0.035             # fingertip height at release (towel on the floor below it)
PLACE_PITCH = 60                # steeper wrist inside the basket (clears the near wall)
LAY_DX = 0.0                    # horizontal travel while laying the towel down


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

    robot.begin_step("pick_crease")
    robot.go_home(arms=("L", "R"), grip=0.4)
    fold_dir = FAR.mean(axis=0) - NEAR.mean(axis=0)
    fold_dir /= np.linalg.norm(fold_dir)
    # nominal fold line = mid-line of the flat towel; the fold drags the whole towel FOLD_SLIP further away
    # (seed 42 trace: fold-line middle at base x 0.547 instead of 0.470)
    centre = np.vstack([NEAR, FAR]).mean(axis=0)
    crease = centre + (FOLD_SLIP + CREASE_IN) * fold_dir + np.array([0.0, CREASE_DY])
    robot.log("crease target", crease=crease.tolist())
    PR = 45
    right([*crease, CLOTH_Z + 0.07], 0.45, PR, "approach fold line", speed=0.2)
    right([*crease, CLOTH_Z + 0.008], 0.45, PR, "lower onto the fold line", speed=0.08)
    right([*crease, CLOTH_Z + 0.008], 0.0, PR, "pinch the fold line middle")
    robot.wait(1, label="hold")
    right([*crease, CARRY_Z], 0.0, PR, "lift folded towel", speed=0.10)
    robot.end_step(True)

    robot.begin_step("over_basket")
    right([*BASKET, CARRY_Z], 0.0, PR, "carry over basket", speed=0.10)
    robot.wait(1, label="let the towel stop swinging")
    robot.end_step(True)

    robot.begin_step("place_folded")
    # lower until the hanging towel touches the floor, then lay it down while moving along LAY_DX
    right([*BASKET, TOUCH_Z], 0.0, PLACE_PITCH, "lower into basket", speed=0.06)
    right([BASKET[0] + LAY_DX, BASKET[1], PLACE_TIP_Z], 0.0, PLACE_PITCH, "lay the towel down", speed=0.04)
    robot.wait(1, label="settle before release")
    right([BASKET[0] + LAY_DX, BASKET[1], PLACE_TIP_Z], 0.65, PLACE_PITCH, "release on the basket floor")
    robot.wait(1, label="let go")
    right([BASKET[0] + LAY_DX, BASKET[1], 0.28], 0.65, PLACE_PITCH, "lift clear of the towel", speed=0.08)
    right([BASKET[0] - 0.10, BASKET[1] + 0.10, 0.28], 0.65, PR, "clear basket", speed=0.15)
    robot.wait(3, label="settle in basket")
    robot.end_step(True)
    while robot.time_left() > 2.0:
        robot.wait(1, label="idle")
