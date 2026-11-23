#!/usr/bin/env python
"""Geometric check for the stack_rings task: left G-gripper pressing the edge of the round
stacker base (final: 8 cm, 5 rings) while the right G-gripper puts a ring onto the peg from straight above.

No simulation: gripper meshes (visual, i.e. incl. finger tips) are read from the robot USD,
fingers are rotated to the required opening, both grippers are posed with the approach axis
pointing down (or tilted for the left one), and the minimum distance between the two point
clouds is reported for every ring / release height / left-gripper orientation.

    python tools/bright/stacker_gripper_clearance.py --asset-root x2real-assets
"""
from __future__ import annotations

import argparse
import itertools
import json
import math

import numpy as np
from pxr import Gf, Usd, UsdGeom
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--asset-root", default="x2real-assets")
ap.add_argument("--base-radius", type=float, default=0.040)
ap.add_argument("--base-height", type=float, default=0.025)
ap.add_argument("--peg-height", type=float, default=0.100, help="peg length above the base top")
ap.add_argument("--ring-id", type=float, default=0.023)
ap.add_argument("--ring-ods", type=float, nargs="+", default=[0.070, 0.063, 0.056, 0.048, 0.040])
ap.add_argument("--ring-t", type=float, default=0.017)
ap.add_argument("--press-inset", type=float, default=0.008, help="finger pad centre inside the base rim")
ap.add_argument("--out", default="")
args = ap.parse_args()

USD = f"{args.asset_root}/simready/robots/ex001/v00.05.05/ex001_6r.usd"
st = Usd.Stage.Open(USD)


def link_points(name):
    prim = st.GetPrimAtPath(f"/World/right_arm_gripper_{name}_link")
    pts = []
    for c in Usd.PrimRange(prim):
        if c.IsA(UsdGeom.Mesh) and "visuals" in str(c.GetPath()):
            xf = UsdGeom.Xformable(c).ComputeLocalToWorldTransform(0)
            p = np.array(UsdGeom.Mesh(c).GetPointsAttr().Get(), dtype=float)
            m = np.array(xf).T  # row-vector convention -> column
            ph = np.c_[p, np.ones(len(p))] @ m.T
            pts.append(ph[:, :3])
    return np.vstack(pts)


base_xf = UsdGeom.Xformable(st.GetPrimAtPath("/World/right_arm_gripper_base_link")).ComputeLocalToWorldTransform(0)
base_o = np.array(base_xf.ExtractTranslation())
P_base = link_points("base") - base_o          # gripper-base frame (rotation is identity in the rest pose)
P_left = link_points("left") - base_o
P_right = link_points("right") - base_o
PIV_L = np.array([0.077, 0.025, 0.0])
PIV_R = np.array([0.077, -0.025, 0.0])
TIP_X = max(P_left[:, 0].max(), P_right[:, 0].max())


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def gripper(open_deg):
    """Point cloud of the whole gripper with each finger opened by open_deg (0 = rest)."""
    a = math.radians(open_deg)
    L = (P_left - PIV_L) @ rotz(a).T + PIV_L
    R = (P_right - PIV_R) @ rotz(-a).T + PIV_R
    return np.vstack([P_base, L, R]), L, R


def tip_gap(open_deg):
    _, L, R = gripper(open_deg)
    lt = L[L[:, 0] > L[:, 0].max() - 0.025]
    rt = R[R[:, 0] > R[:, 0].max() - 0.025]
    return lt[:, 1].min() - rt[:, 1].max()


def opening_for_gap(gap):
    lo, hi = -4.18, 49.96
    for _ in range(50):
        mid = (lo + hi) / 2
        if tip_gap(mid) < gap:
            lo = mid
        else:
            hi = mid
    return hi


def pose(points, approach, opening_dir, tip_world, open_deg):
    """Map gripper frame (x=approach, y=opening) to world; place the tip centre at tip_world."""
    x = np.asarray(approach, float); x /= np.linalg.norm(x)
    y = np.asarray(opening_dir, float); y -= x * (y @ x); y /= np.linalg.norm(y)
    z = np.cross(x, y)
    R = np.c_[x, y, z]
    # tip centre in the gripper frame: between the finger tips at the given opening
    _, L, Rr = gripper(open_deg)
    tip_local = np.array([TIP_X - 0.010, 0.0, 0.0])
    return (points - tip_local) @ R.T + np.asarray(tip_world)


results = []
closed_gap = tip_gap(0.0)
r_in = args.ring_id / 2
peg_top = args.base_height + args.peg_height
press_r = args.base_radius - args.press_inset
for k, od in enumerate(args.ring_ods):
    r_out = od / 2
    wall = r_out - r_in
    open_deg = opening_for_gap(wall + 0.006)
    r_mid = (r_in + r_out) / 2
    PR, _, _ = gripper(open_deg)
    heights = {"release_above_peg_tip": peg_top + 0.005 + args.ring_t / 2,
               "lowered_to_seat": args.base_height + args.ring_t * k + args.ring_t / 2}
    for (hname, zr), (lname, l_app, l_open) in itertools.product(heights.items(), [
            ("left_vertical_fingers_tangential", (0, 0, -1), (1, 0, 0)),
            ("left_vertical_fingers_radial", (0, 0, -1), (0, 1, 0)),
            ("left_tilted30_from_outside", (0, -0.5, -math.sqrt(0.75)), (1, 0, 0)),
            ("left_tilted45_from_outside", (0, -math.sqrt(0.5), -math.sqrt(0.5)), (1, 0, 0)),
            ("left_tilted60_from_outside", (0, -math.sqrt(0.75), -0.5), (1, 0, 0))]):
        PL, _, _ = gripper(0.0)
        # world frame: peg axis at the origin, table top z=0, robot-left is +y
        left = pose(PL, l_app, l_open, (0.0, press_r, args.base_height + 0.002), 0.0)
        right = pose(PR, (0, 0, -1), (0, 1, 0), (0.0, -r_mid, zr), open_deg)
        d, _ = cKDTree(left).query(right, k=1)
        # if they overlap, how much further apart (along y) would the right gripper have to be?
        extra = 0.0
        tree = cKDTree(left)
        while tree.query(right + np.array([0.0, -extra, 0.0]), k=1)[0].min() < 0.003 and extra < 0.3:
            extra += 0.002
        results.append({"ring": f"ring_{k + 1}", "ring_od_m": od, "right_open_deg": round(open_deg, 1),
                        "right_tip_gap_m": round(wall + 0.006, 4), "release": hname, "left": lname,
                        "axis_distance_m": round(press_r + r_mid, 4),
                        "min_distance_m": round(float(d.min()), 4),
                        "extra_separation_needed_m": round(extra, 3)})
summary = {"gripper_length_m": round(float(TIP_X), 4),
           "housing_width_m": round(float(np.ptp(P_base[:, 1])), 4),
           "housing_thickness_m": round(float(np.ptp(P_base[:, 2])), 4),
           "closed_tip_gap_m": round(float(closed_gap), 4),
           "max_tip_gap_m": round(float(tip_gap(49.96)), 4),
           "stacker": vars(args), "cases": results}
print(json.dumps(summary, indent=1))
if args.out:
    open(args.out, "w").write(json.dumps(summary, indent=1))
