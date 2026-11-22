#!/usr/bin/env python
"""Does the toolbox's dowel handle block a top-down placement? (no simulation)

The right G-gripper holds a tool lying flat along the box length (x), fingers closing across the tool
(opening along box y), approach straight down.  For several release heights and lateral offsets of
the tool centre from the box mid-line, report the minimum distance between (a) the gripper meshes and
(b) the tool's box hull and the toolbox geometry (bottom, side walls, tall end panels, dowel).

    python tools/bright/toolbox_handle_clearance.py --asset-root x2real-assets
"""
import argparse
import json
import math

import numpy as np
from pxr import Usd, UsdGeom

ap = argparse.ArgumentParser()
ap.add_argument("--asset-root", default="x2real-assets")
ap.add_argument("--out", default="")
args = ap.parse_args()
st = Usd.Stage.Open(f"{args.asset_root}/simready/robots/ex001/v00.05.05/ex001_6r.usd")


def link_points(name):
    prim = st.GetPrimAtPath(f"/World/right_arm_gripper_{name}_link")
    out = []
    for c in Usd.PrimRange(prim):
        if c.IsA(UsdGeom.Mesh) and "visuals" in str(c.GetPath()):
            m = np.array(UsdGeom.Xformable(c).ComputeLocalToWorldTransform(0))
            p = np.array(UsdGeom.Mesh(c).GetPointsAttr().Get(), dtype=float)
            out.append((np.c_[p, np.ones(len(p))] @ m)[:, :3])
    return np.vstack(out)


o = np.array(UsdGeom.Xformable(st.GetPrimAtPath("/World/right_arm_gripper_base_link"))
             .ComputeLocalToWorldTransform(0).ExtractTranslation())
PB, PL, PR = link_points("base") - o, link_points("left") - o, link_points("right") - o
PIV = 0.077


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def gripper(open_deg):
    a = math.radians(open_deg)
    L = (PL - [PIV, 0.025, 0]) @ rotz(a).T + [PIV, 0.025, 0]
    R = (PR - [PIV, -0.025, 0]) @ rotz(-a).T + [PIV, -0.025, 0]
    return np.vstack([PB, L, R]), L, R


def tip_gap(deg):
    _, L, R = gripper(deg)
    return L[L[:, 0] > L[:, 0].max() - 0.025][:, 1].min() - R[R[:, 0] > R[:, 0].max() - 0.025][:, 1].max()


def open_for(gap):
    lo, hi = -4.18, 49.96
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tip_gap(mid) < gap else (lo, mid)
    return hi


TIP_X = max(PL[:, 0].max(), PR[:, 0].max())
# toolbox (same numbers as gen_bright_assets.gen_toolbox), frame: centre of the bottom, z up
L, W, H, t, tb, endH, dz, dr = 0.28, 0.20, 0.06, 0.008, 0.008, 0.16, 0.145, 0.009
BOXES = [((0, 0, tb / 2), (L, W, tb)), ((0, W / 2 - t / 2, H / 2), (L, t, H)), ((0, -W / 2 + t / 2, H / 2), (L, t, H)),
         ((L / 2 - t / 2, 0, endH / 2), (t, W, endH)), ((-L / 2 + t / 2, 0, endH / 2), (t, W, endH)),
         ((0, 0, dz), (L - 2 * t, 2 * dr, 2 * dr))]


def dist_to_boxes(P):
    d = np.inf
    for c, s in BOXES:
        q = np.abs(P - np.array(c)) - np.array(s) / 2
        dd = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)   # signed
        d = min(d, dd.min())
    return d


TOOLS = {  # grasp at the tool centre across its width; tool hull = (length x, width y, thickness z)
    "hammer": {"hull": (0.176, 0.070, 0.026), "grip_width": 0.016, "grip_top": 0.021},
    "wrench": {"hull": (0.204, 0.046, 0.014), "grip_width": 0.024, "grip_top": 0.014},
}
res = []
for name, tool in TOOLS.items():
    deg = open_for(tool["grip_width"] + 0.004)
    G, _, _ = gripper(deg)
    # gripper frame -> world: approach x -> -z, opening y -> +y, thickness z -> x
    Rm = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
    Gw0 = (G - [TIP_X - 0.010, 0, 0]) @ Rm.T
    for h in (0.0, 0.03, 0.06, 0.09, 0.12, 0.16):          # tool bottom above the box floor
        zb = tb + h
        ok = []
        for e in np.arange(0.0, 0.0905, 0.005):              # tool centre offset from the dowel line
            tip = np.array([0.0, e, zb + tool["grip_top"]])
            dg = dist_to_boxes(Gw0 + tip)
            lx, wy, tz = tool["hull"]
            hull = np.array([[sx * lx / 2, e + sy * wy / 2, zb + sz * tz] for sx in (-1, 1) for sy in (-1, 1)
                             for sz in (0, 1)])
            dt = dist_to_boxes(hull)
            if dg > 0.003 and dt > 0.002:
                ok.append(round(float(e), 3))
            res.append({"tool": name, "release_height_above_floor_m": h, "offset_m": round(float(e), 3),
                        "gripper_min_dist_m": round(float(dg), 4), "tool_min_dist_m": round(float(dt), 4)})
        print(f"{name:7s} release {h:4.2f} m above floor: collision-free offsets {ok[:1]}..{ok[-1:]} ({len(ok)} of 19)")
if args.out:
    open(args.out, "w").write(json.dumps(res, indent=1))
