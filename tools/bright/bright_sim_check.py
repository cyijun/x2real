#!/usr/bin/env python
"""Minimal Isaac Sim checks for BRIGHT props (no ManaEnv pipeline).

Modes
  layout : put every prop on the green-booth table and render head-camera +
           overview images (static, no physics).
  drop   : physics sanity / geometric feasibility tests (CPU PhysX):
           rings dropped onto the stacker peg, tools dropped into the toolbox
           slots, flower stems dropped into the vase, beans poured from a cup
           into a bowl, cup/bowl resting stability.  Writes JSON + frames.

    python tools/bright/bright_sim_check.py --mode layout --out /tmp/bright_check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["layout", "drop"], default="layout")
ap.add_argument("--out", default="/tmp/bright_check")
ap.add_argument("--asset-root", default=os.environ.get("MANAENV_ASSET_ROOT", "x2real-assets"))
ap.add_argument("--frames", type=int, default=0, help="save every Nth physics frame as png (drop mode)")
ap.add_argument("--video", action="store_true")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1280, "height": 720,
                     "extra_args": ["--/UJITSO/textures=false"]})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

ROOT = Path(args.asset_root).resolve()
BRIGHT = ROOT / "bright"
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
TABLE = np.array([0.205, -0.273, -0.2404])   # green booth tabletop centre (world)

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

# background (same pose as the scene YAMLs)
bg = UsdGeom.Xform.Define(stage, "/World/booth")
bg.GetPrim().GetReferences().AddReference(str(ROOT / "art_assets/scenes/green_booth/green_booth_mesh.usd"))
xf = UsdGeom.XformCommonAPI(bg)
bg.ClearXformOpOrder()
bg.AddTranslateOp().Set(Gf.Vec3d(-0.452, -0.271, -0.985))
bg.AddOrientOp().Set(Gf.Quatf(0.709, 0.0, 0.0, -0.704))

ps = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
ps.CreateGravityMagnitudeAttr(9.81)
px = PhysxSchema.PhysxSceneAPI.Apply(ps.GetPrim())
px.CreateEnableGPUDynamicsAttr(False)
px.CreateBroadphaseTypeAttr("MBP")
px.CreateEnableCCDAttr(True)
px.CreateTimeStepsPerSecondAttr(120)
px.CreateSolverTypeAttr("TGS")


def find(asset_id: str) -> str:
    hits = list(BRIGHT.rglob(f"{asset_id}.usd"))
    if not hits:
        raise FileNotFoundError(asset_id)
    return str(hits[0])


def place(name: str, asset_id: str, pos, yaw_deg: float = 0.0, quat=None, physics=True):
    prim_path = f"/World/props/{name}"
    x = UsdGeom.Xform.Define(stage, prim_path)
    x.GetPrim().GetReferences().AddReference(find(asset_id))
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    if quat is None:
        h = math.radians(yaw_deg) / 2
        quat = (math.cos(h), 0.0, 0.0, math.sin(h))
    x.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in quat]))
    if not physics:
        x.GetPrim().GetAttribute("physics:rigidBodyEnabled").Set(False)
    return prim_path


def tab(dx, dy, dz=0.0):
    """Table-relative position: dx toward the far side (+world x), dy toward robot right (-world y)."""
    return TABLE + np.array([dx, -dy, dz])


def camera(name, pos, quat_wxyz=None, look_at=None, focal=2.97, haperture=5.54, vaperture=3.09):
    cam = UsdGeom.Camera.Define(stage, f"/World/{name}")
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(haperture)
    cam.CreateVerticalApertureAttr(vaperture)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100))
    cam.ClearXformOpOrder()
    cam.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if look_at is not None:
        m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos), Gf.Vec3d(*look_at), Gf.Vec3d(0, 0, 1)).GetInverse()
        q = m.ExtractRotationQuat()
        cam.AddOrientOp().Set(Gf.Quatf(q.GetReal(), *q.GetImaginary()))
    else:
        cam.AddOrientOp().Set(Gf.Quatf(*quat_wxyz))
    return cam.GetPath().pathString


# Head camera: robot base (-0.32165, -0.25918, -0.23) + offset (0.084, 0, 0.517), OpenGL rot (w,x,y,z)
HEAD = camera("head_cam", (-0.32165 + 0.084, -0.25918, -0.23 + 0.517), look_at=(0.22, -0.26, -0.24))  # approx. robot head camera
OVER = camera("over_cam", tuple(TABLE + np.array([0.75, 0.0, 0.75])), look_at=tuple(TABLE), focal=18.0,
              haperture=20.955, vaperture=11.79)
TOP = camera("top_cam", tuple(TABLE + np.array([0.0, 0.0, 1.3])), look_at=tuple(TABLE + np.array([0.001, 0, 0])),
             focal=18.0, haperture=20.955, vaperture=11.79)


def renderer(cam_paths):
    out = {}
    for cp in cam_paths:
        rp = rep.create.render_product(cp, (1280, 720))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        out[cp] = ann
    return out


def save(anns, tag):
    from PIL import Image
    for cp, ann in anns.items():
        d = ann.get_data()
        if d is None or getattr(d, "size", 0) == 0:
            continue
        Image.fromarray(np.asarray(d)[..., :3]).save(OUT / f"{tag}_{cp.split('/')[-1]}.png")


def warm(n=30):
    for _ in range(n):
        app.update()


if args.mode == "layout":
    items = [
        # (name, asset, dx, dy, yaw)
        ("tablecloth", "bright_tablecloth_gray_grid", 0.0, 0.0, 0),
        ("bowl_blue", "bright_bowl_blue", 0.05, -0.20, 0), ("bowl_pink", "bright_bowl_pink", -0.15, -0.30, 0),
        ("cup_green", "bright_cup_green", -0.12, -0.08, 0), ("cup_purple", "bright_cup_purple", 0.15, -0.05, 0),
        ("mcup", "bright_measuring_cup_blue", 0.22, -0.18, 0),
        ("stacker", "bright_stacker_round", 0.05, 0.05, 0),
        ("ring_l", "bright_ring_1", -0.12, 0.10, 0), ("ring_m", "bright_ring_3", -0.12, 0.20, 0),
        ("ring_s", "bright_ring_5", -0.05, 0.25, 0),
        ("vase", "bright_vase_white", 0.20, 0.25, 0),
        ("rose", "bright_flower_rose_red", -0.25, 0.30, 90), ("rose2", "bright_flower_rose_pink", -0.28, 0.15, 90),
        ("toolbox", "bright_toolbox", 0.25, 0.42, 90), ("hammer", "bright_hammer", -0.25, -0.15, 0),
        ("screw_twister", "bright_screw_twister", -0.28, -0.30, 0), ("screwdriver", "bright_screwdriver", -0.28, -0.42, 0),
        ("basket", "bright_basket_green", 0.18, -0.42, 90), ("towel", "bright_towel_blue", -0.05, -0.45, 0),
    ]
    for name, aid, dx, dy, yaw in items:
        place(name, aid, tab(dx, dy, 0.002 if name != "tablecloth" else 0.0), yaw, physics=False)
    anns = renderer([HEAD, OVER, TOP])
    warm(60)
    rep.orchestrator.step(rt_subframes=8)
    save(anns, "layout")
    # tablecloth variants top view
    for cloth in ("forest_deer", "blue_cloud", "wood"):
        p = stage.GetPrimAtPath("/World/props/tablecloth")
        p.GetReferences().ClearReferences()
        p.GetReferences().AddReference(find(f"bright_tablecloth_{cloth}"))
        warm(20)
        rep.orchestrator.step(rt_subframes=8)
        save({HEAD: anns[HEAD]}, f"cloth_{cloth}")
    print("LAYOUT_DONE", flush=True)
else:
    from omni.physx import get_physx_interface

    # ---- scene for physics tests (dynamic, no robot): final props (PDF 2026-11-22) ----
    place("tablecloth", "bright_tablecloth_blue_cloud", tab(0, 0, 0.0), 0)
    z0 = 0.0015
    tests = {}
    # stacker (8 cm base, peg top 12.5 cm): 5 rings released level ~5 mm above the peg top, 3.6 mm xy error,
    # largest first -- what a careful top-down placement would do
    base_pos = tab(0.05, 0.0, z0)
    place("stacker", "bright_stacker_round", base_pos, 0)
    peg_world = base_pos.copy()   # single central peg
    ring_ids = [f"bright_ring_{k}" for k in range(1, 6)]
    PEG_TOP = 0.125
    # open toolbox (yaw 0: long side along world x); tools released lying along the box, ~3 cm above the side walls
    tb_pos = tab(0.15, 0.30, z0)
    place("toolbox", "bright_toolbox", tb_pos, 0)
    TOOL_DROPS = [("hammer", "bright_hammer", np.array([0.0, -0.022, 0.105])),
                  ("screwdriver", "bright_screwdriver", np.array([0.0, 0.035, 0.105])),
                  ("screw_twister", "bright_screw_twister", np.array([0.0, 0.0, 0.125]))]
    # vase + five roses released stem-down above the mouth (a few mm to 1.5 cm off-centre)
    vase_pos = tab(-0.15, 0.25, z0)
    place("vase", "bright_vase_white", vase_pos, 0)
    up = (math.cos(math.pi / 4), 0.0, -math.sin(math.pi / 4), 0.0)    # -90 deg about y: +X -> +Z (head up)
    ROSE_IDS = ["purple", "blue", "yellow", "red", "pink"]
    ROSE_OFF = [(0.006, 0.0), (-0.012, 0.008), (0.004, -0.014), (0.010, 0.012), (-0.008, -0.006)]

    # beans poured: bowl on table, cup above the bowl tilted 125 deg -> cup kinematically tipped by script
    bowl_pos = tab(-0.05, -0.20, z0)
    place("bowl", "bright_bowl_blue", bowl_pos, 0)
    cup_pos = tab(-0.20, -0.35, z0)
    cup_path = place("cup", "bright_cup_green", cup_pos, 0)
    nb = 20
    for k in range(nb):
        a = 2 * math.pi * k / 7
        r = 0.012 if k % 2 else 0.022
        place(f"bean_{k}", "bright_bean_red", cup_pos + np.array([r * math.cos(a), r * math.sin(a), 0.12 + 0.008 * k]),
              yaw_deg=37 * k)
    # basket + towel (rigid) dropped in
    bk_pos = tab(0.20, -0.30, z0)
    place("basket", "bright_basket_green", bk_pos, 90)
    place("towel", "bright_towel_pink", bk_pos + np.array([0, 0, 0.20]), 20)

    # step -> (name, asset, position, yaw, quat): objects added while the simulation runs
    SPAWN = {}
    for k, rid in enumerate(ring_ids):
        SPAWN[5 + 110 * k] = (f"ring_{k}", rid, peg_world + np.array([0.003, -0.002, z0 + PEG_TOP + 0.005]), 30.0 * k, None)
    for k, (t, aid, off) in enumerate(TOOL_DROPS):
        SPAWN[8 + 130 * k] = (t, aid, tb_pos + off, 0.0, None)
    ROSE_TILT = math.radians(20)   # 5 heads of 9 cm around the axis need ~20 deg outward lean to not overlap
    ROSE_PATH = {}
    for k, c in enumerate(ROSE_IDS):
        # each rose goes in along its own axis: stem bottom 12 mm across the vase axis, head leaning 20 deg
        # outward, 5 directions 72 deg apart; lowered kinematically along that axis and released 4 cm above the floor
        a = 2 * math.pi * k / 5
        d = np.array([math.cos(a), math.sin(a), 0.0])
        u = np.array([math.sin(ROSE_TILT) * d[0], math.sin(ROSE_TILT) * d[1], math.cos(ROSE_TILT)])
        end = vase_pos + np.array([-0.012 * d[0], -0.012 * d[1], 0.04])
        start = end + 0.32 * u
        ax = np.cross([1.0, 0.0, 0.0], u)
        ang = math.acos(float(u[0]))
        ax = ax / np.linalg.norm(ax)
        q = (math.cos(ang / 2), *(math.sin(ang / 2) * ax))
        ROSE_PATH[f"rose_{k}"] = (start, end)
        SPAWN[11 + 130 * k] = (f"rose_{k}", f"bright_flower_rose_{c}", start, 0.0, q)
    stage.GetPrimAtPath(cup_path).GetAttribute("physics:kinematicEnabled").Set(True)   # scripted cup
    anns = renderer([HEAD, OVER]) if args.frames or args.video else renderer([OVER])
    warm(10)
    tl = omni.timeline.get_timeline_interface()
    tl.play()
    phx = get_physx_interface()

    def pose(name):
        t = phx.get_rigidbody_transformation(f"/World/props/{name}")
        return np.array(t["position"]), np.array(t["rotation"])  # rotation xyzw

    total = 900  # app updates (1 physics step of 1/120 s per update)
    INSERT = {}   # rose prim path -> (spawn step, xy+z start position): scripted kinematic insertion like a gripper would
    for step in range(total):
        if step in SPAWN:
            _n, _a, _p, _y, _q = SPAWN[step]
            _path = place(_n, _a, _p, _y, quat=_q)
            if _n.startswith("rose_"):
                stage.GetPrimAtPath(_path).GetAttribute("physics:kinematicEnabled").Set(True)
                INSERT[_path] = (step,) + ROSE_PATH[_n]
        for _path, (s0, p0, p1) in list(INSERT.items()):
            k = step - s0
            if k <= 80:   # move the stem bottom along the rose axis to 4 cm above the vase floor in 80 steps (0.67 s)
                p = p0 + (p1 - p0) * (k / 80.0)
                UsdGeom.Xformable(stage.GetPrimAtPath(_path)).GetOrderedXformOps()[0].Set(Gf.Vec3d(*p))
            elif k == 90:  # hold 10 steps, then let go
                stage.GetPrimAtPath(_path).GetAttribute("physics:kinematicEnabled").Set(False)
                del INSERT[_path]
        if 300 <= step < 540:
            s = min(1.0, (step - 300) / 120.0)
            tilt = 0.0 if step < 420 else min(1.0, (step - 420) / 80.0) * math.radians(125)
            p = cup_pos * (1 - s) + (bowl_pos + np.array([0.0, 0.07, 0.16])) * s   # side offset toward +y
            xfc = UsdGeom.Xformable(stage.GetPrimAtPath(cup_path))
            ops = xfc.GetOrderedXformOps()
            ops[0].Set(Gf.Vec3d(*p))
            h = tilt / 2  # tilt about world x (cup mouth toward -y, i.e. over the bowl)
            ops[1].Set(Gf.Quatf(math.cos(h), math.sin(h), 0.0, 0.0))
        app.update()
        if args.frames and step % args.frames == 0:
            rep.orchestrator.step(rt_subframes=2, pause_timeline=False)
            save(anns, f"f{step:04d}")
    rep.orchestrator.step(rt_subframes=8, pause_timeline=False)
    save(anns, "drop_final")

    res = {}
    peg_xy = peg_world[:2]
    for k in range(len(ring_ids)):
        p, q = pose(f"ring_{k}")
        x, y, z, w = q
        res[f"ring_{k + 1}"] = {"xy_off_from_peg": round(float(np.linalg.norm(p[:2] - peg_xy)), 4),
                                "z_above_table": round(float(p[2] - (TABLE[2] + z0)), 4),
                                "expected_z": round(0.025 + 0.017 * k, 4),
                                "tilt_dot_up": round(float(1 - 2 * (x * x + y * y)), 4)}
    for t, _, _ in TOOL_DROPS:
        p, q = pose(t)
        rel = p - tb_pos
        res[t] = {"rel_toolbox": rel.round(4).tolist(),
                  "inside": bool(abs(rel[0]) < 0.0915 and abs(rel[1]) < 0.0555 and -0.005 < rel[2] < 0.06)}
    for k, c in enumerate(ROSE_IDS):
        p, q = pose(f"rose_{k}")
        x, y, z, w = q
        axis_x_up = 2 * (x * z - w * y)
        rel = p - vase_pos
        res[f"rose_{c}"] = {"stem_bottom_rel_vase": rel.round(4).tolist(), "stem_axis_dot_up": round(float(axis_x_up), 4),
                            "in_vase": bool(np.linalg.norm(rel[:2]) < 0.06 and rel[2] < 0.10)}
    inbowl = 0
    for k in range(nb):
        p, _ = pose(f"bean_{k}")
        d = p - bowl_pos
        if np.linalg.norm(d[:2]) < 0.095 and -0.01 < d[2] < 0.10:
            inbowl += 1
    res["beans_in_bowl"] = inbowl
    res["beans_total"] = nb
    p, _ = pose("bowl")
    res["bowl_moved"] = float(np.linalg.norm(p[:2] - bowl_pos[:2]))
    p, _ = pose("towel")
    res["towel_rel_basket"] = (p - bk_pos).round(4).tolist()
    p, _ = pose("stacker")
    res["stacker_moved"] = float(np.linalg.norm(p[:2] - base_pos[:2]))
    p, _ = pose("vase")
    res["vase_moved"] = float(np.linalg.norm(p[:2] - vase_pos[:2]))
    (OUT / "drop_results.json").write_text(json.dumps(res, indent=1))
    print("DROP_RESULTS", json.dumps(res), flush=True)
    tl.stop()

app.close()
