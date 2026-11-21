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
        ("stacker", "bright_stacker_base", 0.05, 0.05, 0),
        ("ring_l", "bright_ring_large", -0.12, 0.10, 0), ("ring_m", "bright_ring_medium", -0.12, 0.20, 0),
        ("ring_s", "bright_ring_small", -0.05, 0.25, 0),
        ("vase", "bright_vase_white", 0.20, 0.25, 0),
        ("rose", "bright_flower_rose_red", -0.25, 0.30, 90), ("rose2", "bright_flower_rose_pink", -0.28, 0.15, 90),
        ("toolbox", "bright_toolbox", 0.25, 0.42, 90), ("hammer", "bright_hammer", -0.25, -0.15, 0),
        ("wrench", "bright_wrench", -0.28, -0.30, 0), ("screwdriver", "bright_screwdriver", -0.28, -0.42, 0),
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

    # ---- scene for physics tests (dynamic) ----
    place("tablecloth", "bright_tablecloth_blue_cloud", tab(0, 0, 0.0), 0)
    z0 = 0.0015
    tests = {}
    # stacker: base on table, rings released above the +y end peg (base along table y -> world -y)
    base_pos = tab(0.05, 0.0, z0)
    place("stacker", "bright_stacker_base", base_pos, 0)
    peg_world = base_pos + np.array([0.0, 0.10, 0.0])   # local +y peg (asset local y == world y here)
    ring_ids = ["bright_ring_large", "bright_ring_medium", "bright_ring_small"]
    for k, rid in enumerate(ring_ids):
        place(f"ring_{k}", rid, peg_world + np.array([0.003, -0.002, 0.14 + 0.05 * k]), 17 * k)
    # toolbox with hammer + wrench dropped into their slots (toolbox yaw 0: slots along x, hammer slot -y)
    tb_pos = tab(0.15, 0.30, z0)
    place("toolbox", "bright_toolbox", tb_pos, 0)
    place("hammer", "bright_hammer", tb_pos + np.array([0.0, -0.048, 0.09]), 0)
    place("wrench", "bright_wrench", tb_pos + np.array([0.0, 0.048, 0.09]), 0)
    # vase + two roses held vertical (stem down) above the mouth
    vase_pos = tab(-0.15, 0.25, z0)
    place("vase", "bright_vase_white", vase_pos, 0)
    down = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)  # +X -> -Z  (rotate +90deg about y)
    place("rose_0", "bright_flower_rose_red", vase_pos + np.array([0.008, 0.0, 0.40]), quat=down)
    place("rose_1", "bright_flower_rose_pink", vase_pos + np.array([-0.008, 0.006, 0.70]), quat=down)
    # beans poured: bowl on table, cup above the bowl tilted 120 deg containing beans -> use cup resting upright,
    # beans dropped into cup first, then cup is kinematically tipped by script.
    bowl_pos = tab(-0.05, -0.20, z0)
    place("bowl", "bright_bowl_blue", bowl_pos, 0)
    cup_pos = tab(-0.20, -0.35, z0)
    cup_path = place("cup", "bright_cup_green", cup_pos, 0)
    nb = 20
    for k in range(nb):
        a = 2 * math.pi * k / 7
        r = 0.012 if k % 2 else 0.022
        place(f"bean_{k}", "bright_bean_red", cup_pos + np.array([r * math.cos(a), r * math.sin(a), 0.11 + 0.008 * k]),
              yaw_deg=37 * k)
    # basket + towel (rigid) dropped in
    bk_pos = tab(0.20, -0.30, z0)
    place("basket", "bright_basket_blue", bk_pos, 90)
    place("towel", "bright_towel_pink", bk_pos + np.array([0, 0, 0.20]), 20)

    anns = renderer([HEAD, OVER]) if args.frames or args.video else renderer([OVER])
    warm(10)
    tl = omni.timeline.get_timeline_interface()
    tl.play()
    phx = get_physx_interface()

    def pose(name):
        t = phx.get_rigidbody_transformation(f"/World/props/{name}")
        return np.array(t["position"]), np.array(t["rotation"])  # rotation xyzw

    frames = []
    total = 600  # app updates (~ 5 s at 120 Hz physics if 1 step per update)
    # make the cup kinematic later for the scripted pour
    for step in range(total):
        if step == 240:
            # scripted pour: switch the cup to kinematic and move it above the bowl, then tilt
            cp = stage.GetPrimAtPath(cup_path)
            cp.GetAttribute("physics:kinematicEnabled").Set(True)
        if 240 <= step < 480:
            s = min(1.0, (step - 240) / 120.0)
            tilt = 0.0 if step < 360 else min(1.0, (step - 360) / 80.0) * math.radians(125)
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
    for k in range(3):
        p, q = pose(f"ring_{k}")
        res[f"ring_{k}"] = {"pos": p.round(4).tolist(), "xy_off_from_peg": float(np.linalg.norm(p[:2] - peg_xy)),
                            "z_above_table": float(p[2] - TABLE[2])}
    for t in ("hammer", "wrench"):
        p, q = pose(t)
        rel = p - tb_pos
        res[t] = {"rel_toolbox": rel.round(4).tolist()}
    for k in range(2):
        p, q = pose(f"rose_{k}")
        x, y, z, w = q
        axis_x_up = 2 * (x * z - w * y)
        res[f"rose_{k}"] = {"rel_vase": (p - vase_pos).round(4).tolist(), "stem_axis_dot_up": float(axis_x_up)}
    inbowl = 0
    for k in range(nb):
        p, _ = pose(f"bean_{k}")
        d = p - bowl_pos
        if np.linalg.norm(d[:2]) < 0.085 and -0.01 < d[2] < 0.09:
            inbowl += 1
    res["beans_in_bowl"] = inbowl
    res["beans_total"] = nb
    p, _ = pose("bowl")
    res["bowl_moved"] = float(np.linalg.norm(p[:2] - bowl_pos[:2]))
    p, _ = pose("towel")
    res["towel_rel_basket"] = (p - bk_pos).round(4).tolist()
    p, _ = pose("stacker")
    res["stacker_moved"] = float(np.linalg.norm(p[:2] - base_pos[:2]))
    (OUT / "drop_results.json").write_text(json.dumps(res, indent=1))
    print("DROP_RESULTS", json.dumps(res), flush=True)
    tl.stop()

app.close()
