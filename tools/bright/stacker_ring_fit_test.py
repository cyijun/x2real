#!/usr/bin/env python
"""Ring-onto-peg fit test for the stack_rings props (no robot, CPU PhysX like the eval: TGS, CCD, 120 Hz).

Many stackers stand side by side on a static ground; one ring per stacker is released level (or tilted) a few
mm above the round knob with a horizontal offset from the peg axis, and the final state is classified:
  seated = ring down on the base (z within 2 mm, level, centred), stuck = around the peg but not down,
  off = slid off the knob / lying beside the peg.
Collision recipes are compared to find out why a 3 mm/side clearance jammed in the earlier test:
  legacy: 24 ring sectors, 16-gon peg, 12-gon knob, PhysX default contact/rest offset (the old asset recipe)
  fine:   48 sectors / 48-gon peg + knob, default offsets
  offset: legacy polygons, contactOffset 1 mm / restOffset 0
  final:  48 / 48 and contactOffset 1 mm / restOffset 0   (what gen_bright_assets.py now writes)
plus a reproduction of the old jam (old sizes: peg 2.2, hole 2.8, ring 2 cm thick) with the legacy and the final
recipe, three rings stacked one after another.

    python tools/bright/stacker_ring_fit_test.py --out /tmp/ringfit
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp/ringfit")
ap.add_argument("--offsets-mm", type=float, nargs="+", default=[0, 2, 4, 6, 8])
ap.add_argument("--tilts-deg", type=float, nargs="+", default=[0, 5])
ap.add_argument("--rings", type=int, nargs="+", default=[1, 3, 5])
ap.add_argument("--recipes", nargs="+", default=["legacy", "exact", "chamfer", "final"])
ap.add_argument("--steps", type=int, default=480)
ap.add_argument("--no-eval-solver", action="store_true",
                help="leave PhysX defaults instead of the eval settings (runtime/solver/physx_cpu_async.yaml: swept + "
                     "speculative CCD on every rigid body, >= 32 position / 4 velocity iterations)")
ap.add_argument("--ring-friction", type=float, default=None, help="override the ring static/dynamic friction")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "extra_args": ["--/UJITSO/textures=false"]})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx import get_physx_interface  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_bright_assets as gba  # noqa: E402

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
FINAL = dict(gba.STACK)
NONE = {"hole_chamfer": 0.0, "coll_hole_extra": None, "peg_friction": None, "max_depenetration": None,
        "position_iterations": None}
RECIPES = {
    # the old asset recipe: 24 sectors / 16-gon peg, PhysX default offsets, nothing else
    "legacy": {**NONE, "segments": 24, "peg_segments": 16, "contact_offset": None, "rest_offset": None},
    # exact visual geometry, fine polygons, contactOffset 4 mm / restOffset 0, depenetration 1 m/s, 32 iterations
    "exact": {**NONE, "segments": 48, "peg_segments": 48, "contact_offset": 0.004, "rest_offset": 0.0,
              "max_depenetration": 1.0, "position_iterations": 32},
    # + 1.5 mm lead-in and 0.25 mm looser collision bore
    "chamfer": {**NONE, "segments": 48, "peg_segments": 48, "contact_offset": 0.004, "rest_offset": 0.0,
                "max_depenetration": 1.0, "position_iterations": 32, "hole_chamfer": 0.0015, "coll_hole_extra": 0.00025},
    # + low-friction peg (0.3, min) = what gen_bright_assets.py now writes
    "final": {},
}
OLD_DIMS = {"base_r": 0.050, "base_h": 0.022, "peg_r": 0.011, "peg_h": 0.115, "ring_id": 0.028, "ring_t": 0.020}
OLD_ODS = [0.085, 0.070, 0.055]


def build(tag, dims, recipe, ods):
    root = OUT / "assets" / tag / "bright"
    gba.STACK.clear()
    gba.STACK.update({**FINAL, **dims, **recipe})
    if not recipe and dims:          # old sizes with the final recipe
        pass
    gba.gen_stacker(root)
    if args.ring_friction is not None:
        _pm = gba.physics_material
        gba.physics_material = lambda st, static, dynamic, **kw: _pm(st, static=args.ring_friction,
                                                                      dynamic=args.ring_friction, **kw)
    for k, od in enumerate(ods, start=1):
        gba.gen_ring(root, str(k), od / 2, gba.SAPELE, (0.45, 0.25, 0.18))
    if args.ring_friction is not None:
        gba.physics_material = _pm
    return root / "stacker", dict(gba.STACK)


ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
ps = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
ps.CreateGravityMagnitudeAttr(9.81)
px = PhysxSchema.PhysxSceneAPI.Apply(ps.GetPrim())
px.CreateEnableGPUDynamicsAttr(False)
px.CreateBroadphaseTypeAttr("MBP")
px.CreateEnableCCDAttr(True)
px.CreateTimeStepsPerSecondAttr(120)
px.CreateSolverTypeAttr("TGS")
EVAL_SOLVER = not args.no_eval_solver
if EVAL_SOLVER:
    px.CreateMinPositionIterationCountAttr(32)
    px.CreateMinVelocityIterationCountAttr(4)
ground = UsdGeom.Cube.Define(stage, "/World/ground")
ground.CreateSizeAttr(1.0)
ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
ground.AddScaleOp().Set(Gf.Vec3f(20.0, 20.0, 0.1))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())


def place(path, usd, pos, quat_wxyz=(1, 0, 0, 0)):
    x = UsdGeom.Xform.Define(stage, path)
    x.GetPrim().GetReferences().AddReference(str(usd))
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    x.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in quat_wxyz]))
    if EVAL_SOLVER:   # what manaenv's assembly factory authors on every non-kinematic rigid body
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(x.GetPrim())
        rb.CreateEnableCCDAttr(True)
        rb.CreateEnableSpeculativeCCDAttr(True)
    return path


cases, spawns = [], {}
col = 0
for recipe in args.recipes:
    sdir, S = build(recipe, {}, RECIPES[recipe], gba.RING_ODS)
    knob_top = S["base_h"] + S["peg_h"]
    for ring in args.rings:
        r_out = gba.RING_ODS[ring - 1] / 2
        for tilt in args.tilts_deg:
            for off in args.offsets_mm:
                cid = len(cases)
                base = np.array([0.15 * (cid % 12), 0.15 * (cid // 12) + 0.15 * col, 0.0])
                place(f"/World/c{cid}_base", sdir / "bright_stacker_round.usd", base)
                # tilted about +y by -tilt: the ring edge nearest the peg axis (-x side) is the low one
                h = -math.radians(tilt) / 2
                zr = knob_top + 0.005 + r_out * math.sin(math.radians(tilt))
                spawns.setdefault(5, []).append((f"/World/c{cid}_ring", sdir / f"bright_ring_{ring}.usd",
                                                 base + np.array([off / 1000.0, 0.0, zr]), (math.cos(h), 0, math.sin(h), 0)))
                cases.append({"id": cid, "recipe": recipe, "dims": "final", "ring": ring, "offset_mm": off,
                              "tilt_deg": tilt, "base": base, "expected_z": S["base_h"], "r_in": S["ring_id"] / 2,
                              "ring_path": f"/World/c{cid}_ring", "base_path": f"/World/c{cid}_base"})
    col += 1
# full stacks (5 rings one after another, 2 mm offset) with each recipe, and the old-jam reproduction
seq = []
for tag, dims, recipe, ods, off in ([(f"stack_{r}", {}, RECIPES[r], gba.RING_ODS, 2.0) for r in args.recipes] +
                                    [("oldjam_legacy", OLD_DIMS, RECIPES["legacy"], OLD_ODS, 3.6)]):
    sdir, S = build(tag, dims, recipe, ods)
    knob_top = S["base_h"] + S["peg_h"]
    base = np.array([-0.3, 0.2 * len(seq), 0.0])
    place(f"/World/{tag}_base", sdir / "bright_stacker_round.usd", base)
    rings = []
    for k in range(len(ods)):
        p = f"/World/{tag}_ring{k + 1}"
        spawns.setdefault(5 + 90 * k, []).append((p, sdir / f"bright_ring_{k + 1}.usd",
                                                  base + np.array([off / 1000.0 * 0.8, -off / 1000.0 * 0.6,
                                                                   knob_top + 0.005]), (1, 0, 0, 0)))
        rings.append({"path": p, "expected_z": S["base_h"] + S["ring_t"] * k})
    seq.append({"tag": tag, "base_path": f"/World/{tag}_base", "base": base, "rings": rings, "offset_mm": off,
                "r_in": S["ring_id"] / 2, "peg_r": S["peg_r"]})

tl = omni.timeline.get_timeline_interface()
for _ in range(5):
    app.update()
tl.play()
phx = get_physx_interface()
total = max(args.steps, 5 + 90 * 5 + 240)
for step in range(total):
    for path, usd, pos, q in spawns.get(step, []):
        place(path, usd, pos, q)
    app.update()


def pose(path):
    t = phx.get_rigidbody_transformation(path)
    return np.array(t["position"]), np.array(t["rotation"])   # xyzw


def classify(ring_path, base_path, expected_z, r_in, clearance):
    p, q = pose(ring_path)
    b, _ = pose(base_path)
    x, y, z, w = q
    up = 1 - 2 * (x * x + y * y)
    rel = p - b
    xy = float(np.linalg.norm(rel[:2]))
    dz = float(rel[2] - expected_z)
    if abs(dz) < 0.002 and up > 0.995 and xy < clearance + 0.0015:
        state = "seated"
    elif xy < r_in + 0.002 and dz > 0.002:
        state = "stuck"
    else:
        state = "off"
    return {"state": state, "dz_mm": round(dz * 1000, 1), "xy_mm": round(xy * 1000, 1),
            "tilt_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, up)))), 1)}


res = {"singles": [], "stacks": []}
for c in cases:
    r = classify(c["ring_path"], c["base_path"], c["expected_z"], c["r_in"], c["r_in"] - FINAL["peg_r"])
    bmove = float(np.linalg.norm(pose(c["base_path"])[0][:2] - c["base"][:2]))
    res["singles"].append({k: c[k] for k in ("recipe", "ring", "offset_mm", "tilt_deg")} | r |
                          {"base_moved_mm": round(bmove * 1000, 2)})
for s in seq:
    out = [classify(r["path"], s["base_path"], r["expected_z"], s["r_in"], s["r_in"] - s["peg_r"]) for r in s["rings"]]
    res["stacks"].append({"tag": s["tag"], "offset_mm": s["offset_mm"], "rings": out})
(OUT / "ring_fit_results.json").write_text(json.dumps(res, indent=1))
print("RINGFIT_RESULTS", json.dumps(res), flush=True)
tl.stop()
app.close()
