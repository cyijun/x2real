#!/usr/bin/env python
"""Probe: does PhysX cloth (surface deformable, GPU dynamics) run in this Isaac Sim 6.0.1 / GB10 build?

Creates a 30x30 cm towel (particle cloth, 31x31 vertices) above the green-booth table,
lets it fall for ~3 s, then lifts one edge kinematically by attaching nothing (pure drop),
and reports particle bounds + timing.  Renders before/after frames.

    python tools/bright/bright_cloth_probe.py --out /tmp/bright_cloth
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp/bright_cloth")
ap.add_argument("--asset-root", default=os.environ.get("MANAENV_ASSET_ROOT", "x2real-assets"))
ap.add_argument("--res", type=int, default=31)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "extra_args": ["--/UJITSO/textures=false"]})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(args.asset_root).resolve()
report: dict = {}

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
bg = UsdGeom.Xform.Define(stage, "/World/booth")
bg.GetPrim().GetReferences().AddReference(str(ROOT / "art_assets/scenes/green_booth/green_booth_mesh.usd"))
bg.AddTranslateOp().Set(Gf.Vec3d(-0.452, -0.271, -0.985))
bg.AddOrientOp().Set(Gf.Quatf(0.709, 0.0, 0.0, -0.704))

scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
px.CreateEnableGPUDynamicsAttr(True)
px.CreateBroadphaseTypeAttr("GPU")
px.CreateTimeStepsPerSecondAttr(120)

# Isaac Sim 6.0 removed PhysxParticleClothAPI; cloth is a PhysX *surface deformable* (GPU only).
from omni.physx.scripts import deformableUtils, physicsUtils  # noqa: E402

PhysxSchema.PhysxSceneAPI(scene).GetGpuMaxDeformableSurfaceContactsAttr().Set(1048576)
n = args.res
size = 0.30
table = np.array([0.205, -0.273, -0.2404])
body = UsdGeom.Xform.Define(stage, "/World/towel")
pts, idx = [], []
for j in range(n):
    for i in range(n):
        pts.append(Gf.Vec3f(size * (i / (n - 1) - 0.5), size * (j / (n - 1) - 0.5), 0.0))
for j in range(n - 1):
    for i in range(n - 1):
        a = j * n + i
        idx += [a, a + 1, a + n + 1, a + n]
mesh = UsdGeom.Mesh.Define(stage, "/World/towel/mesh")
mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * (len(idx) // 4)))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
mesh.CreateDoubleSidedAttr(True)
physicsUtils.setup_transform_as_scale_orient_translate(mesh)
physicsUtils.set_or_add_translate_op(body, Gf.Vec3f(*[float(v) for v in table + np.array([0.0, 0.0, 0.10])]))
physicsUtils.set_or_add_orient_op(body, Gf.Quatf(0.9914, Gf.Vec3f(0.0, 0.1305, 0.0)))   # 15 deg tilt
physicsUtils.set_or_add_scale_op(body, Gf.Vec3f(1.0))
deformableUtils.create_auto_surface_deformable_hierarchy(
    stage, root_prim_path=body.GetPath(), simulation_mesh_path="/World/towel/simMesh",
    cooking_src_mesh_path=mesh.GetPath(), cooking_src_simplification_enabled=False,
    set_visibility_with_guide_purpose=True)
bp = body.GetPrim()
bp.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
bp.GetAttribute("physxDeformableBody:selfCollision").Set(True)
bp.GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(16)
sim_prim = stage.GetPrimAtPath("/World/towel/simMesh")
sim_prim.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
PhysxSchema.PhysxCollisionAPI(sim_prim).GetRestOffsetAttr().Set(0.002)
PhysxSchema.PhysxCollisionAPI(sim_prim).GetContactOffsetAttr().Set(0.006)
deformableUtils.add_surface_deformable_material(stage, "/World/towelMaterial", dynamic_friction=0.8,
                                                surface_thickness=0.003, surface_stretch_stiffness=2000.0,
                                                surface_shear_stiffness=0.5, surface_bend_stiffness=0.05)
physicsUtils.add_physics_material_to_prim(stage, bp, "/World/towelMaterial")
mat = UsdShade.Material.Define(stage, "/World/Looks/towel")
sh = UsdShade.Shader.Define(stage, "/World/Looks/towel/Shader")
sh.CreateIdAttr("UsdPreviewSurface")
sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.2, 0.45, 0.9))
mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
# a rigid box under one half so the towel drapes over an edge (shows bending)
box = UsdGeom.Cube.Define(stage, "/World/box")
box.CreateSizeAttr(1.0)
box.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in table + np.array([0.08, 0.0, 0.03])]))
box.AddScaleOp().Set(Gf.Vec3f(0.12, 0.30, 0.06))
UsdPhysics.CollisionAPI.Apply(box.GetPrim())
cam = UsdGeom.Camera.Define(stage, "/World/cam")
cam.CreateFocalLengthAttr(18.0)
cam.CreateHorizontalApertureAttr(20.955)
eye = Gf.Vec3d(*(table + np.array([-0.45, 0.0, 0.45])))
m = Gf.Matrix4d().SetLookAt(eye, Gf.Vec3d(*table), Gf.Vec3d(0, 0, 1)).GetInverse()
cam.AddTranslateOp().Set(eye)
q = m.ExtractRotationQuat()
cam.AddOrientOp().Set(Gf.Quatf(q.GetReal(), *q.GetImaginary()))
rp = rep.create.render_product("/World/cam", (960, 540))
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach([rp])


def snap(tag):
    from PIL import Image
    rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
    d = ann.get_data()
    if d is not None and getattr(d, "size", 0):
        Image.fromarray(np.asarray(d)[..., :3]).save(OUT / f"{tag}.png")


for _ in range(10):
    app.update()
snap("cloth_before")
tl = omni.timeline.get_timeline_interface()
tl.play()
t0 = time.time()
err = None
try:
    for k in range(360):
        app.update()
        if k in (60, 180):
            snap(f"cloth_t{k}")
except Exception as exc:  # noqa: BLE001
    err = repr(exc)
dt = time.time() - t0
snap("cloth_after")
report["wall_s_for_360_updates"] = round(dt, 2)
report["error"] = err
try:
    pts_now = np.array(UsdGeom.Mesh(stage.GetPrimAtPath("/World/towel/simMesh")).GetPointsAttr().Get())
    xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/towel")).ComputeLocalToWorldTransform(0)
    wp = np.array([xf.Transform(Gf.Vec3d(*map(float, q))) for q in pts_now])
    report["sim_points"] = int(len(wp))
    report["z_min_minus_table"] = float(wp[:, 2].min() - table[2])
    report["z_max_minus_table"] = float(wp[:, 2].max() - table[2])
    report["note"] = "USD points (may be stale if fabric does not write back); judge by the rendered frames"
except Exception as exc:  # noqa: BLE001
    report["read_error"] = repr(exc)
(OUT / "cloth_probe.json").write_text(json.dumps(report, indent=1))
print("CLOTH_PROBE", json.dumps(report), flush=True)
tl.stop()
app.close()
