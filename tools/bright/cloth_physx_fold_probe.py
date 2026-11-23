#!/usr/bin/env python
"""Standalone PhysX surface-deformable fold probe (GPU), no robot.

Lays the BRIGHT towel cloth (same USD + profile as the X2Real env) on a static table, then drives
two near-corner vertex patches along a fold arc (positions + velocities written every physics
step = an ideal two-point grasp), releases, lets it settle and reports the same fold metrics as
manaenv/terms/cloth_terms.cloth_folded_in_half.  Used to separate cloth-parameter problems from
grasp/trajectory problems.

    python tools/bright/cloth_physx_fold_probe.py --out /tmp/fp --set body.self_collision=false
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp/cloth_physx_fold")
ap.add_argument("--set", action="append", default=[], help="profile override key.sub=value (json value)")
ap.add_argument("--lift", type=float, default=0.13, help="arc height of the near corners [m]")
ap.add_argument("--margin", type=float, default=0.01, help="land the near corners this far short of the far corners")
ap.add_argument("--end-dz", type=float, default=0.012, help="near-corner height above its start when released")
ap.add_argument("--arc-s", type=float, default=2.0)
ap.add_argument("--patch", type=int, default=1, help="grid cells around each corner that are driven")
ap.add_argument("--settle-s", type=float, default=2.5)
ap.add_argument("--dt", type=float, default=1 / 120)
ap.add_argument("--render", action="store_true")
ap.add_argument("--mode", default="state", choices=["state", "pos", "none"], help="how the grasped patch is driven")
ap.add_argument("--debug-writes", type=int, default=0)
ap.add_argument("--regrasp", action="store_true", help="after the fold, pick the folded towel at the shared edge and carry it")
ap.add_argument("--regrasp-layers", default="both", choices=["both", "top"])
ap.add_argument("--patch-r", type=float, default=0.019)
ap.add_argument("--lift-h", type=float, default=0.25)
args = ap.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, device="cuda:0", enable_cameras=args.render).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import DeformableObjectCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg  # noqa: E402
from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg  # noqa: E402

from manaenv.assets.local_discovery import resolve_usd_path  # noqa: E402
from manaenv.utils import surface_cloth as SC  # noqa: E402
from manaenv.utils.tensor_utils import as_torch  # noqa: E402

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
prof = copy.deepcopy(SC.PROFILES["bright_towel_30"])
for kv in args.set:
    key, val = kv.split("=", 1)
    d = prof
    parts = key.split(".")
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = json.loads(val)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
    dt=args.dt, device="cuda:0",
    physics=PhysxCfg(solver_type=1, min_position_iteration_count=32, min_velocity_iteration_count=4)))
table = sim_utils.CuboidCfg(size=(1.0, 1.0, 0.05), collision_props=sim_utils.CollisionPropertiesCfg(),
                            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.4)))
table.func("/World/table", table, translation=(0.0, 0.0, -0.025))
sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))
spawn = sim_utils.UsdFileCfg(
    usd_path=resolve_usd_path(prof["usd_asset"]),
    deformable_props=PhysxDeformableBodyPropertiesCfg(mass=float(prof["mass"]), **prof["body"]),
    physics_material=PhysxSurfaceDeformableBodyMaterialCfg(**prof["material"]))
spawn.func = SC._spawn_with_collision_offsets(spawn.func, prof["rest_offset"], prof["contact_offset"])
z0 = float(prof["lift_z"]) + 0.0
cloth = DeformableObjectCfg(prim_path="/World/towel", spawn=spawn,
                            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0))).class_type(
    DeformableObjectCfg(prim_path="/World/towel", spawn=spawn,
                        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z0))))

cam_ann = None
if args.render:
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    cam = UsdGeom.Camera.Define(stage, "/World/cam")
    cam.CreateFocalLengthAttr(18.0)
    eye, tgt = Gf.Vec3d(-0.45, -0.35, 0.40), Gf.Vec3d(0.0, 0.0, 0.0)
    m = Gf.Matrix4d().SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1)).GetInverse()
    cam.AddTranslateOp().Set(eye)
    q = m.ExtractRotationQuat()
    cam.AddOrientOp().Set(Gf.Quatf(q.GetReal(), *q.GetImaginary()))
    rp = rep.create.render_product("/World/cam", (640, 400))
    cam_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    cam_ann.attach([rp])

sim.reset()
dt = sim.get_physics_dt()
n = int(prof["grid"])
pos0 = as_torch(cloth.data.nodal_pos_w)[0].clone()
V = pos0.shape[0]
I = torch.arange(V, device=pos0.device) % n
J = torch.arange(V, device=pos0.device) // n
corners = {"nl": 0, "nr": n * (n - 1), "fl": n - 1, "fr": n * n - 1}   # local +x = far (i), j = side
k = args.patch
patch_l = torch.nonzero((I <= k) & (J <= k)).flatten()
patch_r = torch.nonzero((I <= k) & (J >= n - 1 - k)).flatten()
x_near = float(pos0[0, 0])
x_far = float(pos0[n - 1, 0])
R = (x_far - x_near - args.margin) / 2


def smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def arc_offset(s):
    th = math.pi * smooth(s)
    dz = args.end_dz * smooth(s)
    return torch.tensor([R * (1 - math.cos(th)), 0.0, args.lift * math.sin(th) + dz], device=pos0.device)


def metrics():
    p = as_torch(cloth.data.nodal_pos_w)[0]
    v = as_torch(cloth.data.nodal_vel_w)[0]
    c = {kk: p[i] for kk, i in corners.items()}
    a = p[:, 0]
    b = p[:, 1]
    return {
        "d_left": float(torch.linalg.norm(c["nl"][:2] - c["fl"][:2])),
        "d_right": float(torch.linalg.norm(c["nr"][:2] - c["fr"][:2])),
        "ext_a": float(a.max() - a.min()), "ext_b": float(b.max() - b.min()),
        "height": float(p[:, 2].max()), "zmin": float(p[:, 2].min()),
        "speed_max": float(torch.linalg.norm(v, dim=-1).max()),
        "speed_p95": float(torch.quantile(torch.linalg.norm(v, dim=-1), 0.95)),
        "near_z": [float(c["nl"][2]), float(c["nr"][2])],
    }


frames = []


def snap(tag):
    if cam_ann is None:
        return
    import omni.replicator.core as rep
    from PIL import Image
    rep.orchestrator.step(rt_subframes=2, pause_timeline=False)
    d = cam_ann.get_data()
    if d is not None and getattr(d, "size", 0):
        Image.fromarray(np.asarray(d)[..., :3]).save(OUT / f"{tag}.png")
        frames.append(tag)


log = []
t_settle0 = 0.8
t_arc = args.arc_s
t_hold = 0.3
steps = int((t_settle0 + t_arc + t_hold + args.settle_s) / dt)
base_l = base_r = None
wall0 = time.time()
snap_at = {int(t_settle0 / dt): "a_flat", int((t_settle0 + t_arc * 0.5) / dt): "b_mid_arc",
           int((t_settle0 + t_arc + t_hold) / dt): "c_release", steps - 1: "d_settled"}
for step in range(steps):
    t = step * dt
    driving = t_settle0 <= t < t_settle0 + t_arc + t_hold
    if driving:
        p = as_torch(cloth.data.nodal_pos_w)
        v = as_torch(cloth.data.nodal_vel_w)
        if base_l is None:
            base_l = p[0, patch_l].clone()
            base_r = p[0, patch_r].clone()
        s = (t - t_settle0) / t_arc
        off = arc_offset(s)
        off_next = arc_offset(s + dt / t_arc)
        vel = (off_next - off) / dt
        state = torch.cat([p[0], v[0]], dim=-1).unsqueeze(0).clone()
        state[0, patch_l, :3] = base_l + off
        state[0, patch_r, :3] = base_r + off
        state[0, patch_l, 3:] = vel
        state[0, patch_r, 3:] = vel
        if args.mode == "state":
            cloth.write_nodal_state_to_sim_index(state)
        elif args.mode == "pos":
            cloth.write_nodal_pos_to_sim_index(state[..., :3].contiguous())
        if args.debug_writes > 0:
            args.debug_writes -= 1
            prev = p[0].clone()
    sim.step(render=bool(cam_ann) and step in snap_at)
    cloth.update(dt)
    if driving and args.mode != "none" and "prev" in dir() and args.debug_writes >= 0 and step < int(t_settle0 / dt) + 6:
        p2 = as_torch(cloth.data.nodal_pos_w)[0]
        tgt = state[0, :, :3]
        mask = torch.ones(V, dtype=torch.bool, device=p2.device); mask[patch_l] = False; mask[patch_r] = False
        print("DEBUG step", step, "patch err", float((p2[patch_l] - tgt[patch_l]).norm(dim=-1).max()),
              "free move", float((p2[mask] - prev[mask]).norm(dim=-1).max()),
              "vel max", float(as_torch(cloth.data.nodal_vel_w)[0].norm(dim=-1).max()), flush=True)
    if step in snap_at:
        snap(snap_at[step])
    if step % 24 == 0 or step == steps - 1:
        m = metrics()
        m["t"] = round(t, 3)
        log.append(m)
        if step % 120 == 0:
            print("PROGRESS", step, steps, round(time.time() - wall0, 1), json.dumps(m), flush=True)
regrasp_log = []
if args.regrasp:
    p = as_torch(cloth.data.nodal_pos_w)[0]
    far_mid = (p[n - 1] + p[n * n - 1]) / 2
    target = far_mid.clone()
    target[0] -= 0.02
    near_xy = torch.linalg.norm(p[:, :2] - target[:2], dim=-1) < 0.02
    target[2] = p[near_xy, 2].max() if near_xy.any() else target[2]
    nearest = int(torch.argmin(torch.linalg.norm(p - target, dim=-1)))
    sel = torch.linalg.norm(p - p[nearest], dim=-1) < args.patch_r
    if args.regrasp_layers == "top":
        sel &= p[:, 2] > p[nearest, 2] - 0.002
    patch = torch.nonzero(sel).flatten()
    base = p[patch].clone()
    T_lift, T_carry, T_hold, T_rel = 2.5, 2.5, 1.0, 2.0

    def roff(tt):
        if tt < T_lift:
            return torch.tensor([0.0, 0.0, args.lift_h * smooth(tt / T_lift)], device=p.device)
        tt -= T_lift
        return torch.tensor([0.0, -0.30 * smooth(tt / T_carry), args.lift_h], device=p.device)

    n2 = int((T_lift + T_carry + T_hold + T_rel) / dt)
    for k in range(n2):
        tt = k * dt
        if tt < T_lift + T_carry + T_hold:
            pp = as_torch(cloth.data.nodal_pos_w)
            vv = as_torch(cloth.data.nodal_vel_w)
            off = roff(tt)
            vel = (roff(tt + dt) - off) / dt
            state = torch.cat([pp[0], vv[0]], dim=-1).unsqueeze(0).clone()
            state[0, patch, :3] = base + off
            state[0, patch, 3:] = vel
            cloth.write_nodal_state_to_sim_index(state)
        sim.step(render=False)
        cloth.update(dt)
        if k % 24 == 0 or k == n2 - 1:
            pp = as_torch(cloth.data.nodal_pos_w)[0]
            vv = torch.linalg.norm(as_torch(cloth.data.nodal_vel_w)[0], dim=-1)
            regrasp_log.append({"t": round(tt, 3), "zmin": float(pp[:, 2].min()), "zmax": float(pp[:, 2].max()),
                                "speed_max": float(vv.max()), "speed_p95": float(torch.quantile(vv, 0.95))})
    print("REGRASP patch", len(patch), json.dumps(regrasp_log[-1]), flush=True)
wall = time.time() - wall0
final = log[-1]
res = {"profile": prof, "args": vars(args), "steps": steps, "wall_s": round(wall, 2),
       "steps_per_s": round(steps / wall, 1), "final": final,
       "flat_before_grasp": next(m for m in log if m["t"] >= t_settle0 - 0.05),
       "pass_fold": final["d_left"] < 0.05 and final["d_right"] < 0.05 and 0.10 <= final["ext_a"] <= 0.20
       and 0.25 <= final["ext_b"] <= 0.34 and final["speed_max"] < 0.05, "log": log, "frames": frames, "regrasp_log": regrasp_log}
(OUT / "result.json").write_text(json.dumps(res, indent=1))
print("FOLD_PROBE", json.dumps({k: res[k] for k in ("steps_per_s", "final", "flat_before_grasp", "pass_fold")}), flush=True)
app.close()
os._exit(0)
