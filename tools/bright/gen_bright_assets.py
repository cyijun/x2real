#!/usr/bin/env python
"""Procedurally generate the BRIGHT 2026 hackathon prop assets for X2Real.

Writes USD files (Z-up, metres, default prim /World carrying RigidBodyAPI +
MassAPI, OmniPBR materials, physics material, invisible convex collision
pieces) under ``<asset_root>/bright``.  Run inside the manaenv conda env:

    python tools/bright/gen_bright_assets.py --asset-root x2real-assets

Textures: Codex-generated seamless PNGs are expected in
``<asset_root>/bright/textures`` (copied there separately); procedural masks
are generated here with PIL.  Every asset id starts with ``bright_`` so the
stem-keyed local asset index never collides with the official bundle.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

MANIFEST: dict[str, dict] = {}
TEX_DIR_NAME = "textures"


# ----------------------------------------------------------------------------
# Stage / material helpers
# ----------------------------------------------------------------------------

def new_stage(path: Path, *, mass: float, kinematic: bool = False) -> Usd.Stage:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("kilogramsPerUnit", 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")
    rb = UsdPhysics.RigidBodyAPI.Apply(world.GetPrim())
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(world.GetPrim()).CreateMassAttr(float(mass))
    UsdGeom.Xform.Define(stage, "/World/visual")
    UsdGeom.Xform.Define(stage, "/World/collision")
    return stage


def physics_material(stage, *, static: float, dynamic: float, restitution: float = 0.05, combine: str | None = None):
    mat = UsdShade.Material.Define(stage, "/World/Looks/PhysicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(static))
    api.CreateDynamicFrictionAttr(float(dynamic))
    api.CreateRestitutionAttr(float(restitution))
    if combine:  # PhysX default is "average" with the other surface; "min" keeps this low value
        prim = mat.GetPrim()
        prim.AddAppliedSchema("PhysxMaterialAPI")
        prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(combine)
    return mat


def omnipbr(stage, name: str, *, color=(0.8, 0.8, 0.8), texture: str | None = None, tint=None,
            roughness: float = 0.6, metallic: float = 0.0, opacity_texture: str | None = None,
            opacity_threshold: float = 0.5):
    mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if texture:
        sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f"../{TEX_DIR_NAME}/{texture}"))
        sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(tint or (1.0, 1.0, 1.0))))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if opacity_texture:
        sh.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
        sh.CreateInput("enable_opacity_texture", Sdf.ValueTypeNames.Bool).Set(True)
        sh.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(f"../{TEX_DIR_NAME}/{opacity_texture}"))
        sh.CreateInput("opacity_mode", Sdf.ValueTypeNames.Int).Set(0)  # mono / alpha channel
        sh.CreateInput("opacity_threshold", Sdf.ValueTypeNames.Float).Set(float(opacity_threshold))
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateDisplacementOutput("mdl").ConnectToSource(out)
    mat.CreateVolumeOutput("mdl").ConnectToSource(out)
    return mat


def omniglass(stage, name: str, *, color=(0.6, 0.8, 1.0), ior: float = 1.45, roughness: float = 0.05):
    mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
    sh.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    sh.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(float(ior))
    sh.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(True)
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
    mat.CreateVolumeOutput("mdl").ConnectToSource(out)
    return mat


def _bind(prim, mat, purpose=None):
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    if purpose:
        api.Bind(mat, UsdShade.Tokens.weakerThanDescendants, purpose)
    else:
        api.Bind(mat)


# ----------------------------------------------------------------------------
# Geometry helpers (numpy arrays of points, quads/tris faces, uvs)
# ----------------------------------------------------------------------------

def vertex_normals(pts: np.ndarray, counts, idx) -> np.ndarray:
    n = np.zeros_like(pts)
    k = 0
    for c in counts:
        f = idx[k:k + c]
        k += c
        for j in range(1, c - 1):
            a, b, cc = pts[f[0]], pts[f[j]], pts[f[j + 1]]
            fn = np.cross(b - a, cc - a)
            for v in (f[0], f[j], f[j + 1]):
                n[v] += fn
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln < 1e-12] = 1.0
    return n / ln


def add_mesh(stage, path: str, pts, counts, idx, uvs=None, *, mat=None, smooth=True):
    pts = np.asarray(pts, dtype=float)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([int(c) for c in counts]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in idx]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
    if smooth:
        nrm = vertex_normals(pts, counts, idx)
        mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in nrm]))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if uvs is not None:
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                     UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in uvs]))
    if mat is not None:
        _bind(mesh.GetPrim(), mat)
    return mesh


def lathe(profile, segments=64, u_tiles=1.0, v_scale=1.0, angle0=0.0, rmod=None):
    """Revolve [(r, z)] around Z. Returns pts, counts, idx, uvs (v = arc length * v_scale)."""
    prof = np.asarray(profile, dtype=float)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))])
    pts, uvs = [], []
    for s in range(segments + 1):
        th = angle0 + 2 * math.pi * s / segments
        for k, (r, z) in enumerate(prof):
            rr = r * (rmod(th, k) if rmod else 1.0)
            pts.append((rr * math.cos(th), rr * math.sin(th), z))
            uvs.append((u_tiles * s / segments, arc[k] * v_scale))
    m = len(prof)
    counts, idx = [], []
    for s in range(segments):
        for k in range(m - 1):
            a = s * m + k
            b = (s + 1) * m + k
            counts.append(4)
            idx += [a, b, b + 1, a + 1]
    return np.array(pts), counts, idx, uvs


def box_mesh(center, size, uv_scale=1.0):
    cx, cy, cz = center
    sx, sy, sz = (s / 2 for s in size)
    faces = [  # (normal axis, sign)
        ((0, 0, 1), [(-1, -1), (1, -1), (1, 1), (-1, 1)]),
        ((0, 0, -1), [(-1, 1), (1, 1), (1, -1), (-1, -1)]),
        ((1, 0, 0), [(-1, -1), (1, -1), (1, 1), (-1, 1)]),
        ((-1, 0, 0), [(1, -1), (-1, -1), (-1, 1), (1, 1)]),
        ((0, 1, 0), [(1, -1), (-1, -1), (-1, 1), (1, 1)]),
        ((0, -1, 0), [(-1, -1), (1, -1), (1, 1), (-1, 1)]),
    ]
    pts, uvs, counts, idx = [], [], [], []
    for n, quad in faces:
        n = np.array(n)
        base = len(pts)
        for a, b in quad:
            if n[2] != 0:
                p = (cx + a * sx, cy + b * sy, cz + n[2] * sz)
                uv = (p[0], p[1])
            elif n[0] != 0:
                p = (cx + n[0] * sx, cy + a * sy, cz + b * sz)
                uv = (p[1], p[2])
            else:
                p = (cx + a * sx, cy + n[1] * sy, cz + b * sz)
                uv = (p[0], p[2])
            pts.append(p)
            uvs.append((uv[0] * uv_scale, uv[1] * uv_scale))
        counts.append(4)
        idx += [base, base + 1, base + 2, base + 3]
    return np.array(pts), counts, idx, uvs


def merge(*parts):
    pts, counts, idx, uvs = [], [], [], []
    for p, c, i, u in parts:
        off = len(pts)
        pts += [tuple(x) for x in p]
        counts += list(c)
        idx += [j + off for j in i]
        uvs += list(u)
    return np.array(pts), counts, idx, uvs


def transform(part, rot=None, trans=(0, 0, 0)):
    p, c, i, u = part
    p = np.asarray(p, dtype=float)
    if rot is not None:
        p = p @ np.asarray(rot).T
    return p + np.asarray(trans), c, i, u


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---- collision pieces -------------------------------------------------------

def add_convex(stage, name: str, pts, pmat):
    """Convex collision piece from arbitrary points (hull computed by PhysX)."""
    pts = np.asarray(pts, dtype=float)
    # Use a simple fan topology over all points; PhysX cooks the convex hull from the vertices.
    n = len(pts)
    counts = [3] * (n - 2)
    idx = []
    for j in range(1, n - 1):
        idx += [0, j, j + 1]
    mesh = UsdGeom.Mesh.Define(stage, f"/World/collision/{name}")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*pts.min(0)), Gf.Vec3f(*pts.max(0))]))
    mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    _bind(prim, pmat, "physics")
    return mesh


def box_pts(center, size, rot=None):
    c = np.asarray(center, float)
    h = np.asarray(size, float) / 2
    pts = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    if rot is not None:
        pts = pts @ np.asarray(rot).T
    return pts + c


def cyl_pts(r0, z0, r1, z1, n=16, cx=0.0, cy=0.0):
    pts = []
    for k in range(n):
        th = 2 * math.pi * k / n
        pts.append((cx + r0 * math.cos(th), cy + r0 * math.sin(th), z0))
        pts.append((cx + r1 * math.cos(th), cy + r1 * math.sin(th), z1))
    return np.array(pts)


def shell_collision(stage, pmat, outer, inner, segments=16, prefix="wall"):
    """Thin revolved wall: outer/inner are lists of (r, z) at matching band boundaries."""
    count = 0
    for b in range(len(outer) - 1):
        (ro0, z0), (ro1, z1) = outer[b], outer[b + 1]
        (ri0, zi0), (ri1, zi1) = inner[b], inner[b + 1]
        for s in range(segments):
            pts = []
            for th in (2 * math.pi * s / segments, 2 * math.pi * (s + 1) / segments):
                ct, st = math.cos(th), math.sin(th)
                for r, z in ((ro0, z0), (ro1, z1), (ri0, zi0), (ri1, zi1)):
                    pts.append((r * ct, r * st, z))
            add_convex(stage, f"{prefix}_{b}_{s}", pts, pmat)
            count += 1
    return count


def annulus_collision(stage, pmat, r_in, r_out, z0, z1, segments=24, prefix="ring", cx=0.0, cy=0.0):
    for s in range(segments):
        pts = []
        for th in (2 * math.pi * s / segments, 2 * math.pi * (s + 1) / segments):
            for r in (r_in, r_out):
                for z in (z0, z1):
                    pts.append((cx + r * math.cos(th), cy + r * math.sin(th), z))
        add_convex(stage, f"{prefix}_{s}", pts, pmat)


def record(asset_id, path, **info):
    MANIFEST[asset_id] = {"usd": str(path), **info}


def finish(stage, asset_id, path, **info):
    stage.GetRootLayer().Save()
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
        stage.GetPseudoRoot()).ComputeAlignedRange()
    lo, hi = bb.GetMin(), bb.GetMax()
    record(asset_id, path, bbox_min=[round(v, 4) for v in lo], bbox_max=[round(v, 4) for v in hi],
           size_m=[round(hi[i] - lo[i], 4) for i in range(3)], **info)


# ----------------------------------------------------------------------------
# Procedural textures
# ----------------------------------------------------------------------------

def make_procedural_textures(tex: Path):
    from PIL import Image, ImageDraw
    tex.mkdir(parents=True, exist_ok=True)
    # Basket hole mask: one tile = one hole cell; alpha 0 in the hole.
    size = 256
    img = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([40, 52, 216, 204], radius=40, fill=(255, 255, 255, 0))
    img.save(tex / "basket_hole_mask.png")
    # Plain white helper texture.
    Image.new("RGB", (64, 64), (255, 255, 255)).save(tex / "white.png")


# ----------------------------------------------------------------------------
# Assets
# ----------------------------------------------------------------------------
STRAW = "straw_plastic.png"
BEECH = "wood_beech.png"
SAPELE = "wood_sapele.png"
YARN = "yarn_knit.png"
TOWEL = "microfiber_towel.png"
PLASTIC = "plastic_mesh_basket.png"


def tex_or_none(root: Path, name: str):
    return name if (root / TEX_DIR_NAME / name).is_file() else None


def gen_bowl(root: Path, color_name: str, color):
    aid = f"bright_bowl_{color_name}"
    path = root / "tableware" / f"{aid}.usd"
    mass = 0.13   # "摔不坏" straw-plastic bowl, 20 cm: ~130 g (light, so a pour can push it)
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.25, dynamic=0.2, combine="min")   # smooth PP: slides easily -> one hand must hold it
    # final (2026-11-22 PDF): wheat-straw bowl, rim diameter 20 cm, height 10 cm, light matte colour
    R, H, rf, t, tb = 0.100, 0.0985, 0.042, 0.0035, 0.005   # foot 8.4 cm: seller spec image lu_1747_1 (measured)
    zf = 0.008  # foot height
    n_band = 6
    outer = [(rf * 0.92, 0.0), (rf, 0.0), (rf, zf)]
    inner_bands, outer_bands = [], [(rf, zf)]
    for k in range(1, n_band + 1):
        s = k / n_band
        r = rf + (R - rf) * (1 - (1 - s) ** 2.2)
        z = zf + (H - zf) * s
        outer.append((r, z))
        outer_bands.append((r, z))
    inner = []
    for (r, z) in reversed(outer_bands):
        inner.append((max(r - t, 0.0), min(z, H)))
    inner[-1] = (rf - t, zf + tb)
    # visual only: the PDF bowl has a few shallow horizontal ribs on the lower outside
    ribbed = []
    for (r0, z0), (r1, z1) in zip(outer[2:], outer[3:]):
        ribbed.append((r0, z0))
        if z1 < 0.06:
            for f in (0.3, 0.5, 0.7):
                ribbed.append((r0 + (r1 - r0) * f + (0.0012 if f == 0.5 else 0.0), z0 + (z1 - z0) * f))
    ribbed.append(outer[-1])
    profile = [(0.0, 0.0)] + outer[:2] + ribbed + [(R - t * 0.5, H + 0.0015)] + inner + [(0.0, zf + tb)]
    look = omnipbr(st, "straw", color=color, texture=tex_or_none(root, STRAW), tint=color, roughness=0.8)
    p = lathe(profile, segments=72, u_tiles=4.0, v_scale=12.0)
    add_mesh(st, "/World/visual/bowl", *p, mat=look)
    # collision: foot disk + revolved wall bands
    add_convex(st, "bottom", cyl_pts(rf, 0.0, rf, zf + tb, n=20), pm)
    inner_b = [(r - t, z) for (r, z) in outer_bands]
    inner_b[0] = (rf - t, zf + tb)
    shell_collision(st, pm, outer_bands, inner_b, segments=20)
    finish(st, aid, path, mass_kg=mass, friction=[0.25, 0.2], color=color,
           collision="foot cylinder + 20x6 convex wall pieces", texture=STRAW)


def gen_cup(root: Path, color_name: str, color, *, aid=None, r_bot=0.027, r_top=0.039, H=0.105, mass=0.045,
            glass=False, texture=STRAW):
    # final (PDF): wheat-straw cup 350 ml, mouth 7.8 cm, height 10.5 cm; bottom 5.4 cm measured on the seller's spec
    # image lu_1747_5 (5.3-5.5 cm)
    aid = aid or f"bright_cup_{color_name}"
    path = root / "tableware" / f"{aid}.usd"
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.5, dynamic=0.4)
    t, tb = 0.0025, 0.004
    outer = [(r_bot, 0.0), (r_top, H)]
    profile = [(0.0, 0.0), (r_bot - 0.002, 0.0), (r_bot, 0.002), (r_top, H), (r_top - t, H),
               (r_bot - t, tb), (0.0, tb)]
    if glass:
        look = omniglass(st, "glass", color=color)
    else:
        look = omnipbr(st, "straw", color=color, texture=tex_or_none(root, texture), tint=color, roughness=0.75)
    p = lathe(profile, segments=64, u_tiles=4.0, v_scale=12.0)
    add_mesh(st, "/World/visual/cup", *p, mat=look)
    if glass:  # graduation marks as thin opaque rings
        marks = omnipbr(st, "marks", color=(0.05, 0.15, 0.6), roughness=0.4)
        for k in range(1, 5):
            z = tb + (H - tb) * k / 5
            r = r_bot + (r_top - r_bot) * z / H + 0.0003
            pm_ = lathe([(r, z - 0.0006), (r, z + 0.0006)], segments=8, angle0=-0.35)
            pts = pm_[0]
            # restrict to a short arc: rebuild with small angular span
            arc_pts, uv = [], []
            for s in range(9):
                th = -0.35 + 0.7 * s / 8
                for zz in (z - 0.0006, z + 0.0006):
                    arc_pts.append((r * math.cos(th), r * math.sin(th), zz))
                    uv.append((0, 0))
            cnt, ix = [], []
            for s in range(8):
                a = 2 * s
                cnt.append(4)
                ix += [a, a + 2, a + 3, a + 1]
            add_mesh(st, f"/World/visual/mark_{k}", np.array(arc_pts), cnt, ix, uv, mat=marks)
    add_convex(st, "bottom", cyl_pts(r_bot, 0.0, r_bot + (r_top - r_bot) * tb / H, tb, n=16), pm)
    rb2 = r_bot + (r_top - r_bot) * 0.5
    outer_b = [(r_bot, tb), (rb2, H * 0.5), (r_top, H)]
    inner_b = [(r - t, z) for (r, z) in outer_b]
    shell_collision(st, pm, outer_b, inner_b, segments=16)
    finish(st, aid, path, mass_kg=mass, friction=[0.5, 0.4], color=color,
           collision="bottom disk + 16x2 convex wall pieces", texture=("OmniGlass" if glass else texture),
           opening_inner_radius_m=round(r_top - t, 4),
           anchors={"rim_surface_z": round(H + 0.004, 4), "floor_z": tb})


def gen_basket(root: Path, color_name: str, color, *, aid=None, outer_top=(0.35, 0.255), outer_bot=(0.32, 0.24),
               H=0.085, holes=True, mass=0.30):
    aid = aid or f"bright_basket_{color_name}"
    path = root / "containers" / f"{aid}.usd"
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.45, dynamic=0.35)
    t = 0.004
    look = omnipbr(st, "plastic", color=color, texture=tex_or_none(root, PLASTIC), tint=color, roughness=0.55,
                   opacity_texture=("basket_hole_mask.png" if holes else None))
    solid = omnipbr(st, "plastic_solid", color=color, texture=tex_or_none(root, PLASTIC), tint=color, roughness=0.55)
    (tx, ty), (bx, by) = outer_top, outer_bot
    # walls as hexahedra; visual = same boxes (thin) so holes show through
    walls = []
    for sgn in (-1, 1):
        # walls normal to x
        walls.append([(sgn * bx / 2, -by / 2, 0), (sgn * bx / 2, by / 2, 0), (sgn * tx / 2, ty / 2, H),
                      (sgn * tx / 2, -ty / 2, H)])
        walls.append([(-bx / 2, sgn * by / 2, 0), (bx / 2, sgn * by / 2, 0), (tx / 2, sgn * ty / 2, H),
                      (-tx / 2, sgn * ty / 2, H)])
    vis_parts = []
    for k, quad in enumerate(walls):
        q = np.array(quad, float)
        c = np.array([0, 0, q[:, 2].mean()])
        # inward offset along horizontal direction to the centre
        inward = []
        for p in q:
            d = -np.array([p[0], p[1], 0.0])
            d = d / (np.linalg.norm(d) + 1e-9)
            # project to the wall normal (axis aligned)
            if abs(q[0, 0] - q[1, 0]) < 1e-9 and abs(q[0, 0]) > 1e-6:  # x-wall
                d = np.array([-np.sign(p[0]), 0, 0])
            else:
                d = np.array([0, -np.sign(p[1]), 0])
            inward.append(p + d * t)
        hexa = np.vstack([q, np.array(inward)])
        add_convex(st, f"wall_{k}", hexa, pm)
        # visual quad pair (outer + inner) with UVs in hole-cell units (2.2 cm x 1.6 cm cells)
        for j, quad_pts in enumerate((q, np.array(inward))):
            along = np.linalg.norm(quad_pts[1] - quad_pts[0])
            uv = [(0, 0), (along / 0.022, 0), (along / 0.022, H / 0.016), (0, H / 0.016)]
            vis_parts.append((quad_pts, [4], [0, 1, 2, 3], uv))
    p = merge(*vis_parts)
    add_mesh(st, "/World/visual/walls", *p, mat=look, smooth=False)
    # bottom (holes too) and rim band (solid)
    bot = box_mesh((0, 0, t / 2), (bx, by, t), uv_scale=1.0)
    bp, bc, bi, bu = bot
    bu = [(x / 0.022, y / 0.016) for x, y in ((pp[0], pp[1]) for pp in bp)]
    add_mesh(st, "/World/visual/bottom", bp, bc, bi, bu, mat=(look if holes else solid), smooth=False)
    add_convex(st, "bottom", box_pts((0, 0, t / 2), (bx, by, t)), pm)
    rim_parts = []
    rh = 0.012
    for sgn in (-1, 1):
        rim_parts.append(box_mesh((sgn * (tx / 2 - t / 2), 0, H - rh / 2), (t * 1.6, ty, rh), 20))
        rim_parts.append(box_mesh((0, sgn * (ty / 2 - t / 2), H - rh / 2), (tx, t * 1.6, rh), 20))
    add_mesh(st, "/World/visual/rim", *merge(*rim_parts), mat=solid, smooth=False)
    finish(st, aid, path, mass_kg=mass, friction=[0.45, 0.35], color=color,
           collision="bottom slab + 4 tapered wall hexahedra", texture=PLASTIC + (" + hole cutout mask" if holes else ""),
           inner_top_m=[round(tx - 2 * t, 3), round(ty - 2 * t, 3)])


def gen_tray(root: Path, color_name: str, color, L=0.40, W=0.30, H=0.025, mass=0.35):
    aid = f"bright_tray_{color_name}"
    path = root / "containers" / f"{aid}.usd"
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.25, dynamic=0.2)
    look = omnipbr(st, "plastic", color=color, texture=tex_or_none(root, PLASTIC), tint=color, roughness=0.35)
    t, rw = 0.004, 0.012
    parts = [box_mesh((0, 0, t / 2), (L, W, t), 5)]
    add_convex(st, "bottom", box_pts((0, 0, t / 2), (L, W, t)), pm)
    k = 0
    for sgn in (-1, 1):
        for c, s in (((sgn * (L / 2 - rw / 2), 0, H / 2), (rw, W, H)), ((0, sgn * (W / 2 - rw / 2), H / 2), (L, rw, H))):
            parts.append(box_mesh(c, s, 5))
            add_convex(st, f"rim_{k}", box_pts(c, s), pm)
            k += 1
    add_mesh(st, "/World/visual/tray", *merge(*parts), mat=look, smooth=False)
    finish(st, aid, path, mass_kg=mass, friction=[0.25, 0.2], color=color, collision="bottom + 4 rim boxes",
           texture=PLASTIC)


TOOLBOX_DECAL = "toolbox_endpanel.png"   # Codex: beech end panel with the grey gear/hammer print


def gen_toolbox(root: Path):
    """Final (PDF 2026-11-22) kids' wooden toolbox 20.7 x 12.7 x 12.7 cm, used with the top dowel and the middle
    four-hole boards REMOVED: an open box with two tall end panels (oblong handle slot) and lower side walls."""
    aid = "bright_toolbox"
    path = root / "tools" / f"{aid}.usd"
    mass = 0.35   # pine/beech plywood toy box, inferred
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.7, dynamic=0.6)
    wood = omnipbr(st, "wood", color=(0.88, 0.75, 0.56), texture=tex_or_none(root, BEECH), roughness=0.7)
    L, W, endH = 0.207, 0.127, 0.127          # from the PDF
    H, te, ts, tb = 0.065, 0.010, 0.008, 0.008  # side wall ~6-6.5 cm, end board ~1 cm: measured on seller image lu_1747_3
    slot_w, slot_z0, slot_z1 = 0.060, 0.090, 0.110   # handle slot ~6 x 2 cm, top 1.5-2 cm below the rim (lu_1747_3)
    boxes = [((0, 0, tb / 2), (L - 2 * te, W, tb)),                       # bottom (between the end panels)
             ((0, W / 2 - ts / 2, H / 2), (L - 2 * te, ts, H)),           # side +y
             ((0, -W / 2 + ts / 2, H / 2), (L - 2 * te, ts, H))]          # side -y
    for sgn in (-1, 1):                                                   # end panels, split around the slot
        x = sgn * (L / 2 - te / 2)
        post_w = (W - slot_w) / 2
        boxes += [((x, 0, slot_z0 / 2), (te, W, slot_z0)),
                  ((x, 0, (slot_z1 + endH) / 2), (te, W, endH - slot_z1)),
                  ((x, W / 2 - post_w / 2, (slot_z0 + slot_z1) / 2), (te, post_w, slot_z1 - slot_z0)),
                  ((x, -W / 2 + post_w / 2, (slot_z0 + slot_z1) / 2), (te, post_w, slot_z1 - slot_z0))]
    vis = []
    for k, (c, s) in enumerate(boxes):
        vis.append(box_mesh(c, s, 1 / 0.25))
        add_convex(st, f"box_{k}", box_pts(c, s), pm)
    add_mesh(st, "/World/visual/box", *merge(*vis), mat=wood, smooth=False)
    # printed grey gear / hammer graphic on the outside of both end panels (visual decal, below the slot)
    decal_tex = tex_or_none(root, TOOLBOX_DECAL)
    if decal_tex:
        decal = omnipbr(st, "endpanel_print", color=(0.88, 0.75, 0.56), texture=decal_tex, roughness=0.7)
        for sgn in (-1, 1):
            x = sgn * (L / 2 + 0.0004)
            ys = (W / 2, -W / 2) if sgn > 0 else (-W / 2, W / 2)
            q = np.array([(x, ys[0], 0.0), (x, ys[1], 0.0), (x, ys[1], slot_z0), (x, ys[0], slot_z0)])
            add_mesh(st, f"/World/visual/print_{'p' if sgn > 0 else 'm'}", q, [4], [0, 1, 2, 3],
                     [(0, 0), (1, 0), (1, slot_z0 / endH * 1.0), (0, slot_z0 / endH * 1.0)], mat=decal, smooth=False)
    inner = (round(L - 2 * te, 4), round(W - 2 * ts, 4))
    finish(st, aid, path, mass_kg=mass, friction=[0.7, 0.6],
           collision="bottom + 2 side walls + 2 end panels (4 boxes each, around the handle slot)",
           texture=BEECH + (f" + {TOOLBOX_DECAL}" if decal_tex else ""), inner_m=list(inner), wall_height_m=H,
           end_panel_height_m=endH, open_top=True,
           note="PDF: 20.7x12.7x12.7 cm, top dowel + middle four-hole boards removed (fully open). Side-wall height, "
                "board thickness and slot size inferred from the product photo")


def _cyl_x(r0, r1, length, n=24, v_scale=5.0):
    """Lathe a (possibly tapered) solid cylinder along +X starting at x=0."""
    p = lathe([(0, 0), (r0, 0), (r1, length), (0, length)], n, 1, v_scale)
    return transform(p, rot_y(math.pi / 2))


def _sphere_pts(c, r, n=10):
    pts = []
    for j in range(1, n):
        ph = math.pi * j / n
        for k in range(n):
            th = 2 * math.pi * k / n
            pts.append((c[0] + r * math.sin(ph) * math.cos(th), c[1] + r * math.sin(ph) * math.sin(th),
                        c[2] + r * math.cos(ph)))
    return np.array(pts + [(c[0], c[1], c[2] + r), (c[0], c[1], c[2] - r)])


def gen_hammer(root: Path):
    """PDF: toy hammer 16.5 x 7.5 cm, grey plastic head, wooden handle. Lying flat along +X (head at +X);
    the head is a barrel along Y.  Handle / head diameters inferred from the product photo."""
    aid = "bright_hammer"
    path = root / "tools" / f"{aid}.usd"
    mass = 0.05
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.6, dynamic=0.5)
    wood = omnipbr(st, "wood", color=(0.88, 0.72, 0.52), texture=tex_or_none(root, BEECH), roughness=0.6)
    grey = omnipbr(st, "grey_plastic", color=(0.42, 0.44, 0.47), roughness=0.4)
    total, head_len, rh = 0.165, 0.075, 0.0125       # along X; head 7.5 (labelled), face ~2.5 cm (lu_1747_3, measured)
    r_handle0, r_handle1 = 0.0080, 0.0072             # handle ~1.5 cm (measured), slight taper toward the head
    x0 = -total / 2
    zc = rh                                           # handle/head axis height when lying
    hl = total - rh                                   # handle runs into the head centre
    handle = transform(_cyl_x(r_handle0, r_handle1, hl, 24), None, (x0, 0, zc))
    add_mesh(st, "/World/visual/handle", *handle, mat=wood)
    xh = x0 + total - rh
    prof = [(0, -head_len / 2), (rh * 0.8, -head_len / 2), (rh, -head_len / 2 + 0.004), (rh, head_len / 2 - 0.012),
            (rh * 0.85, head_len / 2 - 0.004), (rh * 0.75, head_len / 2), (0, head_len / 2)]
    head = lathe(prof, 32, 1, 5)
    head = transform(head, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), (xh, 0, zc))   # lathe axis Z -> Y
    add_mesh(st, "/World/visual/head", *head, mat=grey)
    add_convex(st, "handle", cyl_pts(r_handle0, 0, r_handle1, hl, n=12) @ rot_y(math.pi / 2).T + np.array([x0, 0, zc]), pm)
    hp = cyl_pts(rh, -head_len / 2, rh, head_len / 2, n=16) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T
    add_convex(st, "head", hp + np.array([xh, 0, zc]), pm)
    finish(st, aid, path, mass_kg=mass, friction=[0.6, 0.5], collision="handle + head convex cylinders",
           texture=BEECH, note="PDF 16.5x7.5 cm, grey plastic head, wooden handle; diameters inferred")


def gen_wrench(root: Path):
    aid = "bright_wrench"
    path = root / "tools" / f"{aid}.usd"
    mass = 0.04
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.6, dynamic=0.5)
    wood = omnipbr(st, "wood", color=(0.5, 0.65, 0.9), texture=tex_or_none(root, BEECH), tint=(0.55, 0.7, 1.0),
                   roughness=0.6)
    T = 0.014
    boxes = [((0, 0, T / 2), (0.12, 0.024, T))]
    for sgn in (-1, 1):
        cx = sgn * 0.072
        boxes.append(((cx, 0, T / 2), (0.024, 0.046, T)))            # jaw block
        boxes.append(((sgn * 0.093, 0.0155, T / 2), (0.018, 0.015, T)))   # prongs
        boxes.append(((sgn * 0.093, -0.0155, T / 2), (0.018, 0.015, T)))
    parts = []
    for k, (c, s) in enumerate(boxes):
        parts.append(box_mesh(c, s, 1 / 0.2))
        add_convex(st, f"box_{k}", box_pts(c, s), pm)
    add_mesh(st, "/World/visual/wrench", *merge(*parts), mat=wood, smooth=False)
    finish(st, aid, path, mass_kg=mass, friction=[0.6, 0.5], collision="7 boxes", texture=BEECH,
           note="flat double open-end wrench, 14 mm thick, blue stain")


def gen_screwdriver(root: Path):
    """PDF: flat-blade toy screwdriver 15 x 1.5 cm, red/orange painted wooden handle, wooden shaft.
    Lying along +X, blade tip at +X; handle/shaft split inferred from the photo."""
    aid = "bright_screwdriver"
    path = root / "tools" / f"{aid}.usd"
    mass = 0.025
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.6, dynamic=0.5)
    red = omnipbr(st, "red_paint", color=(0.93, 0.30, 0.12), roughness=0.35)
    wood = omnipbr(st, "wood", color=(0.88, 0.72, 0.52), texture=tex_or_none(root, BEECH), roughness=0.6)
    rh, lh, rs, ls = 0.0075, 0.090, 0.00425, 0.060        # 15 cm total, handle 1.5 (label) x 9, shaft 0.85 x 6 exposed
    x0 = -(lh + ls) / 2
    handle = lathe([(0, 0), (rh * 0.6, 0), (rh * 0.92, 0.004), (rh, 0.012), (rh, lh), (0, lh)], 24, 1, 5)
    add_mesh(st, "/World/visual/handle", *transform(handle, rot_y(math.pi / 2), (x0, 0, rh)), mat=red)
    shaft = lathe([(0, 0), (rs, 0), (rs, ls - 0.014), (0, ls - 0.014)], 16, 1, 5)
    add_mesh(st, "/World/visual/shaft", *transform(shaft, rot_y(math.pi / 2), (x0 + lh, 0, rh)), mat=wood)
    # flat blade: wedge from the round shaft to a 1.5 mm edge (blade plane = XY, i.e. lying flat)
    xb = x0 + lh + ls - 0.014
    blade = np.array([(xb, -rs, rh - rs * 0.8), (xb, rs, rh - rs * 0.8), (xb, rs, rh + rs * 0.8), (xb, -rs, rh + rs * 0.8),
                      (xb + 0.014, -rs * 1.1, rh - 0.0008), (xb + 0.014, rs * 1.1, rh - 0.0008),
                      (xb + 0.014, rs * 1.1, rh + 0.0008), (xb + 0.014, -rs * 1.1, rh + 0.0008)])
    faces = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]
    add_mesh(st, "/World/visual/blade", blade, [4] * 6, sum(faces, []), [(0, 0)] * 8, mat=wood, smooth=False)
    add_convex(st, "handle", cyl_pts(rh, 0, rh, lh, n=12) @ rot_y(math.pi / 2).T + np.array([x0, 0, rh]), pm)
    add_convex(st, "shaft", cyl_pts(rs, 0, rs, ls - 0.014, n=8) @ rot_y(math.pi / 2).T + np.array([x0 + lh, 0, rh]), pm)
    add_convex(st, "blade", blade, pm)
    finish(st, aid, path, mass_kg=mass, friction=[0.6, 0.5], collision="handle + shaft cylinders + blade wedge",
           texture=BEECH, note="PDF 15x1.5 cm; red/orange painted handle, wooden shaft and flat blade")


def gen_screw_twister(root: Path):
    """PDF '螺丝拧' 9.5 x 8.5 cm: red painted wooden hub coaxial with a grey threaded rod, and a wooden cross dowel
    through the hub carrying two red balls (T handle).  Lying with the rod along +X and the cross dowel along Y.
    Part sizes inferred from the product photo (page-5 / lu_1813)."""
    aid = "bright_screw_twister"
    path = root / "tools" / f"{aid}.usd"
    mass = 0.04
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.6, dynamic=0.5)
    red = omnipbr(st, "red_paint", color=(0.85, 0.07, 0.08), roughness=0.3)
    grey = omnipbr(st, "grey_thread", color=(0.62, 0.62, 0.60), roughness=0.55)
    wood = omnipbr(st, "wood", color=(0.88, 0.72, 0.52), texture=tex_or_none(root, BEECH), roughness=0.6)
    r_hub, l_hub = 0.014, 0.0375         # hub cylinder (coaxial with the rod): ~2.8 x 3.5-4 cm (lu_1747_3, measured)
    r_rod, l_rod, pitch = 0.00525, 0.0575, 0.006   # thread ~1.0-1.1 cm major diameter, 9.5 cm overall
    r_dowel, r_ball, span = 0.0035, 0.0115, 0.085  # cross dowel 0.6-0.8, balls 2.2-2.4, 8.5 cm span (label)
    total = l_hub + l_rod                # 9.5 cm along X
    x0 = -total / 2
    zc = r_hub                           # axis height when lying on the hub
    hub = lathe([(0, 0), (r_hub - 0.002, 0), (r_hub, 0.002), (r_hub, l_hub - 0.002), (r_hub - 0.002, l_hub), (0, l_hub)],
                32, 1, 5)
    add_mesh(st, "/World/visual/hub", *transform(hub, rot_y(math.pi / 2), (x0, 0, zc)), mat=red)
    # threaded rod: helical ridge via a radius modulation that depends on (angle, axial position)
    nz = 90
    zs = [l_rod * k / nz for k in range(nz + 1)]
    prof = [(0.0, 0.0)] + [(r_rod, z) for z in zs] + [(0.0, l_rod)]

    def rmod(th, k):
        if k == 0 or k == len(prof) - 1:
            return 1.0
        z = prof[k][1]
        return 1.0 + 0.16 * math.cos(th - 2 * math.pi * z / pitch)
    rod = lathe(prof, 24, 1, 5, rmod=rmod)
    add_mesh(st, "/World/visual/rod", *transform(rod, rot_y(math.pi / 2), (x0 + l_hub, 0, zc)), mat=grey)
    # cross dowel along Y through the hub centre + two red balls at the ends
    half = span / 2 - r_ball
    dow = lathe([(0, -half), (r_dowel, -half), (r_dowel, half), (0, half)], 16, 1, 5)
    dow = transform(dow, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), (x0 + l_hub / 2, 0, zc))
    add_mesh(st, "/World/visual/dowel", *dow, mat=wood)
    for sgn in (-1, 1):
        c = (x0 + l_hub / 2, sgn * half, zc)
        ball = lathe([(r_ball * math.sin(a), -r_ball * math.cos(a)) for a in np.linspace(0, math.pi, 13)], 24, 1, 5)
        add_mesh(st, f"/World/visual/ball_{'p' if sgn > 0 else 'm'}", *transform(ball, None, c), mat=red)
        add_convex(st, f"ball_{'p' if sgn > 0 else 'm'}", _sphere_pts(c, r_ball), pm)
    add_convex(st, "hub", cyl_pts(r_hub, 0, r_hub, l_hub, n=16) @ rot_y(math.pi / 2).T + np.array([x0, 0, zc]), pm)
    add_convex(st, "rod", cyl_pts(r_rod * 1.1, 0, r_rod * 1.1, l_rod, n=12) @ rot_y(math.pi / 2).T
               + np.array([x0 + l_hub, 0, zc]), pm)
    add_convex(st, "dowel", box_pts((x0 + l_hub / 2, 0, zc), (2 * r_dowel, 2 * half, 2 * r_dowel)), pm)
    finish(st, aid, path, mass_kg=mass, friction=[0.6, 0.5], collision="hub + rod cylinders, dowel box, 2 ball hulls",
           texture=BEECH, note="PDF 9.5x8.5 cm; red painted hub/balls, grey threaded rod; part sizes inferred")


def gen_saw(root: Path):
    """Toy wooden saw (distractor): flat blade with teeth + D-handle block, lying flat, along +x."""
    aid = "bright_saw"
    path = root / "tools" / f"{aid}.usd"
    mass = 0.05
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.6, dynamic=0.5)
    blade = omnipbr(st, "blade", color=(0.78, 0.8, 0.82), texture=tex_or_none(root, BEECH), tint=(0.85, 0.88, 0.92),
                    roughness=0.5)
    handle = omnipbr(st, "handle", color=(0.2, 0.55, 0.3), texture=tex_or_none(root, BEECH), tint=(0.35, 0.75, 0.4),
                     roughness=0.55)
    T = 0.008
    parts_b, parts_h = [box_mesh((0.03, 0, T / 2), (0.14, 0.045, T), 5)], []
    for k in range(13):                                   # teeth along the -y edge
        x = -0.035 + 0.0105 * k
        tooth = np.array([(x, -0.0225, 0), (x + 0.0105, -0.0225, 0), (x + 0.00525, -0.031, 0),
                          (x, -0.0225, T), (x + 0.0105, -0.0225, T), (x + 0.00525, -0.031, T)])
        parts_b.append((tooth, [3, 3, 4, 4, 4], [0, 2, 1, 3, 4, 5, 0, 1, 4, 3, 1, 2, 5, 4, 2, 0, 3, 5],
                        [(0, 0)] * 6))
    parts_h.append(box_mesh((-0.065, 0, 0.007), (0.06, 0.055, 0.014), 5))
    add_mesh(st, "/World/visual/blade", *merge(*parts_b), mat=blade, smooth=False)
    add_mesh(st, "/World/visual/handle", *merge(*parts_h), mat=handle, smooth=False)
    add_convex(st, "blade", box_pts((0.03, -0.003, T / 2), (0.14, 0.051, T)), pm)
    add_convex(st, "handle", box_pts((-0.065, 0, 0.007), (0.06, 0.055, 0.014)), pm)
    finish(st, aid, path, mass_kg=mass, friction=[0.6, 0.5], collision="2 boxes", texture=BEECH, note="distractor tool")


def gen_gear(root: Path):
    """Toy wooden gear (distractor): 10-tooth disc, 6 cm, 1.4 cm thick, with a centre hole."""
    aid = "bright_gear"
    path = root / "tools" / f"{aid}.usd"
    mass = 0.025
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.6, dynamic=0.5)
    look = omnipbr(st, "gear", color=(0.95, 0.6, 0.1), texture=tex_or_none(root, BEECH), tint=(1.0, 0.65, 0.15),
                   roughness=0.5)
    T, r_in = 0.014, 0.006
    g = lathe([(r_in, 0), (0.026, 0), (0.026, T), (r_in, T), (r_in, 0)], 60, 2, 10,
              rmod=lambda th, k: 1.0 + (0.15 if k in (1, 2) else 0.0) * (1 if math.cos(10 * th) > 0 else -1))
    add_mesh(st, "/World/visual/gear", *g, mat=look, smooth=False)
    add_convex(st, "gear", cyl_pts(0.028, 0, 0.028, T, n=20), pm)
    finish(st, aid, path, mass_kg=mass, friction=[0.6, 0.5], collision="convex disc", texture=BEECH,
           note="distractor tool")


def gen_lily(root: Path):
    """White crochet lily-of-the-valley spray with 4 bells (distractor). Authored like the roses:
    lying along +X, origin at the stem bottom; the top 8 cm arches and carries the bells."""
    aid = "bright_flower_lily_white"
    path = root / "flowers" / f"{aid}.usd"
    mass = 0.025
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.8, dynamic=0.7)
    green = (0.3, 0.62, 0.22)
    white = (0.95, 0.95, 0.92)
    stem_m = omnipbr(st, "yarn_stem", color=green, texture=tex_or_none(root, YARN), tint=green, roughness=0.9)
    bell_m = omnipbr(st, "yarn_bell", color=white, texture=tex_or_none(root, YARN), tint=white, roughness=0.9)
    rs = 0.003
    # stem polyline: straight 0.18 then an arch bending toward +y over 0.10
    path_pts = [np.array([x, 0.0, 0.0]) for x in np.linspace(0, 0.18, 4)]
    for a in np.linspace(0.15, 1.1, 6):
        path_pts.append(np.array([0.18 + 0.07 * math.sin(a), 0.07 * (1 - math.cos(a)), 0.0]))
    parts = []
    for k in range(len(path_pts) - 1):
        p0, p1 = path_pts[k], path_pts[k + 1]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        seg = lathe([(0, 0), (rs, 0), (rs, L), (0, L)], 10, 1, 20)
        ang = math.atan2(d[1], d[0])
        parts.append(transform(seg, rot_z(ang) @ rot_y(math.pi / 2), p0))
        add_convex(st, f"stem_{k}", cyl_pts(rs, 0, rs, L, n=8) @ (rot_z(ang) @ rot_y(math.pi / 2)).T + p0, pm)
    add_mesh(st, "/World/visual/stem", *merge(*parts), mat=stem_m)
    # one long leaf along the straight part
    n = 14
    leaf = np.array([(0.05 + 0.06 * (1 + math.cos(2 * math.pi * s / n)), 0.012 * math.sin(2 * math.pi * s / n) - 0.012,
                      0.0) for s in range(n)])
    add_mesh(st, "/World/visual/leaf", leaf, [n], list(range(n)), [(0, 0)] * n, mat=stem_m, smooth=False)
    # 4 bells hanging below the arch (along -z when lying -> sideways; fine for a distractor)
    for k, idx in enumerate((5, 6, 7, 8)):
        c = path_pts[idx] + np.array([0.0, -0.012, 0.0])
        bell = lathe([(0, 0.012), (0.004, 0.011), (0.007, 0.006), (0.008, 0.0), (0.0065, 0.0)], 16, 1, 1)
        bell = transform(bell, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), c)
        add_mesh(st, f"/World/visual/bell_{k}", *bell, mat=bell_m)
        add_convex(st, f"bell_{k}", cyl_pts(0.008, 0, 0.004, 0.012, n=8) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T + c, pm)
    finish(st, aid, path, mass_kg=mass, friction=[0.8, 0.7], collision="9 stem segments + 4 bell hulls",
           texture=YARN, note="distractor; authored along +X with the arch toward +Y")


# Montessori ring stacker, final (PDF 2026-11-22): round base diameter 8 cm, whole set 12.5 cm high, 5 rings,
# beech + sapele, unpainted, two tones.  Same product as Amazon B0DWJYN55J (Adena / Yunhe Xinqiao Toys): base 8 x 1.8
# cm (listing drawing); peg ~1.5 cm, hole ~1.6-1.75 cm, rings 1.8-2.0 cm thick (measured on the listing's part
# photos); ring ODs from our spec photo gu_1804_ring.jpg.  See ASSETS.md section 2 for sources and confidence.
# Used: peg + round knob 1.5 cm, hole 1.7 cm (1 mm per side), rings 1.9 cm, peg 10.7 cm above the base:
# 1.8 + 10.7 = 12.5 cm (5 x 1.9 = 9.5 + 1.2 cm knob above the top ring).
# Collision recipe ("plan A" of the research in ASSETS.md 4.2): 48 convex sectors per ring with the collision bore
# 0.25 mm/side looser than the visual one and a 1.5 mm x 45 deg lead-in at both faces, a 48-gon inscribed peg,
# contactOffset 4 mm / restOffset 0, maxDepenetrationVelocity 1 m/s, 32 position iterations, and a low-friction
# (0.3, combine=min) material on the peg + knob only.
STACK = {"base_r": 0.040, "base_h": 0.018, "peg_r": 0.0075, "peg_h": 0.107, "ring_id": 0.017, "ring_t": 0.019,
         "segments": 48, "peg_segments": 48, "contact_offset": 0.004, "rest_offset": 0.0, "hole_chamfer": 0.0015,
         "coll_hole_extra": 0.00025, "peg_friction": 0.3, "max_depenetration": 1.0, "position_iterations": 32}
RING_ODS = [0.070, 0.063, 0.056, 0.048, 0.040]    # ring 1 (largest, bottom) .. ring 5 (smallest, top)


def set_collision_offsets(stage, contact_offset, rest_offset):
    """PhysxCollisionAPI contact/rest offset on every collider under /World/collision (None = PhysX default)."""
    if contact_offset is None and rest_offset is None:
        return
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/collision")):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        prim.AddAppliedSchema("PhysxCollisionAPI")
        if contact_offset is not None:
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(float(contact_offset))
        if rest_offset is not None:
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(float(rest_offset))


def set_body_solver(stage, max_depenetration=None, position_iterations=None):
    """PhysxRigidBodyAPI on /World: max depenetration velocity and solver position iterations (None = default)."""
    if max_depenetration is None and position_iterations is None:
        return
    prim = stage.GetPrimAtPath("/World")
    prim.AddAppliedSchema("PhysxRigidBodyAPI")
    if max_depenetration is not None:
        prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(
            float(max_depenetration))
    if position_iterations is not None:
        prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
            int(position_iterations))


def peg_material(stage, friction):
    """Separate low-friction material for the peg/knob colliders (combine=min, so ring-peg friction = this value)."""
    mat = UsdShade.Material.Define(stage, "/World/Looks/PegPhysicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(friction))
    api.CreateDynamicFrictionAttr(float(friction))
    api.CreateRestitutionAttr(0.05)
    mat.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    mat.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("min")
    return mat


def gen_stacker(root: Path):
    S = STACK
    pr, ph, Hb, Rb = S["peg_r"], S["peg_h"], S["base_h"], S["base_r"]
    # --- main prop: round base + single peg with a plain rounded (domed) top
    aid = "bright_stacker_round"
    path = root / "stacker" / f"{aid}.usd"
    mass = 0.10   # solid beech disc 8 x 2.5 cm + peg (~700 kg/m^3)
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.7, dynamic=0.6)
    wood = omnipbr(st, "wood", color=(0.85, 0.7, 0.5), texture=tex_or_none(root, BEECH), roughness=0.65)
    base = lathe([(0, 0), (Rb - 0.004, 0), (Rb - 0.0012, 0.0012), (Rb, 0.004), (Rb, Hb - 0.004), (Rb - 0.0012, Hb - 0.0012),
                  (Rb - 0.004, Hb), (0, Hb)], 64, 2, 6)
    # knob = hemispherical end of the peg (same 1.6 cm diameter, as in the photo)
    dome = [(pr * math.cos(a), Hb + ph - pr + pr * math.sin(a)) for a in np.linspace(0, math.pi / 2, 9)]
    peg = lathe([(0, Hb), (pr, Hb)] + dome, 48, 1, 6)
    add_mesh(st, "/World/visual/base", *merge(base, peg), mat=wood)
    n = S["peg_segments"]
    add_convex(st, "base", cyl_pts(Rb, 0, Rb, Hb, n=32), pm)
    pmp = peg_material(st, S["peg_friction"]) if S.get("peg_friction") is not None else pm
    add_convex(st, "peg", cyl_pts(pr, Hb, pr, Hb + ph - pr, n=n), pmp)   # inscribed polygon: never fatter than pr
    domep = []
    for (r, z) in dome:
        for k in range(n):
            th = 2 * math.pi * k / n
            domep.append((r * math.cos(th), r * math.sin(th), z))
    add_convex(st, "dome", domep, pmp)
    set_collision_offsets(st, S["contact_offset"], S["rest_offset"])
    set_body_solver(st, S.get("max_depenetration"), S.get("position_iterations"))
    finish(st, aid, path, mass_kg=mass, friction=[0.7, 0.6],
           collision=f"base disk + {n}-gon peg + {n}-gon hemispherical knob (peg/knob friction "
                     f"{S.get('peg_friction')}, min); contactOffset {S['contact_offset']}, restOffset {S['rest_offset']}",
           texture=BEECH, base_diameter_m=2 * Rb, base_height_m=Hb, peg_diameter_m=2 * pr, peg_height_above_base_m=ph,
           note="PDF: base diameter 8 cm, set height 12.5 cm; base height / peg size inferred from the photo")
    # --- legacy long three-peg base (five_tasks draft); not used by any scene any more, kept for reference
    aid = "bright_stacker_base"
    path = root / "stacker" / f"{aid}.usd"
    st = new_stage(path, mass=0.35)
    pm = physics_material(st, static=0.7, dynamic=0.6)
    wood = omnipbr(st, "wood", color=(0.85, 0.7, 0.5), texture=tex_or_none(root, BEECH), roughness=0.65)
    L, W, Hl = 0.34, 0.09, 0.02
    parts = [box_mesh((0, 0, Hl / 2), (W, L, Hl), 1 / 0.2)]
    add_convex(st, "base", box_pts((0, 0, Hl / 2), (W, L, Hl)), pm)
    for k, py in enumerate((-0.10, 0.0, 0.10)):
        parts.append(transform(lathe([(0, Hl), (pr, Hl)] + [(r, z - Hb + Hl) for r, z in dome], 24, 1, 6), None, (0, py, 0)))
        add_convex(st, f"peg_{k}", cyl_pts(pr, Hl, pr, Hl + ph - pr, n=16, cy=py), pm)
    add_mesh(st, "/World/visual/base", *merge(*parts), mat=wood)
    finish(st, aid, path, mass_kg=0.35, friction=[0.7, 0.6], collision="base box + 3 pegs", texture=BEECH,
           note="legacy three-peg base, unused")


def ring_collision(stage, pmat, r_in, r_out, T, segments, chamfer=0.0):
    """Annulus of convex sectors; with chamfer > 0 the bore gets a 45-degree lead-in of that size at both faces
    (three z-layers per sector, each still convex)."""
    if chamfer <= 0.0:
        annulus_collision(stage, pmat, r_in, r_out, 0.0, T, segments=segments)
        return
    layers = [((0.0, r_in + chamfer), (chamfer, r_in)), ((chamfer, r_in), (T - chamfer, r_in)),
              ((T - chamfer, r_in), (T, r_in + chamfer))]
    for li, ((z0, ri0), (z1, ri1)) in enumerate(layers):
        for s in range(segments):
            pts = []
            for th in (2 * math.pi * s / segments, 2 * math.pi * (s + 1) / segments):
                c, sn = math.cos(th), math.sin(th)
                pts += [(ri0 * c, ri0 * sn, z0), (ri1 * c, ri1 * sn, z1), (r_out * c, r_out * sn, z0),
                        (r_out * c, r_out * sn, z1)]
            add_convex(stage, f"ring_{li}_{s}", pts, pmat)


def gen_ring(root: Path, size_name: str, r_out: float, texture: str, color):
    aid = f"bright_ring_{size_name}"
    path = root / "stacker" / f"{aid}.usd"
    r_in, T = STACK["ring_id"] / 2, STACK["ring_t"]
    vol = math.pi * (r_out ** 2 - r_in ** 2) * T
    mass = round(vol * 650.0, 4)
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.45, dynamic=0.35)   # sanded unpainted wood
    look = omnipbr(st, "wood", color=color, texture=tex_or_none(root, texture), roughness=0.55)
    b, bi = 0.003, max(STACK.get("hole_chamfer", 0.0), 0.0005)   # outer edge rounded 3 mm, bore lead-in
    prof = [(r_in + bi, 0), (r_out - b, 0), (r_out, b), (r_out, T - b), (r_out - b, T), (r_in + bi, T), (r_in, T - bi),
            (r_in, bi), (r_in + bi, 0)]
    add_mesh(st, "/World/visual/ring", *lathe(prof, 64, 6, 8), mat=look)
    nseg = STACK["segments"]
    # flat chords: put the chord mid-points (apothem) at r_in + extra so the polygonal bore is never tighter
    r_coll = (r_in + STACK.get("coll_hole_extra", 0.0)) / math.cos(math.pi / nseg) if STACK.get("coll_hole_extra") \
        is not None else r_in
    ring_collision(st, pm, r_coll, r_out, T, nseg, STACK.get("hole_chamfer", 0.0))
    set_collision_offsets(st, STACK["contact_offset"], STACK["rest_offset"])
    set_body_solver(st, STACK.get("max_depenetration"), STACK.get("position_iterations"))
    finish(st, aid, path, mass_kg=mass, friction=[0.45, 0.35],
           collision=f"{nseg} convex annular sectors; contactOffset {STACK['contact_offset']}, "
                     f"restOffset {STACK['rest_offset']}", texture=texture,
           outer_diameter_m=2 * r_out, inner_diameter_m=2 * r_in, thickness_m=T,
           note="inner diameter / thickness inferred -> measure on arrival")


def gen_vase(root: Path, color_name: str, color):
    """Final (PDF): unglazed (bisque) ceramic vase, white matte, mouth 11.5 cm, height 19.5 cm.  Silhouette taken
    from the PDF product picture (8 px/cm): foot ~9 cm, fluted belly ~13.6 cm at 6 cm, neck ~7.5 cm at 14 cm, flared
    wavy lip 11.5 cm.  Everything but mouth and height is inferred from that picture."""
    aid = f"bright_vase_{color_name}"
    path = root / "flowers" / f"{aid}.usd"
    mass = 0.7
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.7, dynamic=0.6)
    look = omnipbr(st, "bisque", color=color, roughness=0.9)
    t, tb, H = 0.005, 0.015, 0.195
    outer_b = [(0.045, 0.0), (0.046, 0.012), (0.058, 0.035), (0.065, 0.060), (0.062, 0.088), (0.051, 0.110),
               (0.0405, 0.128), (0.0375, 0.140), (0.043, 0.158), (0.052, 0.176), (0.0575, H)]
    inner_b = [(r - t, max(z, tb)) for r, z in outer_b]
    n_out = len(outer_b)
    profile = [(0, 0)] + outer_b + [(0.0575 - t / 2, H + 0.002)] + list(reversed(inner_b)) + [(0, tb)]

    def rmod(th, k):   # vertical flutes on the outside of belly/neck, wavy lip (8 scallops)
        if 2 <= k <= n_out - 1:
            return 1.0 + 0.018 * math.cos(16 * th)
        if n_out <= k <= n_out + 2:
            return 1.0 + 0.035 * math.cos(8 * th)
        return 1.0
    add_mesh(st, "/World/visual/vase", *lathe(profile, 96, 1, 1, rmod=rmod), mat=look)
    add_convex(st, "bottom", cyl_pts(0.045, 0.0, 0.0465, tb, n=24), pm)
    shell_collision(st, pm, outer_b[1:], inner_b[1:], segments=20)
    finish(st, aid, path, mass_kg=mass, friction=[0.7, 0.6], color=color,
           collision=f"bottom + 20x{n_out - 2} convex wall pieces", mouth_outer_diameter_m=0.115,
           mouth_inner_diameter_m=round(2 * (0.0575 - t), 4), neck_inner_diameter_m=round(2 * (0.0375 - t), 4),
           max_diameter_m=0.130, height_m=H)


def gen_flower(root: Path, color_name: str, head_color, stem_len=0.25, head_d=0.09):
    """Final (PDF): crochet rose, head diameter 8-10 cm (9 cm used), overall ~30 cm.  Single rigid body,
    authored lying along +X; origin = stem bottom end on the stem axis."""
    aid = f"bright_flower_rose_{color_name}"
    path = root / "flowers" / f"{aid}.usd"
    mass = 0.03
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.8, dynamic=0.7)
    yarn_h = omnipbr(st, "yarn_head", color=head_color, texture=tex_or_none(root, YARN), tint=head_color, roughness=0.9)
    green = (0.36, 0.58, 0.22)
    yarn_s = omnipbr(st, "yarn_stem", color=green, texture=tex_or_none(root, YARN), tint=green, roughness=0.9)
    rs = 0.0035
    stem = lathe([(0, 0), (rs, 0), (rs, stem_len), (0, stem_len)], 12, 1, 40)
    stem = transform(stem, rot_y(math.pi / 2))
    add_mesh(st, "/World/visual/stem", *stem, mat=yarn_s)
    # rose head: an outer cup of 5 lobed petals + two inner rolled layers, axis along +X
    hr, hh = head_d / 2, 0.055
    x_head = stem_len - 0.004
    prof = [(0.005, 0.0), (0.016, 0.006), (0.030, 0.016), (hr, 0.032), (hr * 0.97, 0.042), (hr * 0.85, 0.048),
            (0.030, 0.050), (0.0, 0.046)]
    head = lathe(prof, 72, 8, 40, rmod=lambda th, k: 1.0 + (0.07 if k >= 2 else 0.0) * math.cos(5 * th))
    add_mesh(st, "/World/visual/head", *transform(head, rot_y(math.pi / 2), (x_head, 0, 0)), mat=yarn_h)
    for j, (r, z0, z1, lobes, a0) in enumerate(((0.030, 0.030, 0.055, 4, 0.4), (0.017, 0.040, 0.058, 3, 1.3))):
        inner = lathe([(r * 0.6, z0), (r, z0 + 0.4 * (z1 - z0)), (r * 0.9, z1), (r * 0.4, z1 - 0.002), (0.0, z1 - 0.004)],
                      48, 4, 40, angle0=a0, rmod=lambda th, k, L=lobes: 1.0 + 0.12 * math.cos(L * th))
        add_mesh(st, f"/World/visual/petals_{j}", *transform(inner, rot_y(math.pi / 2), (x_head, 0, 0)), mat=yarn_h)
    # green sepals under the head + two leaves on the stem
    # (base x, direction angle in the XY plane measured from +X, length, half width); sepals point back down the stem
    for k, (x, ang, ln, wd) in enumerate(((x_head + 0.002, math.pi - 0.7, 0.022, 0.007),
                                          (x_head + 0.002, math.pi + 0.7, 0.022, 0.007),
                                          (0.16, 0.6, 0.040, 0.010), (0.11, -0.7, 0.040, 0.010))):
        pts, uv = [], []
        n = 12
        ca, sa = math.cos(ang), math.sin(ang)
        for s in range(n):
            th = 2 * math.pi * s / n
            u, v = ln * (1 - math.cos(th)) / 2, wd * math.sin(th)
            pts.append((x + u * ca - v * sa, u * sa + v * ca, 0.0))
            uv.append((math.cos(th), math.sin(th)))
        add_mesh(st, f"/World/visual/leaf_{k}", np.array(pts), [n], list(range(n)), uv, mat=yarn_s, smooth=False)
    sp = cyl_pts(rs, 0, rs, stem_len, n=8) @ rot_y(math.pi / 2).T
    add_convex(st, "stem", sp, pm)
    hp = []
    for (r, z) in prof:
        for s in range(16):
            th = 2 * math.pi * s / 16
            hp.append((x_head + z, r * 1.04 * math.cos(th), r * 1.04 * math.sin(th)))
    add_convex(st, "head", hp, pm)
    finish(st, aid, path, mass_kg=mass, friction=[0.8, 0.7], color=head_color,
           collision="8-gon stem prism + convex hull head", texture=YARN, stem_length_m=stem_len,
           stem_diameter_m=2 * rs, head_diameter_m=2 * hr, overall_length_m=round(x_head + 0.05, 3),
           authored_axis="+X (stem base at origin, head at +X)")


def gen_tablecloth(root: Path, name: str, texture: str, tile_m: float, color=(1, 1, 1), size=(0.68, 1.10)):
    aid = f"bright_tablecloth_{name}"
    path = root / "tablecloths" / f"{aid}.usd"
    st = new_stage(path, mass=1.0, kinematic=True)
    pm = physics_material(st, static=0.6, dynamic=0.5)
    look = omnipbr(st, "cloth", color=color, texture=tex_or_none(root, texture), roughness=0.95)
    T = 0.0015
    p, c, i, _ = box_mesh((0, 0, T / 2), (size[0], size[1], T))
    uv = [((pp[0] + size[0] / 2) / tile_m, (pp[1] + size[1] / 2) / tile_m) for pp in p]
    add_mesh(st, "/World/visual/cloth", p, c, i, uv, mat=look, smooth=False)
    add_convex(st, "cloth", box_pts((0, 0, T / 2), (size[0], size[1], T)), pm)
    finish(st, aid, path, mass_kg=1.0, kinematic=True, friction=[0.6, 0.5], collision="thin box",
           texture=texture, tile_m=tile_m, note="kinematic tabletop overlay; real cloth is 90x150 cm")


def gen_towel(root: Path, color_name: str, color, size=0.30):
    aid = f"bright_towel_{color_name}"
    path = root / "towels" / f"{aid}.usd"
    mass = 0.06
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.9, dynamic=0.8)
    look = omnipbr(st, "towel", color=color, texture=tex_or_none(root, TOWEL), tint=color, roughness=1.0)
    edge_c = (0.03, 0.03, 0.035)   # black binding (all colours)
    edge = omnipbr(st, "edge", color=edge_c, roughness=1.0)
    T = 0.006
    p, c, i, _ = box_mesh((0, 0, T / 2), (size - 0.012, size - 0.012, T))
    uv = [((pp[0] + size / 2) / 0.15, (pp[1] + size / 2) / 0.15) for pp in p]
    add_mesh(st, "/World/visual/towel", p, c, i, uv, mat=look, smooth=False)
    parts = []
    for sgn in (-1, 1):
        parts.append(box_mesh((sgn * (size / 2 - 0.003), 0, T / 2), (0.006, size, T * 0.9)))
        parts.append(box_mesh((0, sgn * (size / 2 - 0.003), T / 2), (size, 0.006, T * 0.9)))
    add_mesh(st, "/World/visual/edge", *merge(*parts), mat=edge, smooth=False)
    add_convex(st, "towel", box_pts((0, 0, T / 2), (size, size, T)), pm)
    finish(st, aid, path, mass_kg=mass, friction=[0.9, 0.8], color=color,
           collision="thin box (RIGID PLACEHOLDER - cannot fold)", texture=TOWEL)


def gen_bean(root: Path):
    aid = "bright_bean_red"
    path = root / "tableware" / f"{aid}.usd"
    mass = 0.0004
    st = new_stage(path, mass=mass)
    pm = physics_material(st, static=0.5, dynamic=0.4, restitution=0.1)
    look = omnipbr(st, "bean", color=(0.45, 0.06, 0.06), roughness=0.3)
    a, b, c = 0.0042, 0.0033, 0.003  # semi-axes (8.4 x 6.6 x 6 mm, ~1.5x a real adzuki bean)
    pts, uv = [], []
    nu, nv = 12, 8
    for j in range(nv + 1):
        ph = math.pi * j / nv
        for k in range(nu + 1):
            th = 2 * math.pi * k / nu
            pts.append((a * math.sin(ph) * math.cos(th), b * math.sin(ph) * math.sin(th), c * (1 - math.cos(ph))))
            uv.append((k / nu, j / nv))
    counts, idx = [], []
    for j in range(nv):
        for k in range(nu):
            q = j * (nu + 1) + k
            counts.append(4)
            idx += [q, q + 1, q + nu + 2, q + nu + 1]
    add_mesh(st, "/World/visual/bean", np.array(pts), counts, idx, uv, mat=look)
    hull = [p for n_, p in enumerate(pts) if n_ % 3 == 0]
    add_convex(st, "bean", hull, pm)
    finish(st, aid, path, mass_kg=mass, friction=[0.5, 0.4], collision="convex hull ellipsoid",
           note="1.5x scaled adzuki bean for solver stability")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-root", default=os.environ.get("MANAENV_ASSET_ROOT", "x2real-assets"))
    args = ap.parse_args()
    root = Path(args.asset_root).resolve() / "bright"
    (root / TEX_DIR_NAME).mkdir(parents=True, exist_ok=True)
    make_procedural_textures(root / TEX_DIR_NAME)
    # drop USDs of props that were removed from the plan (stale files would stay registered)
    for stale in ("flowers/bright_vase_celadon.usd", "stacker/bright_ring_large.usd", "stacker/bright_ring_medium.usd",
                  "stacker/bright_ring_small.usd", "stacker/bright_ring_extra.usd"):   # 3+1 rings -> 5 rings (final)
        (root / stale).unlink(missing_ok=True)

    bowls = {"blue": (0.66, 0.78, 0.88), "pink": (0.94, 0.77, 0.77), "beige": (0.91, 0.86, 0.76),
             "mint": (0.72, 0.87, 0.78), "green": (0.64, 0.80, 0.70)}   # final pair: blue + green
    for n, c in bowls.items():
        gen_bowl(root, n, c)
    cups = {"green": (0.73, 0.83, 0.62), "blue": (0.63, 0.75, 0.89), "purple": (0.76, 0.71, 0.89),
            "beige": (0.89, 0.83, 0.71), "pink": (0.94, 0.71, 0.71)}
    for n, c in cups.items():
        gen_cup(root, n, c)
    gen_cup(root, "blue", (0.55, 0.75, 1.0), aid="bright_measuring_cup_blue", r_bot=0.030, r_top=0.040, H=0.095,
            mass=0.05, glass=True)
    gen_basket(root, "green", (0.80, 0.91, 0.64))
    gen_basket(root, "blue", (0.64, 0.85, 0.94))
    gen_basket(root, "pink", (0.85, 0.55, 0.55), aid="bright_bin_pink", outer_top=(0.301, 0.231),
               outer_bot=(0.27, 0.20), H=0.09, holes=False, mass=0.25)
    gen_tray(root, "yellow", (0.96, 0.82, 0.45))      # not used by any scene (no tray was bought)
    gen_tray(root, "pink", (0.95, 0.72, 0.72))
    gen_toolbox(root)
    gen_hammer(root)
    gen_wrench(root)
    gen_screwdriver(root)
    gen_screw_twister(root)
    gen_saw(root)
    gen_gear(root)
    gen_stacker(root)
    sap, bee = (0.45, 0.25, 0.18), (0.85, 0.7, 0.5)
    for k, od in enumerate(RING_ODS, start=1):        # 1 = largest (bottom); dark/light alternate as in the photo
        gen_ring(root, str(k), od / 2, SAPELE if k % 2 else BEECH, sap if k % 2 else bee)
    gen_vase(root, "white", (0.90, 0.88, 0.84))        # bisque off-white
    # the 5 final crochet roses (PDF order: deep purple, light blue, yellow, red, flesh pink)
    gen_flower(root, "purple", (0.33, 0.08, 0.42))
    gen_flower(root, "blue", (0.62, 0.76, 0.93))
    gen_flower(root, "yellow", (0.98, 0.88, 0.55))
    gen_flower(root, "red", (0.80, 0.05, 0.09))
    gen_flower(root, "pink", (0.98, 0.76, 0.70))
    gen_flower(root, "orange", (0.98, 0.55, 0.15))     # legacy, not used by any scene
    gen_lily(root)                                      # legacy, not used by any scene
    gen_tablecloth(root, "gray_grid", "tablecloth_gray_grid.png", 0.48)
    gen_tablecloth(root, "forest_deer", "tablecloth_forest_deer.png", 0.50)
    gen_tablecloth(root, "blue_cloud", "tablecloth_blue_cloud.png", 0.40)
    gen_tablecloth(root, "red_check", "tablecloth_red_check.png", 0.58)
    gen_tablecloth(root, "wood", BEECH, 0.35)          # bare-table look, not in the default pool
    towels = {"blue": (0.2, 0.45, 0.9), "pink": (0.95, 0.5, 0.7), "yellow": (0.98, 0.85, 0.25),
              "green": (0.45, 0.8, 0.35), "purple": (0.6, 0.45, 0.85)}
    for n, c in towels.items():
        gen_towel(root, n, c)
    gen_bean(root)
    (root / "asset_manifest.json").write_text(json.dumps(MANIFEST, indent=1))
    print(f"wrote {len(MANIFEST)} assets under {root}")


if __name__ == "__main__":
    main()
