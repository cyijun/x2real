#!/usr/bin/env python
"""Generate the coral-fleece cleaning-cloth asset (visual variant of bright_towel_cloth).

    python tools/bright/gen_fleece_cloth.py            # -> assets/bright_fleece/{towels,textures}

Same 25 x 25 simulation grid as ``bright_towel_cloth`` (the surface-deformable profile
``bright_fleece_30`` in manaenv/utils/surface_cloth.py sets the physics).  Looks: white fleece
albedo (tinted per episode by the reset event, material name ``towel``), procedural pile normal
map and roughness map, 8 mm black binding baked into all three maps.  The hanging loop of the real
cloth and the different colour on the back side are not modelled (see sim-assets/FLEECE_CLOTH.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from pxr import Sdf, Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_bright_assets as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "assets" / "bright_fleece"
SIZE, BORDER, PX = 0.30, 0.008, 1024


def _noise(rng, px, scale):
    small = rng.random((max(2, px // scale), max(2, px // scale))).astype(np.float32)
    img = Image.fromarray((small * 255).astype(np.uint8)).resize((px, px), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def make_textures(tex: Path):
    tex.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1753)
    # pile height: tufts (~3 mm) + fibre grain (~0.5 mm) + soft clumps (~2 cm)
    h = 0.55 * _noise(rng, PX, 10) + 0.30 * _noise(rng, PX, 2) + 0.15 * _noise(rng, PX, 70)
    h = np.asarray(Image.fromarray((h * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)),
                   dtype=np.float32) / 255.0
    w = max(2, int(round(PX * BORDER / SIZE)))
    border = np.zeros((PX, PX), bool)
    border[:w, :] = border[-w:, :] = border[:, :w] = border[:, -w:] = True
    # albedo: near-white fleece with darker pile valleys; binding near-black with a woven stripe
    alb = 0.80 + 0.20 * h
    rgb = np.stack([alb] * 3, -1)
    stripe = 0.035 + 0.02 * (np.indices((PX, PX)).sum(0) % 6 < 3)
    rgb[border] = np.stack([stripe] * 3, -1)[border]
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(tex / "fleece_albedo.png")
    # tangent-space normal map from the pile height (binding flatter)
    hb = np.where(border, 0.5 + 0.1 * (h - 0.5), h)
    gy, gx = np.gradient(hb * 6.0)
    n = np.stack([-gx, -gy, np.ones_like(hb)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8)).save(tex / "fleece_normal.png")
    # roughness: fleece fully matte, binding slightly smoother
    r = np.where(border, 0.70, 0.92 + 0.08 * h)
    Image.fromarray((np.clip(r, 0, 1) * 255).astype(np.uint8)).save(tex / "fleece_roughness.png")


def make_usd(path: Path, res: int = A.CLOTH_RES):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    st.SetMetadata("kilogramsPerUnit", 1.0)
    world = UsdGeom.Xform.Define(st, "/World")
    st.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(st, "/World/Looks")
    look = A.omnipbr(st, "towel", color=(0.2, 0.45, 0.9), texture="fleece_albedo.png",
                     tint=(0.2, 0.45, 0.9), roughness=0.95)
    sh = st.GetPrimAtPath("/World/Looks/towel/Shader")
    from pxr import UsdShade
    shader = UsdShade.Shader(sh)
    shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("../textures/fleece_normal.png"))
    shader.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("reflectionroughness_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("../textures/fleece_roughness.png"))
    shader.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(1.0)
    n = int(res)
    pts, uv = [], []
    for j in range(n):
        for i in range(n):
            pts.append((SIZE * (i / (n - 1) - 0.5), SIZE * (j / (n - 1) - 0.5), 0.0))
            uv.append((i / (n - 1), j / (n - 1)))
    counts, idx = [], []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b, c, d = a + 1, a + n + 1, a + n
            idx += [a, b, c, a, c, d] if (i + j) % 2 == 0 else [a, b, d, b, c, d]
            counts += [3, 3]
    A.add_mesh(st, "/World/towel", pts, counts, idx, uv, mat=look, smooth=False)  # no authored normals: the mesh deforms
    st.GetRootLayer().Save()


def main():
    make_textures(ROOT / "textures")
    make_usd(ROOT / "towels" / "bright_fleece_cloth.usd")
    print("wrote", ROOT)


if __name__ == "__main__":
    main()
