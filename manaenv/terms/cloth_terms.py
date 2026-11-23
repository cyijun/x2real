"""Stage criteria for PhysX surface-deformable cloth (see :mod:`manaenv.utils.surface_cloth`).

All terms read the cloth simulation-mesh vertices (env frame) and the robot fingertip links, so
they score the actual cloth state rather than a rigid proxy.  Corner roles (near/far, left/right
relative to the robot) are recorded by the cloth reset event at the start of every episode.

Set ``MANA_CLOTH_TRACE=/path/file.jsonl`` to append a per-step trace (corner and fingertip
positions, extents, heights) for offline analysis.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from manaenv.terms.terminations import (
    _apply_consecutive_filter,
    _reference_bbox_env,
    _resolve_name,
)
from manaenv.utils.surface_cloth import cloth_points_env, cloth_state, cloth_velocities
from manaenv.utils.tensor_utils import as_torch

_TIP_BODIES = {
    "left": ("left_arm_gripper_left_tip_link", "left_arm_gripper_right_tip_link"),
    "right": ("right_arm_gripper_left_tip_link", "right_arm_gripper_right_tip_link"),
}


def _pinch_points_env(env: Any, robot_name: str = "robot") -> torch.Tensor:
    """Midpoint of the two fingertip links of each arm, env frame, ``(num_envs, 2, 3)`` [left, right]."""
    robot = env.scene[robot_name]
    key = "_mana_cloth_tip_body_ids"
    ids = getattr(env, key, None)
    if ids is None:
        ids = []
        for arm in ("left", "right"):
            idx, _ = robot.find_bodies(list(_TIP_BODIES[arm]), preserve_order=True)
            ids.append(list(idx))
        setattr(env, key, ids)
    body_pos = as_torch(robot.data.body_pos_w)
    out = torch.stack([body_pos[:, ids[k], :].mean(dim=1) for k in range(2)], dim=1)
    return out - env.scene.env_origins[:, None, :]


def _corners(env: Any, cloth: str, pts: torch.Tensor) -> torch.Tensor | None:
    st = cloth_state(env, cloth)
    roles = st.get("corners")
    if roles is None:
        return None
    return torch.gather(
        pts, 1, roles[:, :, None].expand(-1, -1, 3)
    )  # (E,4,3) [nl, nr, fl, fr]


def _trace(env: Any, cloth: str, tag: str, **values: Any) -> None:
    path = os.environ.get("MANA_CLOTH_TRACE")
    if not path:
        return
    step = (
        int(as_torch(env.episode_length_buf)[0].item())
        if hasattr(env, "episode_length_buf")
        else -1
    )
    every = int(os.environ.get("MANA_CLOTH_TRACE_EVERY", "10"))
    if step % max(1, every):
        return
    rec = {"tag": tag, "cloth": cloth, "step": step}
    for k, v in values.items():
        if isinstance(v, torch.Tensor):
            v = v[0].detach().cpu().tolist() if v.ndim else float(v.item())
        rec[k] = v
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def _min_dist_to_cloth(pinch: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Smallest fingertip-midpoint to cloth-vertex distance per arm, ``(E, 2)``."""
    return torch.cdist(pinch, pts).min(dim=-1).values


def _support_z(env: Any, cloth: str, table_z: float | None) -> torch.Tensor:
    """Height (env frame) of the surface the towel was laid on at reset, ``(E,)``.

    The reset event records the layout proxy's root z (= top of the tablecloth overlay, ~2 cm above
    the bare table).  ``table_z`` is only a fallback for states recorded before that existed.
    """
    sz = cloth_state(env, cloth).get("support_z")
    if sz is not None:
        return sz
    return torch.full((env.num_envs,), float(table_z if table_z is not None else 0.0), device=env.device)


def cloth_corners_held(
    env: Any,
    *,
    cloth: str,
    table_z: float | None = None,
    max_dist: float = 0.045,
    min_lift: float = 0.015,
    far_max_lift: float | None = None,
    consecutive_steps: int = 5,
    robot: str = "robot",
) -> torch.Tensor:
    """[SKILL grasp/lift cloth] Both near corners are each next to that arm's fingertips and raised.

    near-left corner <-> left gripper, near-right corner <-> right gripper.  A corner counts as held
    when it is within ``max_dist`` of the fingertip midpoint and at least ``min_lift`` above the
    table and its paired far corner.  ``far_max_lift`` additionally requires both far
    corners to stay on the table (it is a fold, not a pick-up).
    """
    cloth = _resolve_name(cloth)
    pts = cloth_points_env(env, cloth)
    corners = _corners(env, cloth, pts)
    if corners is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    pinch = _pinch_points_env(env, robot)
    d = torch.linalg.norm(corners[:, :2, :] - pinch, dim=-1)  # (E,2)
    lift = corners[:, :, 2] - _support_z(env, cloth, table_z)[:, None]
    # The support overlay sits above the nominal table. A stationary corner
    # should not count as lifted merely because of that fixed offset.
    relative_lift = lift[:, :2] - lift[:, 2:]
    ok = (d < max_dist).all(dim=1) & (relative_lift > min_lift).all(dim=1)
    ok &= (lift[:, :2] > min_lift).all(dim=1)
    if far_max_lift is not None:
        ok = ok & (lift[:, 2:] < far_max_lift).all(dim=1)
    _trace(
        env,
        cloth,
        f"held_{min_lift}",
        dist=d,
        lift=lift,
        relative_lift=relative_lift,
        pinch=pinch,
        corners=corners,
        ok=ok,
    )
    tag = f"_mana_cloth_held_{cloth}_{max_dist}_{min_lift}_{far_max_lift}"
    return _apply_consecutive_filter(env, ok, int(consecutive_steps), tag)


def cloth_folded_in_half(
    env: Any,
    *,
    cloth: str,
    table_z: float | None = None,
    corner_xy_tol: float = 0.05,
    short_range: tuple[float, float] = (0.10, 0.20),
    long_range: tuple[float, float] = (0.25, 0.34),
    max_height: float = 0.05,
    release_clearance: float = 0.03,
    max_speed: float = 0.05,
    consecutive_steps: int = 10,
    robot: str = "robot",
) -> torch.Tensor:
    """[SKILL fold cloth] Towel lies folded once (near edge onto far edge) and both grippers let go.

    * each near corner lies within ``corner_xy_tol`` (top view) of the far corner on the same side;
    * the footprint along the original near->far axis is in ``short_range`` (30 cm towel -> ~15 cm)
      and across it in ``long_range``;
    * every vertex is below ``table_z + max_height`` (lying on the table) and slower than ``max_speed``;
    * both fingertip midpoints are at least ``release_clearance`` away from every cloth vertex.
    """
    cloth = _resolve_name(cloth)
    pts = cloth_points_env(env, cloth)
    corners = _corners(env, cloth, pts)
    if corners is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    axis = cloth_state(env, cloth)["fold_dir"]
    normal = torch.stack((-axis[:, 1], axis[:, 0]), dim=-1)
    d_left = torch.linalg.norm(corners[:, 0, :2] - corners[:, 2, :2], dim=-1)
    d_right = torch.linalg.norm(corners[:, 1, :2] - corners[:, 3, :2], dim=-1)
    xy = pts[..., :2]
    a = (xy * axis[:, None, :]).sum(-1)
    b = (xy * normal[:, None, :]).sum(-1)
    ext_a = a.max(dim=1).values - a.min(dim=1).values
    ext_b = b.max(dim=1).values - b.min(dim=1).values
    support = _support_z(env, cloth, table_z)
    height = pts[..., 2].max(dim=1).values - support
    speed = torch.linalg.norm(cloth_velocities(env, cloth), dim=-1).max(dim=1).values
    clearance = _min_dist_to_cloth(_pinch_points_env(env, robot), pts).min(dim=1).values
    ok = (
        (d_left < corner_xy_tol)
        & (d_right < corner_xy_tol)
        & (ext_a >= short_range[0])
        & (ext_a <= short_range[1])
        & (ext_b >= long_range[0])
        & (ext_b <= long_range[1])
        & (height < max_height)
        & (pts[..., 2].min(dim=1).values >= support - 0.015)
        & (speed < max_speed)
        & (clearance > release_clearance)
    )
    _trace(
        env,
        cloth,
        "folded",
        d_left=d_left,
        d_right=d_right,
        ext_a=ext_a,
        ext_b=ext_b,
        height=height,
        speed=speed,
        clearance=clearance,
        ok=ok,
    )
    tag = f"_mana_cloth_folded_{cloth}_{corner_xy_tol}"
    return _apply_consecutive_filter(env, ok, int(consecutive_steps), tag)


def cloth_lifted_off_table(
    env: Any,
    *,
    cloth: str,
    table_z: float | None = None,
    min_height: float = 0.02,
    min_fraction: float = 0.9,
    max_hold_dist: float = 0.05,
    max_reach: float = 0.5,
    crease_grasp_tol: float | None = None,
    consecutive_steps: int = 5,
    robot: str = "robot",
) -> torch.Tensor:
    """[SKILL pick cloth] At least ``min_fraction`` of the cloth vertices are ``min_height`` above the table,
    while a gripper is on the cloth (fingertip midpoint within ``max_hold_dist`` of a vertex).

    Sanity guard: every vertex must be finite and within ``max_reach`` of the nearest fingertip
    midpoint, so a diverging cloth (vertices thrown upward by a solver blow-up) does not score.
    """
    cloth = _resolve_name(cloth)
    pts = cloth_points_env(env, cloth)
    frac = (pts[..., 2] > _support_z(env, cloth, table_z)[:, None] + min_height).float().mean(dim=1)
    finite = torch.isfinite(pts).all(dim=-1).all(dim=-1)
    safe = torch.nan_to_num(pts, nan=1e3, posinf=1e3, neginf=-1e3)
    dist = torch.cdist(_pinch_points_env(env, robot), safe)  # (E, 2, V)
    held = dist.min(dim=-1).values.min(dim=-1).values < max_hold_dist
    reach = dist.min(dim=1).values.max(dim=-1).values
    ok = (frac >= min_fraction) & finite & held & (reach < max_reach)
    extra = {}
    if crease_grasp_tol is not None:
        crease = cloth_state(env, cloth).get("crease")
        if crease is None:
            ok = torch.zeros_like(ok)
        else:
            mid = torch.gather(safe, 1, crease[:, :1, None].expand(-1, -1, 3))  # (E,1,3)
            d_mid = torch.linalg.norm(_pinch_points_env(env, robot) - mid, dim=-1).min(dim=1).values
            ok &= d_mid < crease_grasp_tol
            extra["crease_mid_dist"] = d_mid
            extra["crease_mid"] = mid[:, 0]
            extra["pinch"] = _pinch_points_env(env, robot)
    _trace(env, cloth, "lifted", frac=frac, held=held, reach=reach, ok=ok, **extra)
    return _apply_consecutive_filter(
        env, ok, int(consecutive_steps), f"_mana_cloth_lifted_{cloth}_{min_height}"
    )


def cloth_in_container(
    env: Any,
    *,
    cloth: str,
    container_cfg: Any,
    min_fraction: float = 0.85,
    xy_inset: float = 0.0,
    z_margin: float = 0.03,
    release_clearance: float = 0.03,
    max_speed: float = 0.1,
    consecutive_steps: int = 10,
    robot: str = "robot",
) -> torch.Tensor:
    """[SKILL place cloth] ``min_fraction`` of the vertices are inside the container's live footprint
    box (xy shrunk by ``xy_inset``; z from the container bottom to ``z_margin`` above its rim), the
    cloth has settled and neither gripper is touching it."""
    cloth = _resolve_name(cloth)
    container = _resolve_name(container_cfg)
    pts = cloth_points_env(env, cloth)
    frac = torch.zeros(env.num_envs, device=env.device)
    for env_id in range(env.num_envs):
        bbox = _reference_bbox_env(env, container, env_id)
        if bbox is None:
            continue
        bmin, bmax, _ = bbox
        p = pts[env_id]
        inside = (
            (p[:, 0] > bmin[0] + xy_inset)
            & (p[:, 0] < bmax[0] - xy_inset)
            & (p[:, 1] > bmin[1] + xy_inset)
            & (p[:, 1] < bmax[1] - xy_inset)
            & (p[:, 2] > bmin[2] - 0.01)
            & (p[:, 2] < bmax[2] + z_margin)
        )
        frac[env_id] = inside.float().mean()
    speed = torch.linalg.norm(cloth_velocities(env, cloth), dim=-1).max(dim=1).values
    clearance = _min_dist_to_cloth(_pinch_points_env(env, robot), pts).min(dim=1).values
    ok = (frac >= min_fraction) & (speed < max_speed) & (clearance > release_clearance)
    _trace(
        env, cloth, "in_container", frac=frac, speed=speed, clearance=clearance, ok=ok
    )
    return _apply_consecutive_filter(
        env, ok, int(consecutive_steps), f"_mana_cloth_in_{cloth}_{container}"
    )


__all__ = [
    "cloth_corners_reached",
    "cloth_fold_carried",
    "cloth_over_container",
    "cloth_placed_folded",
    "cloth_corners_held",
    "cloth_folded_in_half",
    "cloth_lifted_off_table",
    "cloth_in_container",
]


# ---------------------------------------------------------------------------
# Stage criteria of the final BRIGHT fold-towel demo (6 steps, 2026-11-22 submission):
# M3 align -> D3 fold -> P2 release -> G1 pick fold-line middle -> M1 over basket -> P2 place flat
# ---------------------------------------------------------------------------


def cloth_corners_reached(
    env: Any,
    *,
    cloth: str,
    max_dist: float = 0.05,
    consecutive_steps: int = 3,
    robot: str = "robot",
) -> torch.Tensor:
    """[M3 align] Each arm's fingertip midpoint is within ``max_dist`` of its near corner
    (near-left <-> left arm, near-right <-> right arm)."""
    cloth = _resolve_name(cloth)
    pts = cloth_points_env(env, cloth)
    corners = _corners(env, cloth, pts)
    if corners is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    d = torch.linalg.norm(corners[:, :2, :] - _pinch_points_env(env, robot), dim=-1)
    ok = (d < max_dist).all(dim=1)
    _trace(env, cloth, "reached", dist=d, ok=ok)
    return _apply_consecutive_filter(env, ok, int(consecutive_steps), f"_mana_cloth_reached_{cloth}_{max_dist}")


def cloth_fold_carried(
    env: Any,
    *,
    cloth: str,
    table_z: float | None = None,
    max_hold_dist: float = 0.045,
    align_tol: float = 0.03,
    far_max_lift: float = 0.04,
    consecutive_steps: int = 5,
    robot: str = "robot",
) -> torch.Tensor:
    """[D3 fold] Both near corners are still pinched and have been carried over their far corners:
    near/far corner xy offset <= ``align_tol`` on both sides, far corners still on the table."""
    cloth = _resolve_name(cloth)
    pts = cloth_points_env(env, cloth)
    corners = _corners(env, cloth, pts)
    if corners is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    held = torch.linalg.norm(corners[:, :2, :] - _pinch_points_env(env, robot), dim=-1)
    align = torch.linalg.norm(corners[:, :2, :2] - corners[:, 2:, :2], dim=-1)
    far_lift = corners[:, 2:, 2] - _support_z(env, cloth, table_z)[:, None]
    ok = (held < max_hold_dist).all(dim=1) & (align < align_tol).all(dim=1) & (far_lift < far_max_lift).all(dim=1)
    _trace(env, cloth, "fold_carried", held=held, align=align, far_lift=far_lift, ok=ok)
    return _apply_consecutive_filter(env, ok, int(consecutive_steps), f"_mana_cloth_carried_{cloth}_{align_tol}")


def _container_box(env: Any, container: str, env_id: int):
    bbox = _reference_bbox_env(env, container, env_id)
    if bbox is None:
        return None
    return bbox[0], bbox[1]


def cloth_over_container(
    env: Any,
    *,
    cloth: str,
    container_cfg: Any,
    min_fraction: float = 0.98,
    max_hold_dist: float = 0.05,
    consecutive_steps: int = 5,
    robot: str = "robot",
) -> torch.Tensor:
    """[M1 move] The towel is held and its xy projection lies inside the container mouth
    (live bounding box; ``min_fraction`` of the vertices)."""
    cloth = _resolve_name(cloth)
    container = _resolve_name(container_cfg)
    pts = cloth_points_env(env, cloth)
    finite = torch.isfinite(pts).all(dim=-1).all(dim=-1)
    safe = torch.nan_to_num(pts, nan=1e3, posinf=1e3, neginf=-1e3)
    held = torch.cdist(_pinch_points_env(env, robot), safe).min(dim=-1).values.min(dim=-1).values < max_hold_dist
    frac = torch.zeros(env.num_envs, device=env.device)
    for env_id in range(env.num_envs):
        box = _container_box(env, container, env_id)
        if box is None:
            continue
        bmin, bmax = box
        p = safe[env_id]
        inside = (p[:, 0] > bmin[0]) & (p[:, 0] < bmax[0]) & (p[:, 1] > bmin[1]) & (p[:, 1] < bmax[1])
        frac[env_id] = inside.float().mean()
    ok = finite & held & (frac >= min_fraction)
    _trace(env, cloth, "over_container", frac=frac, held=held, ok=ok)
    return _apply_consecutive_filter(env, ok, int(consecutive_steps), f"_mana_cloth_over_{cloth}_{container}")


def cloth_placed_folded(
    env: Any,
    *,
    cloth: str,
    container_cfg: Any,
    min_fraction: float = 0.95,
    release_max_height: float = 0.03,
    pair_tol: float = 0.05,
    min_length: float = 0.25,
    max_stack_height: float = 0.05,
    z_margin: float = 0.03,
    release_clearance: float = 0.03,
    max_speed: float = 0.1,
    consecutive_steps: int = 10,
    robot: str = "robot",
) -> torch.Tensor:
    """[P2 place] Released near the container floor and lying there still folded.

    * at the last gripper release the lowest towel vertex was <= ``release_max_height`` above the
      container bottom (live bounding-box min z);
    * ``min_fraction`` of the vertices are inside the container box (z up to ``z_margin`` above the rim);
    * still a half fold: each near corner within ``pair_tol`` (3D) of its far corner, the fold line and
      the open edge each spread to >= ``min_length`` (not bunched), whole towel <= ``max_stack_height``
      above the container bottom (lying flat);
    * settled (max vertex speed < ``max_speed``) and no fingertip within ``release_clearance``.
    """
    cloth = _resolve_name(cloth)
    container = _resolve_name(container_cfg)
    st = cloth_state(env, cloth)
    pts = cloth_points_env(env, cloth)
    corners = _corners(env, cloth, pts)
    crease = st.get("crease")
    if corners is None or crease is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    ok = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    vals = {k: torch.full((env.num_envs,), float("nan"), device=env.device)
            for k in ("frac", "release_h", "pair", "crease_len", "open_len", "stack_h")}
    for e in range(env.num_envs):
        box = _container_box(env, container, e)
        rel = st.get("last_release", {}).get(e)
        if box is None or rel is None:
            continue
        bmin, bmax = box
        p = pts[e]
        inside = ((p[:, 0] > bmin[0]) & (p[:, 0] < bmax[0]) & (p[:, 1] > bmin[1]) & (p[:, 1] < bmax[1])
                  & (p[:, 2] > bmin[2] - 0.01) & (p[:, 2] < bmax[2] + z_margin))
        c = corners[e]
        ends = p[crease[e, 1:]]
        vals["frac"][e] = inside.float().mean()
        vals["release_h"][e] = rel["zmin"] - float(bmin[2])
        vals["pair"][e] = torch.maximum(torch.linalg.norm(c[0] - c[2]), torch.linalg.norm(c[1] - c[3]))
        vals["crease_len"][e] = torch.linalg.norm(ends[0] - ends[1])
        vals["open_len"][e] = torch.linalg.norm(c[2] - c[3])
        vals["stack_h"][e] = p[:, 2].max() - bmin[2]
    speed = torch.linalg.norm(cloth_velocities(env, cloth), dim=-1).max(dim=1).values
    clearance = _min_dist_to_cloth(_pinch_points_env(env, robot), pts).min(dim=1).values
    ok = ((vals["frac"] >= min_fraction) & (vals["release_h"] <= release_max_height) & (vals["pair"] <= pair_tol)
          & (vals["crease_len"] >= min_length) & (vals["open_len"] >= min_length)
          & (vals["stack_h"] <= max_stack_height) & (speed < max_speed) & (clearance > release_clearance))
    _trace(env, cloth, "placed", speed=speed, clearance=clearance, ok=ok, **vals)
    return _apply_consecutive_filter(env, ok, int(consecutive_steps), f"_mana_cloth_placed_{cloth}_{container}")
