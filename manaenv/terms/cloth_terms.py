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


def cloth_corners_held(
    env: Any,
    *,
    cloth: str,
    table_z: float,
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
    lift = corners[:, :, 2] - float(table_z)
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
    table_z: float,
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
    height = pts[..., 2].max(dim=1).values - float(table_z)
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
        & (pts[..., 2].min(dim=1).values >= table_z - 0.015)
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
    table_z: float,
    min_height: float = 0.02,
    min_fraction: float = 0.9,
    consecutive_steps: int = 5,
) -> torch.Tensor:
    """[SKILL pick cloth] At least ``min_fraction`` of the cloth vertices are ``min_height`` above the table."""
    cloth = _resolve_name(cloth)
    pts = cloth_points_env(env, cloth)
    frac = (pts[..., 2] > float(table_z) + min_height).float().mean(dim=1)
    ok = frac >= min_fraction
    _trace(env, cloth, "lifted", frac=frac, ok=ok)
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
    "cloth_corners_held",
    "cloth_folded_in_half",
    "cloth_lifted_off_table",
    "cloth_in_container",
]
