"""BRIGHT 2026 generalization-case events: one-shot disturbances, camera occluders, spare objects.

Every function here is an IsaacLab ``mode: interval`` event term meant to be called often
(``interval_range_s: [0.1, 0.1]``, ``is_global_time: false``).  Each term keeps per-env state under
``key``, fires at most once per episode, and re-arms automatically when a new episode starts
(detected from ``episode_length_buf`` going backwards).

Trigger (exactly one of):
  ``at_s``   -- fire once the episode time (``episode_length_buf * step_dt``) reaches ``at_s`` seconds;
  ``stage``  -- fire ``delay_s`` seconds after the Task-DAG stage with that node id is first latched
               (rising edge of ``env.extras["dag_stage_done"]``, labels in ``"dag_stage_done_labels"``).

Validation only: if the environment variable ``BRIGHT_FORCE_STAGE_EVENTS_AT_S`` is set, stage-triggered
events fire at that episode time instead (zero-action smoke runs never latch a stage).  Evaluation runs
must not set it.  Nothing here is exposed to the policy: there is no observation term, flag or log entry
in the policy's inputs -- occluders are real (rendered) objects in front of the head camera.

Every firing prints one ``BRIGHT_EVENT`` line (manaprint) so smoke runs can check that it happened.
"""
from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch

from manaenv.terms.events import _sync_pose_to_usd, _write_root_velocity_if_dynamic
from manaenv.utils.debug_print import manaprint

_FORCE_ENV = "BRIGHT_FORCE_STAGE_EVENTS_AT_S"
TABLE_CENTER_XY = (0.205, -0.273)          # env frame, green booth tabletop centre


# ---------------------------------------------------------------------------------------------------
# state / trigger helpers
# ---------------------------------------------------------------------------------------------------

def _episode_time(env) -> torch.Tensor:
    return env.episode_length_buf.to(torch.float32) * float(env.step_dt)


def _state(env, key: str) -> dict:
    store = env.__dict__.setdefault("_bright_case_events", {})
    n = env.num_envs
    st = store.get(key)
    if st is None or st["fired"].shape[0] != n:
        dev = env.device
        st = {
            "fired": torch.zeros(n, dtype=torch.bool, device=dev),        # main action done
            "ended": torch.zeros(n, dtype=torch.bool, device=dev),        # (occluder) moved back out
            "t_fire": torch.full((n,), float("nan"), device=dev),
            "stage_seen": torch.zeros(n, dtype=torch.bool, device=dev),
            "t_stage": torch.full((n,), float("nan"), device=dev),
            "t_last": torch.full((n,), -1.0, device=dev),
            "home": torch.full((n, 7), float("nan"), device=dev),        # pose at the first call of the episode
        }
        store[key] = st
    return st


def _rearm(env, st: dict) -> torch.Tensor:
    """Reset per-env state when a new episode started; returns current episode time."""
    t = _episode_time(env)
    new_ep = t < st["t_last"] - 1e-6
    if bool(new_ep.any()):
        st["fired"][new_ep] = False
        st["ended"][new_ep] = False
        st["t_fire"][new_ep] = float("nan")
        st["stage_seen"][new_ep] = False
        st["t_stage"][new_ep] = float("nan")
        st["home"][new_ep] = float("nan")
    st["t_last"] = t.clone()
    return t


def _stage_done(env, stage: str) -> torch.Tensor | None:
    extras = getattr(env, "extras", None) or {}
    labels = extras.get("dag_stage_done_labels")
    done = extras.get("dag_stage_done")
    if labels is None or done is None or stage not in list(labels):
        return None
    return torch.as_tensor(done, device=env.device)[:, list(labels).index(stage)].to(torch.bool)


def _due(env, st: dict, t: torch.Tensor, at_s: float | None, stage: str | None, delay_s: float) -> torch.Tensor:
    force = os.environ.get(_FORCE_ENV)
    if at_s is not None:
        return t >= float(at_s)
    if stage is None:
        raise ValueError("bright event needs at_s or stage")
    if force:
        return t >= float(force)
    done = _stage_done(env, stage)
    if done is None:
        return torch.zeros_like(st["fired"])
    newly = done & ~st["stage_seen"]
    st["t_stage"][newly] = t[newly]
    st["stage_seen"] |= done
    return st["stage_seen"] & (t >= st["t_stage"] + float(delay_s))


# ---------------------------------------------------------------------------------------------------
# pose helpers (env-local frame; quaternions only through isaaclab.utils.math, so the convention follows
# the installed IsaacLab)
# ---------------------------------------------------------------------------------------------------

def _asset(env, name: str):
    try:
        from manaenv.terms.terminations import _resolve_live_scene_entity_name

        live = _resolve_live_scene_entity_name(env, name, env_id=0) or name
    except Exception:  # noqa: BLE001
        live = name
    return env.scene[live], live


def _pose_local(env, asset, env_ids: torch.Tensor) -> torch.Tensor:
    pos = asset.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    return torch.cat([pos, asset.data.root_quat_w[env_ids]], dim=-1).clone()


def _write_pose(env, asset, env_ids: torch.Tensor, pose_local: torch.Tensor) -> None:
    pose = pose_local.clone()
    _sync_pose_to_usd(asset, env_ids, pose.clone())
    pose[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    _write_root_velocity_if_dynamic(asset, torch.zeros(len(env_ids), 6, device=env.device), env_ids)


def _yaw_quat(yaw: float, n: int, device) -> torch.Tensor:
    from isaaclab.utils import math as mu

    z = torch.zeros(n, device=device)
    return mu.quat_from_euler_xyz(z, z, torch.full((n,), float(yaw), device=device))


def _log(key: str, what: str, env_ids: torch.Tensor, t: torch.Tensor) -> None:
    ids = [int(i) for i in env_ids.tolist()]
    manaprint(f"BRIGHT_EVENT key={key} {what} env_ids={ids} t={[round(float(t[i]), 2) for i in ids]}")


def _select(env_ids, mask: torch.Tensor) -> torch.Tensor:
    ids = torch.as_tensor(env_ids, device=mask.device, dtype=torch.long)
    return ids[mask[ids]]


# ---------------------------------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------------------------------

def bright_displace(env, env_ids, asset_cfg: str, key: str, at_s: float | None = None, stage: str | None = None,
                    delay_s: float = 0.0, dxy: Sequence[float] = (0.0, 0.0), dz: float = 0.0, dyaw: float = 0.0,
                    away_from_center: bool = False, to: str = "relative", lift: float = 0.004) -> None:
    """One-shot teleport of an object ("the referee moves it").

    to="relative": current pose + (dxy, dz) and + dyaw; with ``away_from_center`` the y component of
                   ``dxy`` points away from the table centre line (so mirrored layouts get mirrored pushes).
    to="home":     back to the pose the object had at the start of the episode (+ dxy), e.g. a ring that was
                   already put on the peg is taken off and laid back where it started.
    to="mirror":   mirror the current position across the table centre line (y -> 2*cy - y), e.g. the basket
                   moves to the other side of the table.
    """
    if env_ids is None:
        return
    st = _state(env, key)
    t = _rearm(env, st)
    asset, _ = _asset(env, asset_cfg)
    ids_all = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    unset = torch.isnan(st["home"][ids_all, 0])
    if bool(unset.any()):
        st["home"][ids_all[unset]] = _pose_local(env, asset, ids_all[unset])
    due = _due(env, st, t, at_s, stage, delay_s) & ~st["fired"]
    ids = _select(env_ids, due)
    if ids.numel() == 0:
        return
    from isaaclab.utils import math as mu

    cur = _pose_local(env, asset, ids)
    new = cur.clone()
    dx, dy = float(dxy[0]), float(dxy[1])
    if to == "home":
        new = st["home"][ids].clone()
        new[:, 0] += dx
        new[:, 1] += dy
        new[:, 2] += lift
    elif to == "mirror":
        new[:, 1] = 2.0 * TABLE_CENTER_XY[1] - cur[:, 1]
        new[:, 2] += lift
    else:
        sign = torch.sign(cur[:, 1] - TABLE_CENTER_XY[1]) if away_from_center else torch.ones(len(ids), device=env.device)
        sign[sign == 0] = 1.0
        new[:, 0] += dx
        new[:, 1] += dy * sign
        new[:, 2] += dz + lift
    if abs(float(dyaw)) > 0:
        new[:, 3:7] = mu.quat_mul(_yaw_quat(dyaw, len(ids), env.device), new[:, 3:7])
    _write_pose(env, asset, ids, new)
    st["fired"][ids] = True
    st["t_fire"][ids] = t[ids]
    _log(key, f"displace {asset_cfg} to={to}", ids, t)


def bright_spare_object(env, env_ids, asset_cfg: str, key: str, park_xyz: Sequence[float],
                        at_s: float | None = None, stage: str | None = None, delay_s: float = 0.0,
                        place_xy: Sequence[float] | None = None, near: str | None = None,
                        near_offset: Sequence[float] = (0.0, 0.08), place_z: float = -0.20,
                        place_yaw: float = 0.0, yaw_toward_center: bool = False) -> None:
    """A spare object waits out of view (re-parked every call so it never falls away), then the "referee"
    puts it on the table: at ``place_xy`` (env frame) or next to object ``near`` (+ ``near_offset``, the y part
    pointing away from the table centre line).  ``place_z`` is the release height (env frame)."""
    if env_ids is None:
        return
    st = _state(env, key)
    t = _rearm(env, st)
    asset, _ = _asset(env, asset_cfg)
    ids_all = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    waiting = ids_all[~st["fired"][ids_all]]
    if waiting.numel() == 0:
        return
    due = _due(env, st, t, at_s, stage, delay_s)
    go = waiting[due[waiting]]
    park = waiting[~due[waiting]]
    if park.numel():
        pose = torch.zeros(park.numel(), 7, device=env.device)
        pose[:, :3] = torch.tensor(list(park_xyz), device=env.device, dtype=torch.float32)
        pose[:, 3:7] = _yaw_quat(0.0, park.numel(), env.device)
        _write_pose(env, asset, park, pose)
    if go.numel():
        pose = torch.zeros(go.numel(), 7, device=env.device)
        if near is not None:
            ref, _ = _asset(env, near)
            rp = _pose_local(env, ref, go)
            sign = torch.sign(rp[:, 1] - TABLE_CENTER_XY[1])
            sign[sign == 0] = 1.0
            pose[:, 0] = rp[:, 0] + float(near_offset[0])
            pose[:, 1] = rp[:, 1] + float(near_offset[1]) * sign
        else:
            pose[:, 0] = float(place_xy[0])
            pose[:, 1] = float(place_xy[1])
        pose[:, 2] = float(place_z)
        pose[:, 3:7] = _yaw_quat(place_yaw, go.numel(), env.device)
        if yaw_toward_center:   # a long object (authored along +x) pointing at the table centre line
            for j in range(go.numel()):
                yaw = math.pi / 2 if float(pose[j, 1]) < TABLE_CENTER_XY[1] else -math.pi / 2
                pose[j:j + 1, 3:7] = _yaw_quat(yaw, 1, env.device)
        _write_pose(env, asset, go, pose)
        st["fired"][go] = True
        st["t_fire"][go] = t[go]
        _log(key, f"place spare {asset_cfg}", go, t)


def bright_occluder(env, env_ids, asset_cfg: str, key: str, duration_s: float, park_xyz: Sequence[float],
                    at_s: float | None = None, stage: str | None = None, delay_s: float = 0.0,
                    distance: float = 0.15, lateral: float = 0.0, camera: str = "head_camera") -> None:
    """Put a (kinematic, collision-free) occluder ``distance`` m in front of the head camera for
    ``duration_s`` seconds, aligned with the camera's optical axis, then park it again."""
    if env_ids is None:
        return
    st = _state(env, key)
    t = _rearm(env, st)
    asset, _ = _asset(env, asset_cfg)
    ids_all = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    due = _due(env, st, t, at_s, stage, delay_s)
    start = ids_all[due[ids_all] & ~st["fired"][ids_all]]
    if start.numel():
        st["fired"][start] = True
        st["t_fire"][start] = t[start]
    active = st["fired"] & ~st["ended"]
    stop = ids_all[active[ids_all] & (t[ids_all] >= st["t_fire"][ids_all] + float(duration_s))]
    show = ids_all[active[ids_all] & (t[ids_all] < st["t_fire"][ids_all] + float(duration_s))]
    if show.numel():
        from isaaclab.utils import math as mu

        cam = env.scene.sensors[camera] if camera in env.scene.sensors.keys() else None
        if cam is not None:
            cpos = cam.data.pos_w[show] - env.scene.env_origins[show]
            cq = cam.data.quat_w_world[show]
            fwd = mu.quat_apply(cq, torch.tensor([[1.0, 0.0, 0.0]], device=env.device).repeat(show.numel(), 1))
            left = mu.quat_apply(cq, torch.tensor([[0.0, 1.0, 0.0]], device=env.device).repeat(show.numel(), 1))
            pose = torch.cat([cpos + float(distance) * fwd + float(lateral) * left, cq], dim=-1)
        else:  # fallback: fixed offset from the robot root (head camera at +0.084, 0, +0.517)
            rb = env.scene["robot"].data.root_pos_w[show] - env.scene.env_origins[show]
            pose = torch.zeros(show.numel(), 7, device=env.device)
            pose[:, :3] = rb + torch.tensor([0.084 + float(distance) * 0.64, float(lateral), 0.517 - float(distance) * 0.77],
                                            device=env.device)
            pose[:, 3:7] = _yaw_quat(0.0, show.numel(), env.device)
        _write_pose(env, asset, show, pose)
        if start.numel():
            _log(key, f"occluder {asset_cfg} on for {duration_s}s", start, t)
    if stop.numel():
        st["ended"][stop] = True
        _log(key, f"occluder {asset_cfg} off", stop, t)
    park = ids_all[~(st["fired"][ids_all] & ~st["ended"][ids_all])]
    if park.numel():
        pose = torch.zeros(park.numel(), 7, device=env.device)
        pose[:, :3] = torch.tensor(list(park_xyz), device=env.device, dtype=torch.float32)
        pose[:, 3:7] = _yaw_quat(0.0, park.numel(), env.device)
        _write_pose(env, asset, park, pose)


def bright_log_poses(env, env_ids, asset_cfgs: Sequence[str], reference: str, key: str, at_s: float) -> None:
    """Validation log only: once per episode at ``at_s``, print each object's position relative to ``reference``
    (dx, dy, dz in metres, env frame) and its tilt from vertical, e.g. to check that pre-stacked rings stay put."""
    if env_ids is None:
        return
    st = _state(env, key)
    t = _rearm(env, st)
    ids = _select(env_ids, (t >= float(at_s)) & ~st["fired"])
    if ids.numel() == 0:
        return
    from isaaclab.utils import math as mu

    ref, _ = _asset(env, reference)
    rp = _pose_local(env, ref, ids)
    parts = []
    for name in asset_cfgs:
        a, _ = _asset(env, name)
        p = _pose_local(env, a, ids)
        up = mu.quat_apply(p[:, 3:7], torch.tensor([[0.0, 0.0, 1.0]], device=env.device).repeat(ids.numel(), 1))
        d = p[0, :3] - rp[0, :3]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up[0, 2])))))
        parts.append(f"{name}:dxy={math.hypot(float(d[0]), float(d[1])) * 1000:.1f}mm,dz={float(d[2]) * 1000:.1f}mm,"
                     f"tilt={tilt:.1f}deg")
    st["fired"][ids] = True
    manaprint(f"BRIGHT_EVENT key={key} poses rel {reference} t={round(float(t[ids[0]]), 2)} " + " ".join(parts))
