"""PhysX surface-deformable cloth for scene-DSL environments (GPU PhysX only).

The vNext scene DSL and the runtime layout sampler only know rigid objects, so a cloth is
declared through a *layout proxy*: an invisible, collision-free, kinematic plate object whose
``semantic_tags`` contain ``surface_cloth_proxy`` and ``cloth_profile=<name>``.  The layout
solver places the proxy like any other object (footprint, support surface, collisions with other
objects, per-episode resampling).  At assembly time :func:`attach_surface_cloths` adds

* an IsaacLab ``DeformableObject`` (PhysX *surface deformable*, i.e. FEM cloth) named
  ``<proxy>_cloth`` to the env scene, spawned from the profile's textured triangle-grid USD with
  the profile's deformable body / material properties;
* a ``reset`` event (appended after the layout events) that lays the cloth flat on the proxy's
  sampled pose, zeroes its velocities, records which grid corners are near/far and left/right of
  the robot, and re-tints the towel texture;
* a ``startup`` event that raises the friction of the gripper finger shapes (PhysX deformables
  only use dynamic friction, see the Omni Physics deformable docs).

Termination terms that read the cloth state live in :mod:`manaenv.terms.cloth_terms`.
Surface deformables only run with GPU dynamics, so the env must use a CUDA PhysX runtime
(``runtime/solver/physx_gpu_sync.yaml``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch

from manaenv.utils.debug_print import manaprint
from manaenv.utils.tensor_utils import as_torch

CLOTH_PROXY_TAG = "surface_cloth_proxy"
PROFILE_TAG_PREFIX = "cloth_profile="
CLOTH_SUFFIX = "_cloth"
STATE_ATTR = "_mana_surface_cloth_state"

GRIPPER_FINGER_BODIES = [
    f"{arm}_arm_gripper_{side}_link"
    for arm in ("left", "right")
    for side in ("left", "right")
]

# Physical profiles.  ``bright_towel_30``: 30 x 30 cm double-layer microfiber towel, ~70 g,
# ~5 mm thick, soft in bending, nearly inextensible.
PROFILES: dict[str, dict[str, Any]] = {
    "bright_towel_30": {
        "usd_asset": "bright_towel_cloth",
        "grid": 25,  # vertices per side of the USD grid (vertex = j * grid + i)
        "mass": 0.07,
        "material": {
            "static_friction": 1.0,
            "dynamic_friction": 1.0,
            "youngs_modulus": 2.0e4,  # stable + settles at 240 Hz (probe sweep); 5e4 jitters, 5e3 tangles
            "poissons_ratio": 0.3,
            "surface_thickness": 0.005,
            "surface_stretch_stiffness": 1.0,
            "surface_shear_stiffness": 0.5,
            "surface_bend_stiffness": 0.2,
            "elasticity_damping": 0.03,
            "bend_damping": 0.05,
        },
        "body": {
            "solver_position_iteration_count": 32,
            "self_collision": True,
            "linear_damping": 0.2,
            "max_depenetration_velocity": 1.0,
        },
        "rest_offset": 0.0025,
        "contact_offset": 0.006,
        "lift_z": 0.004,  # cloth mid-plane above the proxy root (proxy origin = towel bottom)
        "gripper_friction": 3.0,
        # gripper/cloth coupling: "point_substep" (fingertip point attachment, every physics sub-step;
        # default), "point" (same, once per control step, legacy), "friction" (contact only).
        # Override with MANA_CLOTH_GRASP.
        "grasp": "point_substep",
        "grasp_write": "state",
        # towel colours bought for the event (tint of the bordered microfiber texture)
        "tints": {
            "blue": (0.2, 0.45, 0.9),
            "pink": (0.95, 0.5, 0.7),
            "yellow": (0.98, 0.85, 0.25),
        },
    },
}


def _iter_tags(asset: Any) -> list[str]:
    tags: list[str] = []
    for source in (
        getattr(asset, "semantic_tags", None),
        getattr(asset, "tags", None),
        getattr(
            getattr(getattr(asset, "object_cfg", None), "spawn", None),
            "semantic_tags",
            None,
        ),
    ):
        if not source:
            continue
        for entry in source:
            if isinstance(entry, str):
                tags.append(entry)
            elif isinstance(entry, (tuple, list)) and entry:
                tags.append(str(entry[-1]))
    return tags


def find_cloth_proxies(scene: Any) -> list[tuple[str, str]]:
    """Return ``[(proxy_entity_name, profile_name)]`` for scene assets tagged as cloth proxies."""
    assets = getattr(scene, "assets", None)
    if not isinstance(assets, Mapping):
        return []
    found = []
    for name, asset in assets.items():
        tags = _iter_tags(asset)
        if CLOTH_PROXY_TAG not in tags:
            continue
        profile = next(
            (
                t[len(PROFILE_TAG_PREFIX) :]
                for t in tags
                if t.startswith(PROFILE_TAG_PREFIX)
            ),
            None,
        )
        if profile not in PROFILES:
            raise ValueError(
                f"cloth proxy {name!r}: unknown or missing {PROFILE_TAG_PREFIX}<profile> tag ({tags})"
            )
        found.append((str(name), profile))
    return found


def _spawn_with_collision_offsets(
    original_func, rest_offset: float, contact_offset: float, filter_links: list[str] | None = None
):
    """Wrap the cloth spawn: set sim-mesh collision offsets and (optionally) filter collisions
    between the cloth and the given robot links (``{env}/Robot/<link>``)."""

    def _spawn(prim_path, cfg, *args, **kwargs):
        prim = original_func(prim_path, cfg, *args, **kwargs)
        import isaaclab.sim as sim_utils
        from pxr import PhysxSchema

        for path in sim_utils.find_matching_prim_paths(str(prim_path)):
            stage = prim.GetStage()
            sim_mesh = stage.GetPrimAtPath(f"{path}/sim_mesh")
            if not sim_mesh.IsValid():
                manaprint(
                    f"WARNING: surface cloth {path}: no sim_mesh, collision offsets not set"
                )
                continue
            api = PhysxSchema.PhysxCollisionAPI.Apply(sim_mesh)
            api.CreateRestOffsetAttr().Set(float(rest_offset))
            api.CreateContactOffsetAttr().Set(float(contact_offset))
            if filter_links:
                from pxr import Sdf, UsdPhysics

                env_root = path.rsplit("/", 1)[0]
                targets = [Sdf.Path(f"{env_root}/Robot/{link}") for link in filter_links]
                for target_prim in (stage.GetPrimAtPath(path), sim_mesh):
                    UsdPhysics.FilteredPairsAPI.Apply(target_prim).CreateFilteredPairsRel().SetTargets(targets)
        return prim

    return _spawn


def attach_surface_cloths(
    scene: Any, env_cfg: Any, *, device: str, events: Any
) -> list[str]:
    """Add one surface-deformable cloth (+ reset/startup events) per proxy found in ``scene``."""
    proxies = find_cloth_proxies(scene)
    if not proxies:
        return []
    if not str(device).lower().startswith("cuda"):
        raise RuntimeError(
            "PhysX surface-deformable cloth needs GPU dynamics; this env must run with a CUDA PhysX runtime "
            f"(got device={device!r}; use runtime/solver/physx_gpu_sync.yaml)"
        )
    import isaaclab.sim as sim_utils
    from isaaclab.assets import DeformableObjectCfg
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
    from isaaclab_physx.sim.spawners.materials import (
        PhysxSurfaceDeformableBodyMaterialCfg,
    )

    from manaenv.assets.local_discovery import resolve_usd_path

    added = []
    for proxy_name, profile_name in proxies:
        prof = PROFILES[profile_name]
        cloth_name = f"{proxy_name}{CLOTH_SUFFIX}"
        spawn = sim_utils.UsdFileCfg(
            usd_path=resolve_usd_path(prof["usd_asset"]),
            deformable_props=PhysxDeformableBodyPropertiesCfg(
                mass=float(prof["mass"]), **prof["body"]
            ),
            physics_material=PhysxSurfaceDeformableBodyMaterialCfg(**prof["material"]),
        )
        # With the point grasp the captured vertices follow the fingertips kinematically; if the jaws
        # also collide with the cloth, PhysX pushes those vertices out of the finger shapes while the
        # grasp writes them back every sub-step, which blows the cloth up (measured vertex speeds of
        # 15-50 m/s).  So in that mode the jaw links do not collide with the cloth (the palm/arm do).
        filter_links = list(GRIPPER_FINGER_BODIES) if _grasp_mode(prof).startswith("point") else None
        spawn.func = _spawn_with_collision_offsets(
            spawn.func, prof["rest_offset"], prof["contact_offset"], filter_links
        )
        # Spawn pose only matters until the first reset event re-lays the cloth on the proxy.
        cfg = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + cloth_name,
            spawn=spawn,
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.105, -0.273, -0.232)),
        )
        setattr(env_cfg.scene, cloth_name, cfg)
        if events is not None:
            setattr(
                events,
                f"{cloth_name}_reset_to_proxy",
                EventTermCfg(
                    func=reset_surface_cloth_to_proxy,
                    mode="reset",
                    params={
                        "cloth_name": cloth_name,
                        "proxy_name": proxy_name,
                        "profile_name": profile_name,
                    },
                ),
            )
            if _grasp_mode(prof) == "point":
                setattr(
                    events,
                    f"{cloth_name}_point_grasp",
                    EventTermCfg(
                        func=update_point_grasp,
                        mode="interval",
                        interval_range_s=(0.0, 0.0),
                        params={"cloth_name": cloth_name},
                    ),
                )
            mu = float(prof.get("gripper_friction", 0.0) or 0.0)
            if mu > 0.0:
                from isaaclab.envs import mdp as il_mdp

                setattr(
                    events,
                    f"{cloth_name}_gripper_friction",
                    EventTermCfg(
                        func=il_mdp.randomize_rigid_body_material,
                        mode="startup",
                        params={
                            "asset_cfg": SceneEntityCfg(
                                "robot", body_names=list(GRIPPER_FINGER_BODIES)
                            ),
                            "static_friction_range": (mu, mu),
                            "dynamic_friction_range": (mu, mu),
                            "restitution_range": (0.0, 0.0),
                            "num_buckets": 1,
                        },
                    ),
                )
        added.append(cloth_name)
        manaprint(
            f"INFO: surface cloth {cloth_name!r} (profile {profile_name}) attached to layout proxy {proxy_name!r}"
        )
    return added


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


def cloth_state(env: Any, cloth_name: str) -> dict[str, Any]:
    store = getattr(env, STATE_ATTR, None)
    if store is None:
        store = {}
        setattr(env, STATE_ATTR, store)
    return store.setdefault(cloth_name, {})


def cloth_points_env(env: Any, cloth_name: str) -> torch.Tensor:
    """Cloth simulation-mesh vertex positions in the env frame, ``(num_envs, V, 3)``."""
    cloth = env.scene[cloth_name]
    pos = as_torch(cloth.data.nodal_pos_w)
    return pos - env.scene.env_origins[:, None, :]


def cloth_velocities(env: Any, cloth_name: str) -> torch.Tensor:
    return as_torch(env.scene[cloth_name].data.nodal_vel_w)


def _yaw_from_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _set_tint(
    env: Any, cloth_name: str, env_id: int, rgb: tuple[float, float, float]
) -> None:
    try:
        import omni.usd
        from pxr import Gf, UsdShade

        stage = omni.usd.get_context().get_stage()
        root = f"/World/envs/env_{env_id}/{cloth_name}"
        sh = stage.GetPrimAtPath(f"{root}/Looks/towel/Shader")
        if not sh.IsValid():
            return
        UsdShade.Shader(sh).GetInput("diffuse_tint").Set(Gf.Vec3f(*rgb))
    except Exception as exc:  # noqa: BLE001 - cosmetic only
        manaprint(f"WARNING: surface cloth tint failed: {exc}")


def reset_surface_cloth_to_proxy(
    env: Any,
    env_ids: torch.Tensor | None,
    cloth_name: str,
    proxy_name: str,
    profile_name: str,
) -> None:
    """Lay the cloth flat (rest shape) on the proxy's current pose and record its corner roles."""
    prof = PROFILES[profile_name]
    cloth = env.scene[cloth_name]
    proxy = env.scene[proxy_name]
    device = env.device
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=device)
    env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    st = cloth_state(env, cloth_name)
    default = as_torch(cloth.data.default_nodal_state_w)[..., :3].to(device)
    num_v = default.shape[1]
    n = int(prof["grid"])
    if n * n != num_v:
        raise RuntimeError(
            f"surface cloth {cloth_name}: {num_v} sim vertices, expected {n}x{n} grid"
        )
    if "local" not in st:
        local = default - default.mean(dim=1, keepdim=True)
        local[..., 2] = 0.0
        st["local"] = local.clone()
        st["grid"] = n
        st["corners"] = torch.zeros((env.num_envs, 4), dtype=torch.long, device=device)
        st["fold_dir"] = torch.zeros((env.num_envs, 2), device=device)
        st["tint"] = ["blue"] * env.num_envs
    if "support_z" not in st:
        st["support_z"] = torch.zeros(env.num_envs, device=device)
    if "point_grips" not in st:
        st["point_grips"] = [[{}, {}] for _ in range(env.num_envs)]
    for e in env_ids.tolist():
        st["point_grips"][e] = [{}, {}]
    install_substep_point_grasp(env, cloth_name, profile_name)
    local = st["local"][env_ids]
    ppos = as_torch(proxy.data.root_pos_w)[env_ids].to(device)
    pquat = as_torch(proxy.data.root_quat_w)[env_ids].to(device)
    yaw = _yaw_from_quat_xyzw(pquat)
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    pos = torch.stack(
        (
            c * local[..., 0] - s * local[..., 1],
            s * local[..., 0] + c * local[..., 1],
            local[..., 2],
        ),
        dim=-1,
    )
    pos = pos + ppos[:, None, :]
    pos[..., 2] += float(prof["lift_z"])
    # support height (env frame) for the stage criteria: the proxy root sits on the tablecloth top
    st["support_z"][env_ids] = ppos[:, 2] - env.scene.env_origins[env_ids, 2]
    state = torch.zeros((len(env_ids), num_v, 6), dtype=torch.float32, device=device)
    state[..., :3] = pos
    cloth.write_nodal_state_to_sim_index(state, env_ids=env_ids)
    cloth.reset(env_ids)
    # corner roles: [near_left, near_right, far_left, far_right]; near = smaller world x (robot side),
    # left = larger world y (the robot's left arm side).
    # Identify rest corners geometrically; PhysX may reorder cooked mesh vertices.
    signs = torch.tensor(
        [[-1, -1], [-1, 1], [1, -1], [1, 1]], device=device, dtype=pos.dtype
    )
    grid_corners = torch.einsum("evd,cd->ecv", local[..., :2], signs).argmax(dim=-1)
    for row, env_id in enumerate(env_ids.tolist()):
        cp = pos[row, grid_corners[row]]
        order = torch.argsort(cp[:, 0])
        near, far = order[:2], order[2:]
        near = near[torch.argsort(cp[near, 1], descending=True)]
        far = far[torch.argsort(cp[far, 1], descending=True)]
        st["corners"][env_id] = grid_corners[row, torch.cat((near, far))]
        direction = cp[far, :2].mean(0) - cp[near, :2].mean(0)
        st["fold_dir"][env_id] = torch.nn.functional.normalize(direction, dim=0)
        from manaenv.terms.cloth_terms import _trace

        _trace(
            env,
            cloth_name,
            "reset",
            corners=pos[row : row + 1, st["corners"][env_id]],
            robot_root=as_torch(env.scene["robot"].data.root_pos_w)[
                env_id : env_id + 1
            ],
            robot_quat=as_torch(env.scene["robot"].data.root_quat_w)[
                env_id : env_id + 1
            ],
        )
        tints = prof.get("tints") or {}
        if tints:
            names = sorted(tints)
            pick = names[int(torch.randint(len(names), (1,)).item())]
            st["tint"][env_id] = pick
            _set_tint(env, cloth_name, env_id, tints[pick])
    st["reset_count"] = int(st.get("reset_count", 0)) + 1


def describe_profile(profile_name: str) -> str:
    prof = PROFILES[profile_name]
    m = prof["material"]
    return (
        f"{profile_name}: mass {prof['mass']} kg, grid {prof['grid']}x{prof['grid']}, thickness "
        f"{m['surface_thickness']} m, E {m['youngs_modulus']:.3g} Pa, bend {m['surface_bend_stiffness']}, "
        f"mu {m['dynamic_friction']}, gripper mu {prof.get('gripper_friction')}"
    )


__all__ = [
    "CLOTH_PROXY_TAG",
    "PROFILES",
    "attach_surface_cloths",
    "cloth_points_env",
    "cloth_state",
    "cloth_velocities",
    "describe_profile",
    "find_cloth_proxies",
    "reset_surface_cloth_to_proxy",
]


def update_point_grasp(env, env_ids, cloth_name: str) -> None:
    """Approximate closed-jaw grip with a local patch driven by measured fingertips.

    Only the closest particles in the measured pinch volume are driven. Free
    vertices keep their simulated state. A missed close requires reopening.
    """
    from manaenv.terms.cloth_terms import _TIP_BODIES
    from manaenv.utils.tensor_utils import as_torch

    st = cloth_state(env, cloth_name)
    if "point_grips" not in st:
        return
    robot = env.scene["robot"]
    if "grip_links" not in st:
        st["grip_links"] = [
            list(robot.find_bodies(list(names), preserve_order=True)[0])
            for names in _TIP_BODIES.values()
        ]
        st["grip_joints"] = list(
            robot.find_joints(
                ["left_arm_gripper", "right_arm_gripper"], preserve_order=True
            )[0]
        )
    cloth = env.scene[cloth_name]
    pos = as_torch(cloth.data.nodal_pos_w)
    vel = as_torch(cloth.data.nodal_vel_w)
    tips = as_torch(robot.data.body_pos_w)[:, st["grip_links"]]
    openings = as_torch(robot.data.joint_pos)[:, st["grip_joints"]]
    ids = (
        list(range(env.num_envs))
        if env_ids is None or isinstance(env_ids, slice)
        else torch.as_tensor(env_ids).tolist()
    )
    for e in ids:
        updates = []
        for arm in range(2):
            grasp = st["point_grips"][e][arm]
            opening = openings[e, arm].item()
            mid = tips[e, arm].mean(0)
            if opening > 0.50:
                grasp.clear()
                grasp["armed"] = True
                continue
            if grasp.get("armed") and opening < 0.30:
                grasp["armed"] = False
                distance, vertices = torch.topk(
                    torch.linalg.vector_norm(pos[e] - mid, dim=-1), 6, largest=False
                )
                vertices = vertices[distance < 0.024]
                if len(vertices):
                    grasp.update(
                        vertices=vertices,
                        offset=(pos[e, vertices] - mid).clone(),
                        previous=mid.clone(),
                    )
            if "vertices" in grasp:
                updates.append(
                    (
                        grasp["vertices"],
                        mid + grasp["offset"],
                        (mid - grasp["previous"]) / env.step_dt,
                    )
                )
                grasp["previous"] = mid.clone()
        if updates:
            state = torch.cat([pos[e], vel[e]], dim=-1).unsqueeze(0)
            for vertices, target, velocity in updates:
                state[0, vertices, :3] = target
                state[0, vertices, 3:] = velocity
            cloth.write_nodal_state_to_sim_index(
                state, env_ids=torch.tensor([e], device=env.device)
            )


# ---------------------------------------------------------------------------
# Point-grasp approximation, applied before *every physics sub-step*
# ---------------------------------------------------------------------------
#
# The interval-event version above runs once per control step (4 physics sub-steps at 120 Hz):
# the pinched vertices fall freely for 3 sub-steps and are then snapped back, which pumps energy
# into the cloth (measured vertex speeds of 1-3 m/s that never settle).  This version wraps the
# cloth's ``write_data_to_sim`` (called by ``ManagerBasedEnv.step`` right before each
# ``sim.step`` in synchronous mode) and drives the captured vertices with the measured fingertip
# pose and velocity on every sub-step.

GRASP_OPEN_RAD = 0.50     # gripper joint above this -> released / armed
GRASP_CLOSE_RAD = 0.30    # armed gripper closing below this -> capture
GRASP_RADIUS = 0.03       # the nearest cloth vertex must be this close to the fingertip midpoint [m]
GRASP_PATCH_RADIUS = 0.019  # captured patch: every vertex within this distance of that nearest vertex
# (~3x3 vertices of the 1.25 cm grid, ~2.5 cm like a jaw pad; with a folded towel both layers).
# Driving a single vertex is not an option: the probe blows up (NaN) when only one vertex is held.


def _grasp_mode(prof: Mapping[str, Any]) -> str:
    return os.environ.get("MANA_CLOTH_GRASP", str(prof.get("grasp", "point")))


def install_substep_point_grasp(env: Any, cloth_name: str, profile_name: str) -> bool:
    """Wrap ``env.scene[cloth_name].write_data_to_sim`` with the per-sub-step point grasp (idempotent)."""
    prof = PROFILES[profile_name]
    if _grasp_mode(prof) != "point_substep":
        return False
    cloth = env.scene[cloth_name]
    if getattr(cloth, "_mana_point_grasp_installed", False):
        return True
    original = cloth.write_data_to_sim

    def _write_data_to_sim():
        original()
        try:
            _substep_point_grasp(env, cloth_name, prof)
        except Exception as exc:  # noqa: BLE001 - never kill the physics loop
            manaprint(f"WARNING: surface cloth point grasp failed: {exc!r}")

    cloth.write_data_to_sim = _write_data_to_sim
    cloth._mana_point_grasp_installed = True
    manaprint(f"INFO: surface cloth {cloth_name!r}: per-sub-step point grasp installed")
    return True


def _grasp_trace(env: Any, cloth_name: str, rec: dict) -> None:
    path = os.environ.get("MANA_CLOTH_TRACE")
    if not path:
        return
    import json

    step = int(as_torch(env.episode_length_buf)[0].item()) if hasattr(env, "episode_length_buf") else -1
    with open(path, "a") as fh:
        fh.write(json.dumps({"tag": "grasp", "cloth": cloth_name, "step": step, **rec}) + "\n")
    manaprint(f"INFO: surface cloth grasp {rec}")


def _substep_point_grasp(env: Any, cloth_name: str, prof: Mapping[str, Any]) -> None:
    from manaenv.terms.cloth_terms import _TIP_BODIES

    st = cloth_state(env, cloth_name)
    grips = st.get("point_grips")
    if grips is None:
        return
    robot = env.scene["robot"]
    if "grip_links" not in st:
        st["grip_links"] = [list(robot.find_bodies(list(n), preserve_order=True)[0]) for n in _TIP_BODIES.values()]
        st["grip_joints"] = list(robot.find_joints(["left_arm_gripper", "right_arm_gripper"], preserve_order=True)[0])
    cloth = env.scene[cloth_name]
    tips_p = as_torch(robot.data.body_pos_w)
    tips_v = as_torch(robot.data.body_lin_vel_w)
    openings = as_torch(robot.data.joint_pos)[:, st["grip_joints"]]
    pos = vel = None
    write_vel = str(prof.get("grasp_write", "state")) == "state"
    for e in range(env.num_envs):
        updates = []
        for arm in range(2):
            g = grips[e][arm]
            opening = float(openings[e, arm])
            links = st["grip_links"][arm]
            mid = tips_p[e, links].mean(0)
            if opening > GRASP_OPEN_RAD:
                if g.get("vertices") is not None:
                    rec = {"env": e, "arm": arm, "event": "release"}
                    st.setdefault("grasp_log", []).append(rec)
                    _grasp_trace(env, cloth_name, rec)
                g.clear()
                g["armed"] = True
                continue
            if g.get("armed") and opening < GRASP_CLOSE_RAD:
                g["armed"] = False
                if pos is None:
                    pos = as_torch(cloth.data.nodal_pos_w).clone()
                    vel = as_torch(cloth.data.nodal_vel_w).clone()
                dist_all = torch.linalg.vector_norm(pos[e] - mid, dim=-1)
                nearest = int(torch.argmin(dist_all))
                d0 = float(dist_all[nearest])
                verts = pos.new_zeros(0, dtype=torch.long)
                if d0 < GRASP_RADIUS:
                    verts = torch.nonzero(
                        torch.linalg.vector_norm(pos[e] - pos[e, nearest], dim=-1) < GRASP_PATCH_RADIUS
                    ).flatten()
                    g["vertices"] = verts
                    g["offset"] = (pos[e, verts] - mid).clone()
                rec = {"env": e, "arm": arm, "event": "close", "captured": int(len(verts)), "nearest_m": round(d0, 4)}
                st.setdefault("grasp_log", []).append(rec)
                _grasp_trace(env, cloth_name, rec)
            if g.get("vertices") is not None:
                updates.append((g["vertices"], mid + g["offset"], tips_v[e, links].mean(0)))
        if updates:
            if pos is None:
                pos = as_torch(cloth.data.nodal_pos_w).clone()
                vel = as_torch(cloth.data.nodal_vel_w).clone()
            for verts, target, v in updates:
                pos[e, verts] = target
                vel[e, verts] = v
            ids = torch.tensor([e], device=env.device)
            if write_vel:
                cloth.write_nodal_state_to_sim_index(torch.cat([pos[e:e + 1], vel[e:e + 1]], dim=-1), env_ids=ids)
            else:
                cloth.write_nodal_pos_to_sim_index(pos[e:e + 1], env_ids=ids)
