#!/usr/bin/env python
"""Generate the BRIGHT 2026 hackathon task definitions (scene + Task DAG) and
register them as a separate evaluation board ``bright_arm6a_tabletop``.

Run from the repository root:

    python tools/bright/gen_bright_defs.py            # write YAML + register
    python tools/bright/gen_bright_defs.py --no-register

Outputs
  manaenv/defination/x2real/scene/bright/<task>/vnext.yaml
  manaenv/defination/x2real/task/bright/<task>/vnext_dag.yaml + stages/*.yaml
  manaenv/defination/x2real/task/x2real_bright_arm6a_tabletop_leaderboard.csv
and idempotently patches x2real.yaml (scene/task/assembly/rollout orders) and
boards.yaml (board entry).

Table surface frame used by every ``ranges`` block (green booth tabletop,
0.725 m deep x 1.15 m wide, centre ~world (0.205, -0.273, -0.240)):
  +x  -> toward the robot (near edge at x=+0.36)
  +y  -> robot's RIGHT     (left-arm side is y<0)
"""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DEF = REPO / "manaenv" / "defination"
SCENE_DIR = DEF / "x2real" / "scene" / "bright"
TASK_DIR = DEF / "x2real" / "task" / "bright"
BOARD = "bright_arm6a_tabletop"
CSV = DEF / "x2real" / "task" / f"x2real_{BOARD}_leaderboard.csv"
ROBOT = "EX001-6R-G"
TIMEOUTS = {"pour_beans": 120, "fold_towel": 120, "stack_rings": 150, "handover_tools": 150,
            "handover_tools_basket": 150, "arrange_flowers": 150}
TABLE_Z = -0.2404   # env-frame tabletop height
# runtime asset selectors that must not pick the same candidate in one episode
DISTINCT = {"towel": "towel_color", "distractor_towel": "towel_color"}
# non-uniform runtime asset selectors (weights are keyed by pool candidate id)
SELECTORS = {"basket": {"type": "weighted", "weights": {"green": 1.0, "blue": 0.5, "pink_box": 0.0}}}
LEFT_GRIPPER = [f"{{ENV_REGEX_NS}}/Robot/left_arm_gripper_{s}_link" for s in ("left", "left_tip", "right", "right_tip")]
RIGHT_GRIPPER = [f"{{ENV_REGEX_NS}}/Robot/right_arm_gripper_{s}_link" for s in ("left", "left_tip", "right", "right_tip")]


class _Dumper(yaml.SafeDumper):
    pass


def _repr_list(dumper, data):
    flow = all(isinstance(v, (int, float)) for v in data) and len(data) <= 4
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _repr_list)


def dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=110))


# ----------------------------------------------------------------------------
# Scene building blocks
# ----------------------------------------------------------------------------
BACKGROUND = {
    "id": "booth",
    "asset": {"type": "asset_id", "id": "green_booth_mesh"},
    "pose": {"position": [-0.452, -0.271, -0.985], "orientation": [0.0, 0.0, -0.704, 0.709]},
    "required_surfaces": ["tabletop"],
}
CLOTHS = ["bright_tablecloth_blue_cloud", "bright_tablecloth_red_check", "bright_tablecloth_forest_deer",
          "bright_tablecloth_gray_grid"]   # the 4 bought 90x150 cm cloths


def pool(obj_id, assets, tags, **extra):
    o = {"id": obj_id, "asset": {"type": "pool", "candidates": [{"id": "any", "assets": list(assets)}]},
         "semantic_tags": list(tags)}
    o.update(extra)
    return o


def pool_multi(obj_id, candidates, tags, default=None, **extra):
    """Pool with one candidate id per style, so runtime selectors can weight them (SELECTORS)."""
    o = {"id": obj_id, "asset": {"type": "pool", "cache_size": len(candidates), "cache_slot_order": "authored",
                                 "candidates": [{"id": cid, "assets": list(a)} for cid, a in candidates.items()]},
         "semantic_tags": list(tags), "_default": default or next(iter(candidates))}
    o.update(extra)
    return o


def fixed(obj_id, asset, tags, **extra):
    o = {"id": obj_id, "asset": {"type": "asset_id", "id": asset}, "semantic_tags": list(tags)}
    o.update(extra)
    return o


def surf(x, y, yaws=None, surface="cloth", orientation=None):
    p = {"type": "surface", "surface": surface, "ranges": {"x": x, "y": y}}
    if orientation is not None:
        p["orientation"] = orientation
    if yaws is not None:
        p["yaws"] = yaws
    return p


def scene_doc(task, goal, objects, nodes, groups=None, surfaces=None):
    pools = [o for o in objects if o["asset"]["type"] == "pool"]
    defaults = {o["id"]: o.pop("_default") for o in pools if "_default" in o}
    nodes = dict(nodes)
    for o in objects:  # group members must still be declared as (placement-less) nodes
        if "instances" not in o and o["id"] not in nodes:
            nodes[o["id"]] = {"object": o["id"]}
    has_cloth = any(o["id"] == "tablecloth" for o in objects)
    layout = {"surfaces": {"tabletop": {"type": "ref", "ref": "background.tabletop"},
                           **({"cloth": {"type": "ref", "ref": "tablecloth.top"}} if has_cloth else {}),
                           **(surfaces or {})},
              "nodes": nodes}
    if groups:
        layout["groups"] = groups
    return {
        "definition": {"scene_type": f"bright_{task}_scene",
                       "metadata": {"name": f"bright_{task}", "source": f"bright/{task}", "semantic_goal": goal}},
        "render_settings": {"gen_method": 0, "tags": "green_booth"},
        "background": copy.deepcopy(BACKGROUND),
        "objects": objects,
        "layout": layout,
        "variants": {
            "materialization": [
                {"id": f"{o['id']}_pick", "type": "asset_pool_choice",
                 # one choice per candidate (a choice may not bind the same object twice); build-time default
                 # candidate gets the weight, runtime episodes follow SELECTORS / uniform
                 "choices": [{"id": f"pick_{c['id']}", "candidate_refs": [f"{o['id']}.{c['id']}"],
                              "weight": (1.0 if c["id"] == defaults.get(o["id"], c["id"]) else 0.001)}
                             for c in o["asset"]["candidates"]]}
                for o in pools],
            "assignments": [],
        },
    }


def tablecloth():
    return pool("tablecloth", CLOTHS, ["background_cloth"], kinematic=True,
                anchors={"local_surfaces": {"top": {"center": [0.0, 0.0, 0.0015], "x_axis": [-1.0, 0.0, 0.0], "y_axis": [0.0, -1.0, 0.0], "bounds": {"x": [-0.32, 0.32], "y": [-0.53, 0.53]}}}})


def slot_group(members, slot_ranges, yaws=None, surface="cloth"):
    """Permutation slot group: each member lands in one slot; the permutation is sampled per episode."""
    slots = {}
    for k, (x, y) in enumerate(slot_ranges):
        s = {"type": "surface", "surface": surface, "ranges": {"x": x, "y": y}}
        if yaws is not None:
            s["yaws"] = yaws
        slots[f"slot_{k}"] = s
    return {"type": "slot", "nodes": list(members), "slots": slots,
            "assignments": {"mode": "permutation", "nodes": list(members), "slots": list(slots)},
            "default_assignment": "perm_0"}


TABLECLOTH_NODE = {"object": "tablecloth", "placement": surf([0.0, 0.0], [0.0, 0.0], yaws=0.0, surface="tabletop")}


# ----------------------------------------------------------------------------
# Task building blocks
# ----------------------------------------------------------------------------

def stage_base(task, sensors, *, dropped="@pickable"):
    return {"direct_task_generator": {
        "definition": {"gen_method": 0, "episode_length_s": float(TIMEOUTS[task]), "num_episodes": 10,
                       "max_steps_per_episode": TIMEOUTS[task] * 30},
        "component_sections": {
            "termination_component": {"terms": {
                "mdp.time_out": {"time_out": True},
                "object_dropped": {"object_cfg": dropped, "minimum_height": -0.25, "ignore_below": -1.0},
            }},
            "event_component": {"terms": {"mdp.reset_scene_to_default": {"mode": "reset",
                                                                           "reset_joint_targets": True}}},
            "metric_component": {"terms": {"success_rate": {}, "object_moved_rate": {"object_cfg": dropped}}},
            "viewer_component": {"terms": {"look_at_position": {
                "lookat_position": [0.204651, -0.268246, -0.240423], "offset": [-1.0, 0.0, 1.5]}}},
            "scene_constraint_component": {"constraints": [{"mode": 0, "tags": f"bright_{task}_scene"}]},
            "sensor_component": {"sensors": [{k: v} for k, v in sensors.items()]},
        }}}


def stage(task, name, success, progress):
    return {"include": f"x2real/task/bright/{task}/stages/_stage_base.yaml",
            "direct_task_generator": {
                "definition": {"task_type": f"bright_{task}_{name}"},
                "component_sections": {
                    "termination_component": {"terms": {"success": success}},
                    "log_component": {"entries": {"progress": {"trigger": "success", "value": progress}}},
                }}}


def all_of(**terms):
    return {"terms": [{k: v} for k, v in terms.items()]}


def contact(sensor, thr=0.2, steps=5):
    return {"_func": "object_contact_destination", "contact_sensor_cfgs": f"@{sensor}", "force_threshold": thr,
            "consecutive_steps": steps}


def no_contact(sensors, thr=0.05, steps=5):
    return {"_func": "object_no_contact_destination", "contact_sensor_cfgs": [f"@{s}" for s in sensors],
            "force_threshold": thr, "consecutive_steps": steps}


def lifted(obj, min_lift=0.04, sensor=None, steps=3):
    t = {"_func": "object_lifted", "object_cfg": obj, "min_lift": min_lift, "consecutive_steps": steps}
    if sensor:
        t["contact_sensor"] = sensor
    else:
        t["settle_steps"] = 30
    return t


def zone(obj, ref=None, steps=5, vel=None, **kw):
    t = {"_func": "object_position_in_zone", "object_cfg": obj, "consecutive_steps": steps}
    if ref:
        t["reference_cfg"] = ref
    if vel is not None:
        t["velocity_threshold"] = vel
    t.update(kw)
    return t


def tilt(obj, axis="z", steps=5, **kw):
    t = {"_func": "object_tilt_in_range", "object_cfg": obj, "axis": axis, "velocity_threshold": 0.5,
         "consecutive_steps": steps}
    t.update(kw)
    return t


def dag_doc(task, instruction, nodes, *, runtime_assets, runtime_nodes, runtime_groups=(), lighting=True):
    targets = {
        "assets": [{"id": f"{n}_asset", "target": {"type": "node", "id": n}, "selector": SELECTORS.get(n) or ({"type": "uniform", "distinct_id": DISTINCT[n]} if n in DISTINCT else {"type": "uniform"})}
                   for n in runtime_assets],
        "placements": ([{"id": n, "target": {"type": "node", "id": n}, "sample": {"type": "inherit"}}
                        for n in runtime_nodes] +
                       [{"id": g, "target": {"type": "group", "id": g}, "sample": {"type": "inherit"}}
                        for g in runtime_groups]),
    }
    rs = {"id": "runtime", "targets": targets}
    if lighting:
        rs["global"] = {"lighting": {"dome": {"intensity": {"range": [600.0, 2600.0]},
                                              "color_temperature": {"range": [3600.0, 7500.0]},
                                              "rotation": {"range": [-180.0, 180.0]}}}}
    edges = [{"src": a["id"], "dst": b["id"]} for a, b in zip(nodes, nodes[1:])]
    return {"graph_task_generator": {
        "definition": {
            "task_type": f"bright_{task}",
            "layout": {"runtime_sample": rs},
            "component_sections": {
                "log_component": {"entries": {"active_assets": {
                    "trigger": "enter", "func": "schema_vnext_active_asset_ids_label",
                    "params": {"object_names": list(runtime_assets)}}}},
                "instruction_component": {"instructions": [{"text": instruction,
                                                            "log": {"instruction_variant": "direct"}}]},
                "overlay_component": {"evaluation": True, "entries": [
                    {"log_key": "active_assets", "label": "Assets", "font_size": 18}]},
            }},
        "nodes": [{"id": n["id"], "include": f"x2real/task/bright/{task}/stages/{n['id']}.yaml",
                   "score": n.get("score", 2)} for n in nodes],
        "edges": edges,
    }}


def write_task(task, sensors, stages, dag, scene, *, dropped="@pickable"):
    tdir = TASK_DIR / task
    if tdir.exists():
        for f in sorted(tdir.rglob("*.yaml")):
            f.unlink()
    dump(SCENE_DIR / task / "vnext.yaml", scene)
    dump(tdir / "stages" / "_stage_base.yaml", stage_base(task, sensors, dropped=dropped))
    for name, (success, progress) in stages.items():
        dump(tdir / "stages" / f"{name}.yaml", stage(task, name, success, progress))
    dump(tdir / "vnext_dag.yaml", dag)


# ----------------------------------------------------------------------------
# Final-PDF helpers (2026-11-22): left/right interchangeable roles, OR branches, latches, mirrored layouts
#   "两臂分工按物体位置分配，左右可互换" -> no stage hard-codes the left or the right gripper; layouts are
#   authored once (source objects on the robot-LEFT side, y<0) and mirrored (y -> -y) as a 2nd slot case.
# ----------------------------------------------------------------------------
HANDS = ("left", "right")
OTHER = {"left": "right", "right": "left"}
GRIPPER_LINKS = {"left": LEFT_GRIPPER, "right": RIGHT_GRIPPER}
ANY_GRIPPER = ["left_gripper_contact", "right_gripper_contact"]   # object_lifted anchors on either hand


def hand_sensors(prefix, role):
    """``<prefix>_left_contact`` / ``<prefix>_right_contact``: contact of ``@role`` with each gripper's 4 finger links."""
    return {f"{prefix}_{h}_contact": {"type": "contact", "source": f"@{role}", "against": GRIPPER_LINKS[h]}
            for h in HANDS}


def every(**terms):
    return {"_combine": "and", "terms": [{k: v} for k, v in terms.items()]}


def any_of(**terms):
    return {"_combine": "or", "terms": [{k: v} for k, v in terms.items()]}


def negate(name, term):
    return {"_combine": "not", "terms": [{name: term}]}


def latched(term):
    """Hold-after-hit. Stage terms only run while the stage is the DAG frontier, so this latches what happened
    *during* the stage (e.g. which hand held the object when the hand-over started)."""
    return {**term, "_latch": True}


def held_by_any(prefix, thr=0.1, steps=1):
    return any_of(**{f"{prefix}_by_{h}": contact(f"{prefix}_{h}_contact", thr, steps) for h in HANDS})


def held_by_one(prefix, thr=0.1, steps=3):
    """Held by exactly one gripper (either)."""
    return any_of(**{f"{prefix}_only_{h}": every(**{
        f"{prefix}_{h}_on": contact(f"{prefix}_{h}_contact", thr, steps),
        f"{prefix}_{OTHER[h]}_off": no_contact([f"{prefix}_{OTHER[h]}_contact"], 0.05, 1)}) for h in HANDS})


def split_hands(a, b, a_thr=0.1, a_steps=1, b_thr=0.1, b_steps=1):
    """Object ``a`` held by one gripper and object ``b`` by the OTHER gripper -- either assignment.
    (Contact counters are keyed by sensor, so each sensor appears at most once with steps > 1 per stage.)"""
    return any_of(**{f"{a}_{h}__{b}_{OTHER[h]}": every(**{
        f"{a}_{h}": contact(f"{a}_{h}_contact", a_thr, a_steps),
        f"{b}_{OTHER[h]}": contact(f"{b}_{OTHER[h]}_contact", b_thr, b_steps)}) for h in HANDS})


def _mirror(y, side):
    return list(y) if side == "L" else [-y[1], -y[0]]


def mirrored_slot_group(fixed_slots, pools=(), surface="cloth", n_perm=6, seed=2026):
    """Explicit slot group = authored layout (case ``l*``) + its mirror image across the table centre line (``r*``).

    fixed_slots: {node: (x_range, y_range, yaws)}       one slot per node
    pools:       [(nodes, [(x_range, y_range), ...], yaws)]  nodes permuted over these slots, ``n_perm`` sampled
                 permutations per side (identity first)
    Every reset picks one case uniformly, so each object shows up on both sides of the table.
    """
    import math
    import random

    slots = {}

    def add(sid, x, y, yaws):
        # slot yaws must be a [lo, hi] list; None = any yaw (round objects)
        slots[sid] = {"type": "surface", "surface": surface, "ranges": {"x": list(x), "y": list(y)},
                      "yaws": list(yaws) if yaws is not None else [-3.14, 3.14]}
    rng = random.Random(seed)
    perms = []
    for nodes, ranges, _ in pools:
        idx = list(range(len(ranges)))
        cand = [idx[:len(nodes)]]
        limit = min(n_perm, math.perm(len(ranges), len(nodes)))
        while len(cand) < limit:
            p = rng.sample(idx, len(nodes))
            if p not in cand:
                cand.append(p)
        perms.append(cand)
    n_cases = max([len(c) for c in perms] + [1])
    items = []
    for side in ("L", "R"):
        for n, (x, y, yaws) in fixed_slots.items():
            add(f"{n}_{side}", x, _mirror(y, side), yaws)
        for gi, (nodes, ranges, yaws) in enumerate(pools):
            for j, (x, y) in enumerate(ranges):
                add(f"p{gi}s{j}_{side}", x, _mirror(y, side), yaws)
        for c in range(n_cases):
            b = {n: f"{n}_{side}" for n in fixed_slots}
            for gi, (nodes, _, _) in enumerate(pools):
                for node, j in zip(nodes, perms[gi][c % len(perms[gi])]):
                    b[node] = f"p{gi}s{j}_{side}"
            items.append({"id": f"{side.lower()}{c}", "bindings": b})
    members = list(fixed_slots) + [n for nodes, _, _ in pools for n in nodes]
    return {"type": "slot", "nodes": members, "slots": slots, "assignments": {"mode": "explicit", "items": items},
            "default_assignment": "l0"}


# The Task DAG is static per env, so "loop per instruction" (PDF: stack rings 1-5, tools 1-3, flowers 1-5) is
# expressed as one env per round count / target list; all variants of a task share the same props and layout.
TIMEOUTS.update({"pour_beans": 120, "fold_towel": 120})


def rounds_timeout(n, base, per_round):
    return int(base + per_round * n)


# ----------------------------------------------------------------------------
# 1. Pour beans: one hand steadies the chosen (blue) bowl on the table, the OTHER hand pours (final PDF)
# ----------------------------------------------------------------------------
N_BEANS = 9


# Pour scene layout notes (final): the bowls must swap sides every episode, and a slot group only keeps its members
# coupled (one case for all of them) on an object surface such as the table cloth -- on the background "tabletop"
# surface each bowl picked its slot independently and they spawned on top of each other half of the time.  So this
# scene now has the cloth too.  The runtime sampler supports only ONE level of object-on-object, and the beans sit
# on the cup's rim, so the cup + beans are NOT re-sampled per episode: the cup stays front-centre (both arms reach it)
# and the beans are reset into it by the scene reset; only the cloth and the bowls are re-sampled.
CUP_RIM_Z = 0.111   # cup 10.5 cm high + 6 mm


def task_pour_beans():
    task = "pour_beans"
    beans = [f"bean_{i}" for i in range(1, N_BEANS + 1)]
    objects = [
        tablecloth(),
        fixed("target_bowl", "bright_bowl_blue", ["container", "target_bowl"]),
        pool("distractor_bowl", ["bright_bowl_green"], ["distractor"]),     # PDF: 2 bowls, blue + green
        fixed("bean_cup", "bright_cup_green", ["pickable", "bean_cup"],
              anchors={"local_surfaces": {"rim": {"center": [0.0, 0.0, CUP_RIM_Z], "x_axis": [1.0, 0.0, 0.0],
                                                  "y_axis": [0.0, 1.0, 0.0],
                                                  "bounds": {"x": [-0.026, 0.026], "y": [-0.026, 0.026]}}}}),
        {"id": "beans", "asset": {"type": "asset_id", "id": "bright_bean_red"},
         "instances": {"type": "auto", "count": N_BEANS, "template": "bean_{i}", "start_index": 1},
         "spawn_mass": [0.0003, 0.0005], "semantic_tags": ["content", "beans"]},
    ]
    nodes = {"tablecloth": TABLECLOTH_NODE,
             "bean_cup": {"object": "bean_cup", "placement": surf([0.10, 0.12], [-0.01, 0.01], yaws=0.0)}}
    groups = {
        # blue bowl left / green bowl right (authored) or mirrored, either side of the cup
        "bowls_layout": mirrored_slot_group(
            {"target_bowl": ((0.06, 0.12), (-0.30, -0.22), None), "distractor_bowl": ((0.06, 0.12), (0.22, 0.30), None)},
            n_perm=1),
        "beans_in_cup": {"type": "region", "nodes": "@instances:beans", "surface": "cup_rim",
                         "ranges": {"x": [-0.024, 0.024], "y": [-0.024, 0.024]}, "yaws": 0.0,
                         "sample_method": "stratified", "spacing_mode": "center_distance",
                         "min_spacing": 0.003, "jitter": 0.2, "max_attempts": 4000,
                         "ignore_intra_group_collision": True}}
    surfaces = {"cup_rim": {"type": "ref", "ref": "bean_cup.rim"}}
    scene = scene_doc(task, "One hand steadies the light (20 cm, 130 g, low-friction) blue bowl on the table, the other "
                            "hand pours all red beans from the cup into it and puts the cup back (sides swap)",
                      objects, nodes, groups, surfaces)
    sensors = {**hand_sensors("bowl", "target_bowl"), **hand_sensors("cup", "bean_cup")}
    # PDF: "落点在碗口 Φ200mm 内，洒出 0 粒" -> every bean inside the blue bowl
    beans_in_bowl = {"_func": "objects_count_in_relative_zone", "object_names": beans,
                     "reference_cfg": "@target_bowl", "min_count": N_BEANS, "xy_distance_threshold": 0.09,
                     "z_min_offset": -0.01, "z_max_offset": 0.10, "velocity_threshold": 0.3,
                     "consecutive_steps": 10}
    # "put the cup back to its original place (±30 mm)": the displacement anchor is locked on the first frame of the
    # episode by a no-op (min_displacement 0) copy in the first stage; anchors are keyed by object, so the
    # return_cup stage measures against the cup's spawn position.
    cup_moved = {"_func": "any_object_displaced_from_anchor", "object_cfgs": "@bean_cup", "xy_only": True}
    stages = {
        "hold_bowl": (every(hold=held_by_any("bowl", 0.2, 10), cup_anchor={**cup_moved, "min_displacement": 0.0}),
                      "one hand holds the rim of the blue bowl (C1)"),
        "pick_cup": (every(hands=split_hands("bowl", "cup", 0.1, 1, 0.2, 5),
                           cup_lifted=lifted("@bean_cup", 0.05, sensor=ANY_GRIPPER)),
                     "the other hand picks up the bean cup (G1)"),
        "cup_over_bowl": (every(over=zone("@bean_cup", "@target_bowl", 5, xy_distance_threshold=0.09,
                                          z_min_offset=0.06, z_max_offset=0.30),
                                hands=split_hands("bowl", "cup")),
                          "cup moved above the bowl, bowl still held by the other hand (M1)"),
        "pour": (every(all_beans=beans_in_bowl, hands=split_hands("bowl", "cup")),
                 "all beans poured into the bowl while the other hand holds it (L3)"),
        "return_cup": (every(upright=tilt("@bean_cup", "z", 10, min_dot=0.9, max_dot=1.0),
                             on_table=zone("@bean_cup", None, 10, vel=0.05, z_range=[TABLE_Z - 0.02, TABLE_Z + 0.03]),
                             home=negate("moved", {**cup_moved, "min_displacement": 0.03}),
                             released=no_contact(["cup_left_contact", "cup_right_contact"], 0.05, 10),
                             beans_kept={**beans_in_bowl, "consecutive_steps": 1}),
                       "cup put back upright within 30 mm of where it was (P2)"),
    }
    dag = dag_doc(task, "Use the arm on the blue bowl's side to hold the blue bowl steady by its rim. With the other "
                        "arm pick up the green cup with red beans, move it above the blue bowl and tilt it to pour all "
                        "the beans into the bowl without spilling any, then put the cup back where it was.",
                  [{"id": s} for s in stages],
                  runtime_assets=["tablecloth", "distractor_bowl"], runtime_nodes=["tablecloth"],
                  runtime_groups=["bowls_layout"])
    write_task(task, sensors, stages, dag, scene, dropped="@pickable")


# ----------------------------------------------------------------------------
# 2. Fold towel once + put it into the basket (RIGID placeholder towel; the cloth version is a separate env)
# ----------------------------------------------------------------------------
TOWELS = ["bright_towel_blue", "bright_towel_pink", "bright_towel_yellow", "bright_towel_green",
          "bright_towel_purple"]
BASKET_RIM_Z, BASKET_FLOOR_Z = 0.085, 0.004   # bright_basket_green, basket frame (long 35 cm side = local x)


def task_fold_towel():
    task = "fold_towel"
    objects = [
        tablecloth(),
        pool("towel", TOWELS, ["pickable", "towel", "cloth_placeholder"]),
        fixed("basket", "bright_basket_green", ["container"]),        # PDF: 1 light-green basket, 35 x 25.5 x 8.5
    ]
    nodes = {
        "tablecloth": TABLECLOTH_NODE,
        "towel": {"object": "towel", "placement": surf([0.10, 0.16], [-0.04, 0.04], yaws=[-0.15, 0.15])},
    }
    # basket behind the towel on the robot-right side, or (mirrored) on the left
    groups = {"basket_side": mirrored_slot_group({"basket": ((-0.16, -0.08), (0.24, 0.32), [1.5608, 1.5808])}, n_perm=1)}
    scene = scene_doc(task, "Fold the towel in half away from the robot, then put it flat into the basket",
                      objects, nodes, groups)
    sensors = hand_sensors("towel", "towel")
    both = ["towel_left_contact", "towel_right_contact"]
    in_basket = dict(in_reference_frame=True, x_range=[-0.06, 0.06], y_range=[-0.07, 0.07])
    stages = {
        "grasp_corners": (all_of(left=contact("towel_left_contact", 0.1, 5), right=contact("towel_right_contact", 0.1, 5)),
                          "both grippers pinch the near corners (M3)"),
        "fold_lift": (all_of(left=contact("towel_left_contact", 0.1, 3), right=contact("towel_right_contact", 0.1, 3),
                             raised=lifted("@towel", 0.03, sensor=ANY_GRIPPER)),
                      "corners lifted and carried away from the robot (D3; rigid towel proxy)"),
        "release": (all_of(flat=tilt("@towel", "z", 10, min_dot=0.9, max_dot=1.0),
                           down=zone("@towel", None, 10, vel=0.05, z_range=[TABLE_Z - 0.02, TABLE_Z + 0.03]),
                           free=no_contact(both, 0.05, 10)),
                    "towel put down and released (P2)"),
        "pick_folded": (every(one_hand=held_by_one("towel", 0.1, 3),
                              raised=lifted("@towel", 0.05, sensor=ANY_GRIPPER)),
                        "one hand grasps the folded edge and lifts the towel (G1)"),
        # PDF M1: towel projection inside the 350 x 255 mm opening (folded towel 300 x 150 -> centre within
        # about +-50 / +-60 mm of the basket centre), carried above the rim
        "over_basket": (every(above=zone("@towel", "@container", 5, x_range=[-0.05, 0.05], y_range=[-0.06, 0.06],
                                         z_range=[BASKET_RIM_Z, 0.40], in_reference_frame=True),
                              held=held_by_any("towel", 0.05, 1)),
                        "towel carried right above the basket opening (M1)"),
        # PDF P2: released at <= 30 mm above the basket floor, lies flat and folded inside
        "into_basket": (every(lowered=latched(every(
                                  near_floor=zone("@towel", "@container", 1, z_range=[-0.01, BASKET_FLOOR_Z + 0.03],
                                                  **{**in_basket, "x_range": [-0.10, 0.10]}),
                                  held=held_by_any("towel", 0.05, 1))),
                              inside=zone("@towel", "@container", 10, vel=0.05, z_range=[-0.01, 0.04], **in_basket),
                              flat=tilt("@towel", "z", 10, min_dot=0.9, max_dot=1.0),
                              free=no_contact(both, 0.05, 10)),
                        "lowered to <= 30 mm above the basket floor, released, lying flat inside (P2)"),
    }
    dag = dag_doc(task, "Grasp the two near corners of the towel with both hands and fold it once away from you, put "
                        "it down and let go. Then with one hand grasp the middle of the folded edge, lift the folded "
                        "towel, move it right above the basket, lower it close to the basket bottom and release it so "
                        "it lies flat and folded in the basket.",
                  [{"id": s} for s in stages],
                  runtime_assets=["tablecloth", "towel"],
                  runtime_nodes=["tablecloth", "towel"], runtime_groups=["basket_side"])
    write_task(task, sensors, stages, dag, scene, dropped="@pickable")


# ----------------------------------------------------------------------------
# 3. Stack rings large -> small onto the single peg (5 rings; 1..5 per instruction)
# ----------------------------------------------------------------------------
BASE_H = 0.025    # keep in sync with STACK in gen_bright_assets.py
RING_T = 0.017
PEG_TOP = 0.125   # base 2.5 cm + peg 10 cm (whole set 12.5 cm)
N_RINGS = 5
# rings scattered on the side opposite to the base (authored: base left of centre, rings on the right)
RING_SLOTS = [((0.13, 0.17), (0.08, 0.12)), ((0.13, 0.17), (0.20, 0.24)), ((0.13, 0.17), (0.32, 0.36)),
              ((0.00, 0.04), (0.14, 0.18)), ((0.00, 0.04), (0.26, 0.30))]
COUNT_WORDS = {1: "the largest ring", 2: "the two largest rings", 3: "the three largest rings",
               4: "the four largest rings", 5: "all five rings"}


def task_stack_rings(n=N_RINGS, suffix=""):
    task = f"stack_rings{suffix}"
    rings = [f"ring_{k}" for k in range(1, N_RINGS + 1)]          # ring_1 = largest
    objects = [tablecloth(), fixed("stacker", "bright_stacker_round", ["destination", "stacker"])]
    objects += [fixed(r, f"bright_{r}", ["pickable", r]) for r in rings]
    nodes = {"tablecloth": TABLECLOTH_NODE}
    groups = {"stack_layout": mirrored_slot_group({"stacker": ((0.00, 0.06), (-0.10, -0.05), None)},
                                                  pools=[(rings, RING_SLOTS, [-3.14, 3.14])], n_perm=6)}
    scene = scene_doc(task, "One hand holds the round stacker base, the other hand stacks rings large-to-small on its "
                            "single peg (5 rings, sides swap)", objects, nodes, groups)
    sensors = hand_sensors("base", "stacker")
    for k in range(1, n + 1):
        sensors.update(hand_sensors(f"ring{k}", f"ring_{k}"))
    stages = {"press_base": (held_by_any("base", 0.5, 10), "one hand holds the stacker base (C1)")}
    for k in range(1, n + 1):
        ring, p = f"@ring_{k}", f"ring{k}"
        z_on = BASE_H + RING_T * (k - 1)
        stages[f"pick_ring{k}"] = (every(hands=split_hands("base", p, 0.1, 1, 0.1, 3),
                                         raised=lifted(ring, 0.03, sensor=ANY_GRIPPER)),
                                   f"the other hand picks ring {k} from above (G1)")
        stages[f"ring{k}_above_peg"] = (every(above=zone(ring, "@destination", 3, in_reference_frame=True,
                                                         x_range=[-0.06, 0.06], y_range=[-0.06, 0.06],
                                                         z_range=[PEG_TOP + 0.02, 0.40]),
                                              hands=split_hands("base", p)),
                                        f"ring {k} lifted >= 20 mm above the peg top (M1)")
        stages[f"ring{k}_aligned"] = (every(aligned=zone(ring, "@destination", 3, in_reference_frame=True,
                                                         x_range=[-0.012, 0.012], y_range=[-0.012, 0.012],
                                                         z_range=[PEG_TOP - 0.04, PEG_TOP + 0.04]),
                                            hands=split_hands("base", p)),
                                      f"ring {k} hole aligned with the peg tip (M3)")
        stages[f"ring{k}_on_peg"] = (every(on_peg=zone(ring, "@destination", 10, vel=0.05, in_reference_frame=True,
                                                       x_range=[-0.012, 0.012], y_range=[-0.012, 0.012],
                                                       z_range=[z_on - 0.008, z_on + 0.008]),
                                           released=no_contact([f"{p}_left_contact", f"{p}_right_contact"], 0.05, 10),
                                           base_held=held_by_any("base", 0.1, 1)),
                                     f"ring {k} put onto the peg and released (P3)")
    dag = dag_doc(task, f"Hold the base of the ring stacker steady with the arm on its side. With the other arm put "
                        f"{COUNT_WORDS[n]} onto the peg one at a time, from the largest to the smallest.",
                  [{"id": s} for s in stages], runtime_assets=["tablecloth"],
                  runtime_nodes=["tablecloth"], runtime_groups=["stack_layout"])
    write_task(task, sensors, stages, dag, scene)


# ----------------------------------------------------------------------------
# 4. Hand over tools (mid-air) and put them into the open toolbox (or the basket variant); 1-3 tools
# ----------------------------------------------------------------------------
TOOLS = {"hammer": ("bright_hammer", "hammer"), "screwdriver": ("bright_screwdriver", "flat-head screwdriver"),
         "screw_twister": ("bright_screw_twister", "red T-handle screw")}   # PDF: 锤子 / 一字螺丝刀 / 螺丝拧
# PDF M1 "lift to ~15 cm above the table centre": object root 10-25 cm above the table, over the middle of the table
CARRY_Z = (0.10, 0.25)
MID_X, MID_Y = [0.0, 0.40], [-0.40, -0.14]        # env frame (table centre y = -0.273)
HANDOVER_Z = (0.05, 0.35)                          # C2: in the air (root >= 5 cm, see guide), not touching the table
HANDOVER_X, HANDOVER_Y = [-0.05, 0.45], [-0.45, -0.10]
CONTAINERS = {
    # name: (asset, slot on the robot-right side, (x, y) half-extents of the inside in the container frame, z max)
    "toolbox": ("bright_toolbox", ((-0.04, 0.04), (0.24, 0.32), [-0.1, 0.1]), (0.085, 0.05), 0.06),
    "basket": ("bright_basket_green", ((-0.12, -0.04), (0.22, 0.30), [1.47, 1.67]), (0.16, 0.115), 0.08),
}
# (kept within |y| <= 0.37 so the wrist/gripper does not hide them in the head camera)
TOOL_SLOTS = [((0.06, 0.14), (-0.37, -0.35)), ((0.06, 0.14), (-0.26, -0.24)), ((0.06, 0.14), (-0.15, -0.13))]


def airborne_handover(prefix, obj):
    """C2 mid-air hand-over, either direction: when the stage starts the giver holds the object alone (latched);
    it succeeds once the OTHER gripper holds it, the giver has let go, and the object is in the air without touching
    the table cloth.  A "taker grabs then lets go again" sequence does not count."""
    swaps = {}
    for giver in HANDS:
        taker = OTHER[giver]
        swaps[f"{giver}_to_{taker}"] = every(**{
            f"{giver}_gave": latched(every(**{f"{giver}_had": contact(f"{prefix}_{giver}_contact", 0.05, 1),
                                               f"{taker}_had_not": no_contact([f"{prefix}_{taker}_contact"], 0.05, 1)})),
            f"{taker}_holds": contact(f"{prefix}_{taker}_contact", 0.05, 5),
            f"{giver}_released": no_contact([f"{prefix}_{giver}_contact"], 0.05, 5)})
    return every(swap=any_of(**swaps),
                 airborne=zone(obj, None, 5, x_range=HANDOVER_X, y_range=HANDOVER_Y,
                               z_range=[round(TABLE_Z + HANDOVER_Z[0], 4), round(TABLE_Z + HANDOVER_Z[1], 4)]),
                 untouched=no_contact([f"{prefix}_cloth_contact"], 0.05, 5))


def carry_to_middle(prefix, obj):
    return every(mid=zone(obj, None, 5, x_range=MID_X, y_range=MID_Y,
                          z_range=[round(TABLE_Z + CARRY_Z[0], 4), round(TABLE_Z + CARRY_Z[1], 4)]),
                 held=held_by_any(prefix, 0.05, 1),
                 untouched=no_contact([f"{prefix}_cloth_contact"], 0.05, 5))


def task_handover_tools(container="toolbox", tools=tuple(TOOLS), suffix=""):
    task = ("handover_tools" if container == "toolbox" else f"handover_tools_{container}") + suffix
    asset, (bx, by, byaw), (hx, hy), zmax = CONTAINERS[container]
    objects = [tablecloth(), fixed("box", asset, ["container", "toolbox"])]
    objects += [fixed(t, TOOLS[t][0], ["pickable", f"tool_{t}"]) for t in TOOLS]
    nodes = {"tablecloth": TABLECLOTH_NODE}
    # authored: tools lying along world x on the robot-left side, container on the right; mirrored case swaps
    groups = {"tools_layout": mirrored_slot_group({"box": (bx, by, byaw)},
                                                  pools=[(list(TOOLS), TOOL_SLOTS, [-0.08, 0.08])], n_perm=6)}
    where = "toolbox" if container == "toolbox" else "green basket"
    scene = scene_doc(task, f"One hand picks a tool, hands it to the other hand in mid-air above the table centre, "
                            f"which puts it into the {where} (sides swap)", objects, nodes, groups)
    sensors = {}
    stages = {}
    for t in tools:
        tool, p = f"@tool_{t}", f"tool_{t}"
        sensors.update(hand_sensors(p, p))
        sensors[f"{p}_cloth_contact"] = {"type": "contact", "source": tool, "against": "@background_cloth"}
        stages[f"pick_{t}"] = (every(grab=held_by_any(p, 0.1, 3), raised=lifted(tool, 0.04, sensor=ANY_GRIPPER)),
                               f"one hand picks the {TOOLS[t][1]} (G1)")
        stages[f"{t}_to_middle"] = (carry_to_middle(p, tool),
                                    f"{TOOLS[t][1]} lifted ~15 cm above the table centre, not touching the table (M1)")
        stages[f"{t}_handover"] = (airborne_handover(p, tool),
                                   f"the other hand takes the {TOOLS[t][1]} in mid-air, the giver releases (C2)")
        stages[f"{t}_over_box"] = (every(above=zone(tool, "@toolbox", 3, in_reference_frame=True, x_range=[-hx, hx],
                                                    y_range=[-hy, hy], z_range=[0.04, 0.35]),
                                         held=held_by_any(p, 0.05, 1)),
                                   f"{TOOLS[t][1]} aligned above the {where} opening (M3)")
        stages[f"{t}_in_box"] = (every(inside=zone(tool, "@toolbox", 10, vel=0.05, in_reference_frame=True,
                                                   x_range=[-hx, hx], y_range=[-hy, hy], z_range=[-0.005, zmax]),
                                       free=no_contact([f"{p}_left_contact", f"{p}_right_contact"], 0.05, 10)),
                                 f"{TOOLS[t][1]} placed in the {where} (P2)")
    names = [TOOLS[t][1] for t in tools]
    listing = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    dag = dag_doc(task, f"Hand over the {listing} one at a time, in this order. For each tool: with the arm on the "
                        f"tools' side pick it up, lift it about 15 cm above the middle of the table and hand it over in "
                        f"mid-air to the other arm, then the other arm puts it into the {where}. Leave the other tools "
                        f"on the table.",
                  [{"id": s} for s in stages], runtime_assets=["tablecloth"],
                  runtime_nodes=["tablecloth"], runtime_groups=["tools_layout"])
    write_task(task, sensors, stages, dag, scene)


def task_handover_tools_basket():
    task_handover_tools("basket")


# ----------------------------------------------------------------------------
# 5. Arrange flowers: one hand picks the stem, mid-air hand-over (the other hand takes the head), insert into the
#    vase; 5 roses of different colours, 1-5 per instruction
# ----------------------------------------------------------------------------
ROSES = {"purple": "deep purple", "blue": "light blue", "yellow": "yellow", "red": "red", "pink": "flesh-pink"}
# roses lie along world +x (head away from the robot), 10.5 cm apart (heads are 9 cm), authored on the robot-left.
# Slot x ranges act on the asset ORIGIN, which for the roses is the stem bottom (the head is 30 cm further away).
FLOWER_SLOTS = [((0.06, 0.10), (-0.433, -0.427)), ((0.06, 0.10), (-0.328, -0.322)), ((0.06, 0.10), (-0.223, -0.217)),
                ((0.06, 0.10), (-0.118, -0.112)), ((0.06, 0.10), (-0.013, -0.007))]


def task_arrange_flowers(colors=tuple(ROSES), suffix=""):
    task = f"arrange_flowers{suffix}"
    objects = [tablecloth(), fixed("vase", "bright_vase_white", ["container", "vase"])]
    objects += [fixed(f"flower_{c}", f"bright_flower_rose_{c}", ["pickable", f"flower_{c}"]) for c in ROSES]
    nodes = {"tablecloth": TABLECLOTH_NODE}
    groups = {"flowers_layout": mirrored_slot_group({"vase": ((-0.06, 0.04), (0.18, 0.30), None)},
                                                    pools=[([f"flower_{c}" for c in ROSES], FLOWER_SLOTS, [-0.02, 0.02])],
                                                    n_perm=6)}
    scene = scene_doc(task, "Insert crochet roses stem-first into the vase after a mid-air hand-over (5 colours, sides "
                            "swap)", objects, nodes, groups)
    sensors = {}
    stages = {}
    for c in colors:
        fl, p = f"@flower_{c}", f"flower_{c}"
        sensors.update(hand_sensors(p, p))
        sensors[f"{p}_vase_contact"] = {"type": "contact", "source": fl, "against": "@vase"}
        sensors[f"{p}_cloth_contact"] = {"type": "contact", "source": fl, "against": "@background_cloth"}
        stages[f"pick_{c}"] = (every(grab=held_by_any(p, 0.05, 3), raised=lifted(fl, 0.04, sensor=ANY_GRIPPER)),
                               f"one hand picks the {ROSES[c]} rose by the stem from above (G1)")
        stages[f"{c}_to_middle"] = (carry_to_middle(p, fl),
                                    f"{ROSES[c]} rose lifted ~15 cm above the table centre, not touching the table (M1)")
        stages[f"{c}_handover"] = (airborne_handover(p, fl),
                                   f"the other hand takes the {ROSES[c]} rose (head) in mid-air, the giver releases (C2)")
        # PDF M3: stem bottom (= asset origin) within 50 mm of the vase mouth centre, stem roughly vertical
        stages[f"{c}_over_vase"] = (every(above=zone(fl, "@vase", 3, xy_distance_threshold=0.05, z_min_offset=0.15,
                                                     z_max_offset=0.45),
                                          vertical=tilt(fl, "x", 3, min_abs_dot=0.8)),
                                    f"{ROSES[c]} stem bottom above the vase mouth (M3)")
        stages[f"{c}_inserted"] = (every(touch=contact(f"{p}_vase_contact", 0.01, 5),
                                         inside=zone(fl, "@vase", 5, vel=0.5, reference_footprint_zone=True,
                                                     xy_inset_ratio=0.1, z_range=[-0.03, 0.35]),
                                         upright=tilt(fl, "x", 5, min_abs_dot=0.6),
                                         free=no_contact([f"{p}_left_contact", f"{p}_right_contact"], 0.05, 5)),
                                   f"{ROSES[c]} rose inserted and released, stays in the vase (P3)")
    names = [ROSES[c] for c in colors]
    listing = names[0] + " rose" if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1] + " roses"
    dag = dag_doc(task, f"Put the {listing} into the vase one at a time, in this order. For each rose: with the arm on "
                        f"the flowers' side pick it up by the stem from above, lift it about 15 cm above the middle of "
                        f"the table and hand it over in mid-air to the other arm, which grasps the flower head; then "
                        f"insert the stem into the vase and let go. Leave the other roses on the table.",
                  [{"id": s} for s in stages], runtime_assets=["tablecloth"],
                  runtime_nodes=["tablecloth"], runtime_groups=["flowers_layout"])
    write_task(task, sensors, stages, dag, scene)


TASKS = {
    "pour_beans": task_pour_beans,
    "fold_towel": task_fold_towel,
    "stack_rings": task_stack_rings,
    "handover_tools": task_handover_tools,
    "handover_tools_basket": task_handover_tools_basket,
    "arrange_flowers": task_arrange_flowers,
}

# Final PDF (2026-11-22) "按指令循环" variants: one env per round count / target list (see the guide, section 8,
# for adding another one: one line here + one timeout, then re-run this script).
VARIANTS = {
    "stack_rings_r1": lambda: task_stack_rings(1, "_r1"),                            # largest ring only
    "stack_rings_r3": lambda: task_stack_rings(3, "_r3"),                            # 3 largest rings
    "handover_tools_r1": lambda: task_handover_tools("toolbox", ("hammer",), "_r1"),  # hammer only
    "arrange_flowers_r1": lambda: task_arrange_flowers(("red",), "_r1"),             # red rose only
    "arrange_flowers_r2": lambda: task_arrange_flowers(("yellow", "blue"), "_r2"),   # yellow, then light blue
}
TASKS.update(VARIANTS)
TIMEOUTS.update({
    "stack_rings": rounds_timeout(5, 60, 40), "stack_rings_r1": rounds_timeout(1, 60, 40),
    "stack_rings_r3": rounds_timeout(3, 60, 40),
    "handover_tools": rounds_timeout(3, 60, 60), "handover_tools_basket": rounds_timeout(3, 60, 60),
    "handover_tools_r1": rounds_timeout(1, 60, 60),
    "arrange_flowers": rounds_timeout(5, 60, 60), "arrange_flowers_r1": rounds_timeout(1, 60, 60),
    "arrange_flowers_r2": rounds_timeout(2, 60, 60),
})


def env_name(task):
    return f"artixon_arm6a_G_bright_{task}_id_env"


# ----------------------------------------------------------------------------
# Registration (idempotent text patches; keeps upstream formatting intact)
# ----------------------------------------------------------------------------
BEGIN, END = "# >>> bright2026 begin", "# <<< bright2026 end"


def _strip_block(text: str) -> str:
    return re.sub(rf"[ \t]*{re.escape(BEGIN)}.*?{re.escape(END)}[ \t]*\n", "", text, flags=re.S)


def _insert_list_items(text: str, key: str, items_yaml: str) -> str:
    """Append items at the end of top-level list ``key:``."""
    lines = text.split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.rstrip() == f"{key}:")
    end = start + 1
    while end < len(lines) and (lines[end].startswith("- ") or lines[end].startswith("  ") or lines[end] == ""):
        end += 1
    while end > start + 1 and lines[end - 1] == "":
        end -= 1
    block = [BEGIN] + items_yaml.rstrip("\n").split("\n") + [END]
    return "\n".join(lines[:end] + block + lines[end:])


def register(tasks):
    suite = DEF / "x2real.yaml"
    text = suite.read_text()
    for _ in range(4):
        text = _strip_block(text)
    scenes = "".join(f"- include: x2real/scene/bright/{t}/vnext.yaml\n" for t in tasks)
    taskso = "".join(f"- include: x2real/task/bright/{t}/vnext_dag.yaml\n" for t in tasks)
    asm = yaml.dump([{"assembly_generator": {
        "name": env_name(t), "scene_generator_name": f"bright_{t}_scene", "robot_generator_name": ROBOT,
        "task_generator_name": f"bright_{t}", "num_envs": 1,
        "metadata": {"x2real": {"embodiment": "artixon_arm6a", "capabilities": ["benchmark"],
                                "robot_generator_identity": ROBOT, "board": BOARD}}}} for t in tasks],
        sort_keys=False, Dumper=_Dumper)
    roll = yaml.dump([{"env_ref": env_name(t), "num_episodes": 10, "episode_length_s": TIMEOUTS[t],
                       "max_steps": TIMEOUTS[t] * 30, "num_parallel_envs": 1, "show_evaluation_overlay": True,
                       "planner": {"PolicyPlanner": {"policy_cfg": {"include": "policy/x2real_ps@v1.0.0.yaml"}}}}
                      for t in tasks], sort_keys=False, Dumper=_Dumper)
    text = _insert_list_items(text, "scene_orders", scenes)
    text = _insert_list_items(text, "task_orders", taskso)
    text = _insert_list_items(text, "assembly_orders", asm)
    text = _insert_list_items(text, "rollout", roll)
    suite.write_text(text)

    boards = DEF / "x2real" / "task" / "boards.yaml"
    btext = _strip_block(boards.read_text()).rstrip("\n") + "\n"
    btext += (f"  {BEGIN}\n  {BOARD}:\n    robot: artixon_arm6a\n    grippers:\n    - G\n    num_episodes: 10\n"
              f"    num_parallel_envs: 1\n    leaderboard_csv: x2real/task/{CSV.name}\n    suite: x2real.yaml\n"
              f"    scoring: id_only\n    capabilities:\n    - benchmark\n    case_count: {len(tasks)}\n"
              f"    save_path: ./outputs/x2real_eval/{BOARD}/benchmark\n  {END}\n")
    boards.write_text(btext)
    rows = ["subcategory,task,x2real_package,n_id,id_envs,id_timeout_s"]
    rows += [f"bright2026,{t},bright/{t},1,{env_name(t)},{TIMEOUTS[t]}" for t in tasks]
    CSV.write_text("\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    for t in a.tasks:
        TASKS[t]()
        print("wrote", t)
    if not a.no_register:
        register(list(TASKS))
        print("registered board", BOARD, "with", len(TASKS), "cases")


if __name__ == "__main__":
    main()
