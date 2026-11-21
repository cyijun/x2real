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
TIMEOUTS = {"pour_beans": 120, "fold_towel": 120, "stack_rings": 150, "handover_tools": 150, "arrange_flowers": 150}
TABLE_Z = -0.2404   # env-frame tabletop height
# runtime asset selectors that must not pick the same candidate in one episode
DISTINCT = {"towel": "towel_color", "distractor_towel": "towel_color"}
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
CLOTHS = ["bright_tablecloth_gray_grid", "bright_tablecloth_forest_deer", "bright_tablecloth_blue_cloud",
          "bright_tablecloth_wood"]


def pool(obj_id, assets, tags, **extra):
    o = {"id": obj_id, "asset": {"type": "pool", "candidates": [{"id": "any", "assets": list(assets)}]},
         "semantic_tags": list(tags)}
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
                 "choices": [{"id": "pick", "candidate_refs": [f"{o['id']}.any"], "weight": 1.0}]}
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
        "assets": [{"id": f"{n}_asset", "target": {"type": "node", "id": n}, "selector": ({"type": "uniform", "distinct_id": DISTINCT[n]} if n in DISTINCT else {"type": "uniform"})}
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
# 1. Pour beans (left hand steadies the bowl)
# ----------------------------------------------------------------------------
N_BEANS = 9


# Pour scene layout notes: ManaEnv's runtime sampler supports only ONE level of object-relative
# placement.  Beans must sit on the cup ("cup.rim" local surface), so the cup has to be placed directly
# on the background tabletop -- which means this scene cannot use the tablecloth object (that would make
# beans a 2nd-level child) nor the tray.  Background variation for this task comes from dome lighting.
CUP_RIM_Z = 0.102


def task_pour_beans():
    task = "pour_beans"
    beans = [f"bean_{i}" for i in range(1, N_BEANS + 1)]
    objects = [
        fixed("target_bowl", "bright_bowl_blue", ["container", "target_bowl"]),
        pool("distractor_bowl", ["bright_bowl_pink", "bright_bowl_beige", "bright_bowl_mint"], ["distractor"]),
        fixed("bean_cup", "bright_cup_green", ["pickable", "bean_cup"],
              anchors={"local_surfaces": {"rim": {"center": [0.0, 0.0, CUP_RIM_Z], "x_axis": [1.0, 0.0, 0.0],
                                                  "y_axis": [0.0, 1.0, 0.0],
                                                  "bounds": {"x": [-0.026, 0.026], "y": [-0.026, 0.026]}}}}),
        pool("empty_cup", ["bright_cup_purple", "bright_cup_pink", "bright_cup_beige", "bright_cup_blue",
                           "bright_measuring_cup_blue"], ["distractor_cup"]),
        {"id": "beans", "asset": {"type": "asset_id", "id": "bright_bean_red"},
         "instances": {"type": "auto", "count": N_BEANS, "template": "bean_{i}", "start_index": 1},
         "spawn_mass": [0.0003, 0.0005], "semantic_tags": ["content", "beans"]},
    ]
    nodes = {
        "target_bowl": {"object": "target_bowl", "placement": surf([0.04, 0.12], [-0.22, -0.14], surface="tabletop")},
        "bean_cup": {"object": "bean_cup", "placement": surf([0.04, 0.12], [0.08, 0.16], surface="tabletop")},
        "distractor_bowl": {"object": "distractor_bowl", "placement": surf([-0.22, -0.16], [-0.36, -0.20],
                                                                          surface="tabletop")},
        "empty_cup": {"object": "empty_cup", "placement": surf([-0.14, 0.10], [0.30, 0.42], surface="tabletop")},
    }
    groups = {"beans_in_cup": {"type": "region", "nodes": "@instances:beans", "surface": "cup_rim",
                               "ranges": {"x": [-0.024, 0.024], "y": [-0.024, 0.024]}, "yaws": 0.0,
                               "sample_method": "stratified", "spacing_mode": "center_distance",
                               "min_spacing": 0.003, "jitter": 0.2, "max_attempts": 4000,
                               "ignore_intra_group_collision": True}}
    surfaces = {"cup_rim": {"type": "ref", "ref": "bean_cup.rim"}}
    scene = scene_doc(task, "Left hand steadies the blue bowl, right hand pours red beans from the green cup into it "
                            "and puts the cup back", objects, nodes, groups, surfaces)
    sensors = {
        "bowl_left_contact": {"type": "contact", "source": "@target_bowl", "against": LEFT_GRIPPER},
        "cup_right_contact": {"type": "contact", "source": "@bean_cup", "against": RIGHT_GRIPPER},
    }
    beans_in_bowl = {"_func": "objects_count_in_relative_zone", "object_names": beans,
                     "reference_cfg": "@target_bowl", "min_count": (N_BEANS + 1) // 2, "xy_distance_threshold": 0.08,
                     "z_min_offset": -0.01, "z_max_offset": 0.09, "velocity_threshold": 0.3,
                     "consecutive_steps": 10}
    stages = {
        "hold_bowl": (contact("bowl_left_contact", 0.2, 10), "left hand holds the rim of the blue bowl (C1)"),
        "pick_cup": (all_of(right_grasp=contact("cup_right_contact", 0.2, 5),
                            cup_lifted=lifted("@bean_cup", 0.05, sensor="right_gripper_contact")),
                     "right hand picks up the bean cup (G1)"),
        "cup_over_bowl": (all_of(over=zone("@bean_cup", "@target_bowl", 5, xy_distance_threshold=0.13,
                                           z_min_offset=0.06, z_max_offset=0.30),
                                 bowl_held=contact("bowl_left_contact", 0.1, 1)),
                          "cup moved above the bowl (M1)"),
        "pour": (beans_in_bowl, "beans poured into the bowl (L3)"),
        # "back to its original place" ~ upright, on the table, on the right half (y<-0.273 world), released
        "return_cup": (all_of(upright=tilt("@bean_cup", "z", 10, min_dot=0.9, max_dot=1.0),
                              home=zone("@bean_cup", None, 10, vel=0.05, x_range=[0.0, 0.30],
                                        y_range=[-0.50, -0.30], z_range=[TABLE_Z - 0.02, TABLE_Z + 0.03]),
                              released=no_contact(["cup_right_contact"], 0.05, 10),
                              beans_kept={**beans_in_bowl, "consecutive_steps": 1}),
                       "cup put back on its side of the table (P2)"),
    }
    dag = dag_doc(task, "Hold the light-blue bowl steady with the left hand. With the right hand pick up the green "
                        "cup with red beans, move it above the bowl, tilt it to pour the beans into the bowl, then "
                        "put the cup back where it was.",
                  [{"id": s} for s in stages],
                  runtime_assets=["distractor_bowl", "empty_cup"],
                  runtime_nodes=["target_bowl", "bean_cup", "distractor_bowl", "empty_cup"],
                  runtime_groups=["beans_in_cup"])
    write_task(task, sensors, stages, dag, scene, dropped="@pickable")

# ----------------------------------------------------------------------------
# 2. Fold towel once + put into basket (rigid placeholder towel)
# ----------------------------------------------------------------------------

def task_fold_towel():
    task = "fold_towel"
    towels = ["bright_towel_blue", "bright_towel_pink", "bright_towel_yellow", "bright_towel_green",
              "bright_towel_purple"]
    objects = [
        tablecloth(),
        pool("towel", towels, ["pickable", "towel", "cloth_placeholder"]),
        pool("basket", ["bright_basket_green", "bright_basket_blue"], ["container"]),
        pool("distractor_towel", towels, ["distractor"]),
    ]
    nodes = {
        "tablecloth": TABLECLOTH_NODE,
        "towel": {"object": "towel", "placement": surf([0.10, 0.16], [-0.04, 0.04], yaws=[-0.15, 0.15])},
        "basket": {"object": "basket", "placement": surf([-0.16, -0.08], [0.24, 0.32], yaws=1.5708)},
        "distractor_towel": {"object": "distractor_towel",
                             "placement": surf([-0.18, -0.08], [-0.42, -0.28], yaws=[-0.5, 0.5])},
    }
    scene = scene_doc(task, "Fold the towel in half away from the robot, then put it into the basket", objects, nodes)
    sensors = {
        "towel_left_contact": {"type": "contact", "source": "@towel", "against": LEFT_GRIPPER},
        "towel_right_contact": {"type": "contact", "source": "@towel", "against": RIGHT_GRIPPER},
    }
    stages = {
        "grasp_corners": (all_of(left=contact("towel_left_contact", 0.1, 5), right=contact("towel_right_contact", 0.1, 5)),
                          "both grippers pinch the near corners (M3)"),
        "fold_lift": (all_of(left=contact("towel_left_contact", 0.1, 3), right=contact("towel_right_contact", 0.1, 3),
                             raised=lifted("@towel", 0.03, sensor=["left_gripper_contact", "right_gripper_contact"])),
                      "corners lifted and carried away from the robot (D3; rigid towel proxy)"),
        "release": (all_of(flat=tilt("@towel", "z", 10, min_dot=0.9, max_dot=1.0),
                           down=zone("@towel", None, 10, vel=0.05, z_range=[TABLE_Z - 0.02, TABLE_Z + 0.03]),
                           free=no_contact(["towel_left_contact", "towel_right_contact"], 0.05, 10)),
                    "towel put down and released (P2)"),
        "pick_folded": (all_of(raised=lifted("@towel", 0.05, sensor=["left_gripper_contact", "right_gripper_contact"])),
                        "folded towel picked up (G1)"),
        "into_basket": (all_of(inside=zone("@towel", "@container", 10, vel=0.05, reference_footprint_zone=True,
                                           xy_inset_ratio=0.05, z_range=[-0.02, 0.12]),
                               free=no_contact(["towel_left_contact", "towel_right_contact"], 0.05, 10)),
                        "towel dropped into the basket (P1)"),
    }
    dag = dag_doc(task, "Grasp the two near corners of the towel with both hands, fold it once away from you, put it "
                        "down, then pick the folded towel up and drop it into the basket.",
                  [{"id": s} for s in stages],
                  runtime_assets=["tablecloth", "towel", "basket", "distractor_towel"],
                  runtime_nodes=["tablecloth", "towel", "basket", "distractor_towel"])
    write_task(task, sensors, stages, dag, scene, dropped="@pickable")


# ----------------------------------------------------------------------------
# 3. Stack rings large -> small onto the right peg while the left hand holds the base
# ----------------------------------------------------------------------------
PEG_Y = 0.10      # target peg in base-local +y
BASE_H = 0.02
RING_T = 0.02


def task_stack_rings():
    task = "stack_rings"
    objects = [
        tablecloth(),
        fixed("stacker", "bright_stacker_base", ["destination", "stacker"]),
        fixed("ring_large", "bright_ring_large", ["pickable", "ring_1"]),
        fixed("ring_medium", "bright_ring_medium", ["pickable", "ring_2"]),
        fixed("ring_small", "bright_ring_small", ["pickable", "ring_3"]),
        fixed("distractor_stacker", "bright_stacker_round", ["distractor"]),
    ]
    nodes = {
        "tablecloth": TABLECLOTH_NODE,
        # world-frame yaw ~pi so the base-local +y (red-capped) peg sits on the robot's right
        "stacker": {"object": "stacker", "placement": surf([-0.06, -0.01], [-0.04, 0.04], yaws=[3.06, 3.22])},
        "distractor_stacker": {"object": "distractor_stacker", "placement": surf([-0.20, -0.14], [-0.44, -0.32])},
    }
    groups = {"rings_region": slot_group(["ring_large", "ring_medium", "ring_small"],
                                         [([0.10, 0.22], [0.05, 0.13]), ([0.10, 0.22], [0.17, 0.25]),
                                          ([0.10, 0.22], [0.29, 0.37])], yaws=[-3.14, 3.14])}
    scene = scene_doc(task, "Hold the long base with the left hand and stack the three rings large-to-small on the "
                            "right peg", objects, nodes, groups)
    sensors = {"base_left_contact": {"type": "contact", "source": "@stacker", "against": LEFT_GRIPPER}}
    for k in (1, 2, 3):
        sensors[f"ring{k}_right_contact"] = {"type": "contact", "source": f"@ring_{k}", "against": RIGHT_GRIPPER}
    stages = {"press_base": (contact("base_left_contact", 0.5, 10), "left hand presses the base end (C1)")}
    for k, name in ((1, "large"), (2, "medium"), (3, "small")):
        ring = f"@ring_{k}"
        z_on = BASE_H + RING_T * (k - 1)
        stages[f"pick_{name}"] = (all_of(grasp=contact(f"ring{k}_right_contact", 0.1, 3),
                                         raised=lifted(ring, 0.03, sensor="right_gripper_contact")),
                                  f"right hand picks the {name} ring (G1)")
        stages[f"{name}_over_peg"] = (zone(ring, "@destination", 3, in_reference_frame=True,
                                           x_range=[-0.02, 0.02], y_range=[PEG_Y - 0.02, PEG_Y + 0.02],
                                           z_range=[z_on + 0.01, 0.30]),
                                      f"{name} ring aligned above the peg (M1/M3)")
        stages[f"{name}_on_peg"] = (all_of(on_peg=zone(ring, "@destination", 10, vel=0.05, in_reference_frame=True,
                                                       x_range=[-0.012, 0.012],
                                                       y_range=[PEG_Y - 0.012, PEG_Y + 0.012],
                                                       z_range=[z_on - 0.008, z_on + 0.008]),
                                           released=no_contact([f"ring{k}_right_contact"], 0.05, 10)),
                                    f"{name} ring dropped onto the peg (P3)")
    dag = dag_doc(task, "Press the left end of the wooden base with the left hand. With the right hand put the rings "
                        "onto the right peg from the largest to the smallest.",
                  [{"id": s} for s in stages], runtime_assets=["tablecloth"],
                  runtime_nodes=["tablecloth", "stacker", "distractor_stacker"], runtime_groups=["rings_region"])
    write_task(task, sensors, stages, dag, scene)


# ----------------------------------------------------------------------------
# 4. Hand over tools: left picks, hands to right in the middle, right places into slot
# ----------------------------------------------------------------------------
SLOT_Y = 0.048


def task_handover_tools():
    task = "handover_tools"
    objects = [
        tablecloth(),
        fixed("toolbox", "bright_toolbox", ["container", "toolbox"]),
        fixed("hammer", "bright_hammer", ["pickable", "tool_1"]),
        fixed("wrench", "bright_wrench", ["pickable", "tool_2"]),
        fixed("screwdriver", "bright_screwdriver", ["distractor"]),
    ]
    nodes = {
        "tablecloth": TABLECLOTH_NODE,
        "toolbox": {"object": "toolbox", "placement": surf([-0.04, 0.04], [0.24, 0.32], yaws=[-0.1, 0.1])},
    }
    groups = {"tools_region": slot_group(["hammer", "wrench", "screwdriver"],
                                         [([0.06, 0.16], [-0.44, -0.38]), ([0.06, 0.16], [-0.31, -0.25]),
                                          ([0.06, 0.16], [-0.18, -0.12])], yaws=[-0.15, 0.15])}
    scene = scene_doc(task, "Left hand picks a tool, hands it over at the table centre, right hand puts it into the "
                            "matching toolbox slot", objects, nodes, groups)
    sensors = {}
    for k in (1, 2):
        sensors[f"tool{k}_left_contact"] = {"type": "contact", "source": f"@tool_{k}", "against": LEFT_GRIPPER}
        sensors[f"tool{k}_right_contact"] = {"type": "contact", "source": f"@tool_{k}", "against": RIGHT_GRIPPER}
    stages = {}
    for k, name, slot in ((1, "hammer", [-SLOT_Y - 0.04, -SLOT_Y + 0.04]), (2, "wrench", [SLOT_Y - 0.04, SLOT_Y + 0.04])):
        tool = f"@tool_{k}"
        stages[f"pick_{name}"] = (all_of(grasp=contact(f"tool{k}_left_contact", 0.1, 3),
                                         raised=lifted(tool, 0.04, sensor="left_gripper_contact")),
                                  f"left hand picks the {name} (G1)")
        # table middle (env frame): world x 0.05..0.40, y -0.40..-0.14 (table centre y=-0.273), lifted
        stages[f"{name}_to_middle"] = (all_of(mid=zone(tool, None, 5, x_range=[0.0, 0.40], y_range=[-0.40, -0.14],
                                                       z_range=[TABLE_Z + 0.05, TABLE_Z + 0.45]),
                                              held=contact(f"tool{k}_left_contact", 0.1, 1)),
                                       f"{name} carried to the handover point (M1)")
        stages[f"{name}_handover"] = (all_of(right=contact(f"tool{k}_right_contact", 0.1, 5),
                                             left_free=no_contact([f"tool{k}_left_contact"], 0.05, 5),
                                             airborne=zone(tool, None, 5, z_range=[TABLE_Z + 0.03, TABLE_Z + 0.5])),
                                      f"right hand takes the {name}, left releases (C2)")
        stages[f"{name}_over_slot"] = (zone(tool, "@toolbox", 3, in_reference_frame=True, x_range=[-0.12, 0.12],
                                            y_range=slot, z_range=[0.04, 0.30]),
                                       f"{name} aligned above its slot (M3)")
        stages[f"{name}_in_slot"] = (all_of(inside=zone(tool, "@toolbox", 10, vel=0.05, in_reference_frame=True,
                                                        x_range=[-0.13, 0.13], y_range=slot, z_range=[-0.005, 0.05]),
                                            free=no_contact([f"tool{k}_right_contact"], 0.05, 10)),
                                     f"{name} placed in its slot (P2)")
    dag = dag_doc(task, "With the left hand pick up the hammer, hand it to the right hand at the middle of the table, "
                        "and put it into the red-labelled slot of the toolbox; then do the same with the wrench and "
                        "the blue-labelled slot.",
                  [{"id": s} for s in stages], runtime_assets=["tablecloth"],
                  runtime_nodes=["tablecloth", "toolbox"], runtime_groups=["tools_region"])
    write_task(task, sensors, stages, dag, scene)


# ----------------------------------------------------------------------------
# 5. Arrange flowers: left picks the stem, right takes the head, inserts into vase
# ----------------------------------------------------------------------------

def task_arrange_flowers():
    task = "arrange_flowers"
    objects = [
        tablecloth(),
        pool("vase", ["bright_vase_white", "bright_vase_celadon"], ["container", "vase"]),
        fixed("flower_1", "bright_flower_rose_red", ["pickable", "flower_1"]),
        fixed("flower_2", "bright_flower_rose_pink", ["pickable", "flower_2"]),
        fixed("flower_3", "bright_flower_rose_purple", ["distractor"]),
    ]
    head_away = [0.0, 0.0, 0.0, 1.0]   # orientations are world-frame: authored +X head already points away (world +x)
    nodes = {
        "tablecloth": TABLECLOTH_NODE,
        "vase": {"object": "vase", "placement": surf([-0.06, 0.04], [0.18, 0.30])},
    }
    for k, yr in ((1, [-0.46, -0.38]), (2, [-0.33, -0.25]), (3, [-0.20, -0.12])):
        nodes[f"flower_{k}"] = {"object": f"flower_{k}", "placement": surf([-0.03, 0.07], yr, yaws=[-0.15, 0.15],
                                                                          orientation=head_away)}
    groups = None
    scene = scene_doc(task, "Insert the red and the pink crochet roses stem-first into the vase", objects, nodes,
                      groups)
    sensors = {}
    for k in (1, 2):
        sensors[f"flower{k}_left_contact"] = {"type": "contact", "source": f"@flower_{k}", "against": LEFT_GRIPPER}
        sensors[f"flower{k}_right_contact"] = {"type": "contact", "source": f"@flower_{k}", "against": RIGHT_GRIPPER}
        sensors[f"flower{k}_vase_contact"] = {"type": "contact", "source": f"@flower_{k}", "against": "@vase"}
    stages = {}
    for k, name in ((1, "red"), (2, "pink")):
        fl = f"@flower_{k}"
        stages[f"pick_{name}"] = (all_of(grasp=contact(f"flower{k}_left_contact", 0.05, 3),
                                         raised=lifted(fl, 0.04, sensor="left_gripper_contact")),
                                  f"left hand picks the {name} rose by the stem (G1)")
        stages[f"{name}_to_middle"] = (all_of(mid=zone(fl, None, 5, x_range=[0.0, 0.40], y_range=[-0.40, -0.14],
                                                       z_range=[TABLE_Z + 0.05, TABLE_Z + 0.50]),
                                              held=contact(f"flower{k}_left_contact", 0.05, 1)),
                                       f"{name} rose lifted to the table centre (M1)")
        stages[f"{name}_handover"] = (all_of(right=contact(f"flower{k}_right_contact", 0.05, 5),
                                             left_free=no_contact([f"flower{k}_left_contact"], 0.05, 5)),
                                      f"right hand takes the {name} rose head, left releases (C2)")
        stages[f"{name}_over_vase"] = (all_of(above=zone(fl, "@vase", 3, xy_distance_threshold=0.05,
                                                         z_min_offset=0.10, z_max_offset=0.45),
                                              vertical=tilt(fl, "x", 3, min_abs_dot=0.8)),
                                       f"{name} stem vertical above the vase mouth (M3)")
        stages[f"{name}_inserted"] = (all_of(touch=contact(f"flower{k}_vase_contact", 0.01, 5),
                                             inside=zone(fl, "@vase", 5, vel=0.5, reference_footprint_zone=True,
                                                         xy_inset_ratio=0.1, z_range=[-0.03, 0.35]),
                                             upright=tilt(fl, "x", 5, min_abs_dot=0.6),
                                             free=no_contact([f"flower{k}_right_contact"], 0.05, 5)),
                                      f"{name} rose inserted and released (P1)")
    dag = dag_doc(task, "With the left hand pick up the red rose by its stem, hand it to the right hand which holds "
                        "the flower head, and insert the stem into the vase; then do the same with the pink rose. "
                        "Leave the purple rose on the table.",
                  [{"id": s} for s in stages], runtime_assets=["tablecloth", "vase"],
                  runtime_nodes=["tablecloth", "vase", "flower_1", "flower_2", "flower_3"])
    write_task(task, sensors, stages, dag, scene)


TASKS = {
    "pour_beans": task_pour_beans,
    "fold_towel": task_fold_towel,
    "stack_rings": task_stack_rings,
    "handover_tools": task_handover_tools,
    "arrange_flowers": task_arrange_flowers,
}


def env_name(task):
    return f"artixon_arm6a_G_bright_{task}_id_env"


# ----------------------------------------------------------------------------
# Registration (idempotent text patches; keeps upstream formatting intact)
# ----------------------------------------------------------------------------
BEGIN, END = "# >>> bright2026 begin", "# <<< bright2026 end"


def _strip_block(text: str) -> str:
    return re.sub(rf"\n?{re.escape(BEGIN)}.*?{re.escape(END)}\n?", "\n", text, flags=re.S)


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
