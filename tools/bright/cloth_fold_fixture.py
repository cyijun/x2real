"""Reproduce a two-gripper towel fold with MuJoCo 3.11 on CPU.

The two end effectors have commanded poses. A closed jaw may attach nearby mesh
corners using equality constraints. All other nodes, all releases, the table
and the basket are simulated. This is a fixture test, independent of robot IK.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np


def build_model(n):
    last = n - 1
    pairs = [(0, "R"), (last, "L"), (last * n, "R"), (n * n - 1, "L")]
    jaws = []
    for hand, y in [("R", -0.15), ("L", 0.15)]:
        jaws.append(f'<body name="{hand}" mocap="true" pos="-.15 {y} .10">')
        for side in (-1, 1):
            jaws.append(
                f'<geom name="{hand}_{side}" type="box" pos="0 {side * 0.016} .012" size=".008 .003 .012" rgba=".21 .25 .30 1" contype="0" conaffinity="0"/>'
            )
        jaws.append("</body>")
    ties = "".join(
        f'<connect name="pin{i}" body1="towel_{i}" body2="{hand}" anchor="0 0 0" active="false" solref=".004 1"/>'
        for i, hand in pairs
    )
    # Interior spans 24 cm along x and 32 cm along y. Keep its physical rim.
    basket = []
    for x, y, z, sx, sy, sz in [
        (0, 0, 0.002, 0.125, 0.165, 0.002),
        (-0.122, 0, 0.044, 0.002, 0.164, 0.042),
        (0.122, 0, 0.044, 0.002, 0.164, 0.042),
        (0, -0.162, 0.044, 0.12, 0.002, 0.042),
        (0, 0.162, 0.044, 0.12, 0.002, 0.042),
    ]:
        basket.append(
            f'<geom type="box" pos="{0.40 + x} {y} {z}" size="{sx} {sy} {sz}" rgba=".38 .62 .42 1"/>'
        )
    spacing = 0.30 / last
    xml = f'''<mujoco><option timestep=".002" integrator="implicitfast" solver="Newton" iterations="40" tolerance="1e-8"/>
    <size memory="128M"/><default><geom friction=".9 .005 .001" solref=".008 1"/></default>
    <worldbody><light pos="0 -.3 1.4"/><geom type="plane" size="1 1 .02" rgba=".80 .77 .70 1"/>
    {"".join(basket)}{"".join(jaws)}
    <flexcomp name="towel" dim="2" type="grid" count="{n} {n} 1" spacing="{spacing} {spacing} .02" pos="0 0 .008" radius=".0022" mass=".070" rgba=".30 .48 .80 1">
    <edge equality="true" solref=".006 1"/><elasticity young="30000" poisson=".25" thickness=".002" elastic2d="bend"/>
    <contact selfcollide="auto" friction=".95 .005 .001" solref=".008 1" condim="3"/></flexcomp>
    </worldbody><equality>{ties}</equality></mujoco>'''
    return mujoco.MjModel.from_xml_string(xml), pairs


def run(out, n, render):
    out.mkdir(parents=True, exist_ok=True)
    model, pairs = build_model(n)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    corners = np.array([i for i, _ in pairs])
    starts = data.flexvert_xpos[corners].copy()
    # Use the mesh's rest coordinates to verify node order for this MuJoCo build.
    assert np.all(starts[:2, 0] < 0) and np.all(starts[2:, 0] > 0) and starts[0, 1] < 0
    renderer = writer = None
    if render:
        import cv2

        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.15, 0, 0.035]
        cam.distance = 0.95
        cam.azimuth = 219
        cam.elevation = -44
        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = 0
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = 0
        writer = cv2.VideoWriter(
            str(out / "fixture.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 25, (640, 480)
        )
        if not writer.isOpened():
            raise RuntimeError("Cannot open MJPEG video writer")
    attachments = []
    samples = []
    steps = 0

    def sim_step():
        nonlocal steps
        mujoco.mj_step(model, data)
        steps += 1
        if steps % 20 == 0 and renderer:
            renderer.update_scene(data, camera=cam, scene_option=opt)
            writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    def move(target, duration):
        start = data.mocap_pos.copy()
        ticks = max(1, round(duration / model.opt.timestep))
        for j in range(ticks):
            u = (j + 1) / ticks
            u = u * u * (3 - 2 * u)
            data.mocap_pos[:] = start + (target - start) * u
            sim_step()

    def hold(seconds):
        move(data.mocap_pos.copy(), seconds)

    def command_grip(close, indices=()):
        wanted = set(indices)
        for k, (i, hand) in enumerate(pairs):
            moc = model.body(hand).mocapid[0]
            delta = data.flexvert_xpos[i] - data.mocap_pos[moc]
            if close and i in wanted:
                if np.linalg.norm(delta) > 0.023:
                    raise ValueError(f"corner {i} outside jaw capture radius: {delta}")
                model.eq_data[k, :3] = 0
                model.eq_data[k, 3:6] = delta
                data.eq_active[k] = 1
                attachments.append(
                    dict(
                        time=float(data.time),
                        node=i,
                        distance_m=float(np.linalg.norm(delta)),
                    )
                )
            elif not close:
                data.eq_active[k] = 0
        for hand in ("R", "L"):
            for side in (-1, 1):
                model.geom(f"{hand}_{side}").pos[1] = side * (0.005 if close else 0.016)

    def measure(label):
        p = data.flexvert_xpos.copy()
        c = p[corners]
        inside = (
            (abs(p[:, 0] - 0.40) < 0.12)
            & (abs(p[:, 1]) < 0.16)
            & (p[:, 2] > 0.001)
            & (p[:, 2] < 0.086)
        )
        row = dict(
            phase=label,
            time_s=round(float(data.time), 3),
            corner_gap_m=np.linalg.norm(c[:2] - c[2:], axis=1).round(5).tolist(),
            extent_m=np.ptp(p, axis=0).round(5).tolist(),
            near_z_m=c[:2, 2].round(5).tolist(),
            far_z_m=c[2:, 2].round(5).tolist(),
            inside_fraction=float(inside.mean()),
            pins=int(np.sum(data.eq_active[:4])),
        )
        samples.append(row)
        return row

    tic = time.monotonic()
    hold(0.35)
    near = data.flexvert_xpos[corners[:2]].copy()
    far = data.flexvert_xpos[corners[2:]].copy()
    move(near, 0.4)
    command_grip(True, corners[:2])
    for theta in np.linspace(0, np.pi, 45)[1:]:
        u = (1 - np.cos(theta)) / 2
        target = (1 - u) * near + u * far
        target[:, 2] += 0.122 * np.sin(theta) + 0.006 * u
        move(target, 0.035)
        if 1.5 < theta < 1.59:
            measure("lift_near_edge")
    command_grip(False)
    move(data.mocap_pos + [0, 0, 0.09], 0.30)
    hold(0.75)
    folded = measure("fold_released")
    p = data.flexvert_xpos[corners].copy()
    grip_at = (p[:2] + p[2:]) / 2
    move(grip_at, 0.5)
    command_grip(True, corners)
    move(grip_at + [0, 0, 0.22], 0.85)
    measure("regrasp_lift")
    # Carry the hanging folded panel over the rim, then lay it down from its free edge.
    top = data.mocap_pos.copy()
    top[:, 0] = 0.345
    move(top, 1.0)
    for u in np.linspace(0, 1, 32)[1:]:
        dst = top.copy()
        dst[:, 0] += 0.125 * u
        dst[:, 2] = top[:, 2] * (1 - u) + 0.014 * u
        move(dst, 0.045)
    command_grip(False)
    move(data.mocap_pos + [0, 0, 0.10], 0.40)
    hold(0.8)
    final = measure("basket_released")
    wall = time.monotonic() - tic
    if writer:
        writer.release()
        renderer.close()
    fold_ok = (
        max(folded["corner_gap_m"]) < 0.05
        and 0.10 < folded["extent_m"][0] < 0.20
        and 0.25 < folded["extent_m"][1] < 0.34
        and folded["extent_m"][2] < 0.05
    )
    result = dict(
        n=n,
        threshold_m=0.023,
        attachments=attachments,
        stages=samples,
        fold_ok=bool(fold_ok),
        basket_ok=final["inside_fraction"] >= 0.85,
        steps=steps,
        wall_s=wall,
        steps_per_wall_second=steps / wall,
        rendered=render,
    )
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--n", type=int, default=11)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    run(args.out, args.n, args.render)
