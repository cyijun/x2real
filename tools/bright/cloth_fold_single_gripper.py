"""Fold a MuJoCo flex towel with two pose-controlled grippers and transfer it with one.

Run with MuJoCo 3.11 and NumPy. --video adds an MJPEG recording via OpenCV.
A closed jaw attaches selected vertices within its capture radius. Gripper meshes
serve as visual markers. Table, basket and cloth self-contact use the solver.
"""
import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np


N = 11
CORNERS = [0, N - 1, N * (N - 1), N * N - 1]
# Two matching layers on the right edge, 6 cm from the initially grasped corner.
LAYERS = [2 * N, (N - 3) * N]
CAPTURE_RADIUS = 0.012
BASKET_X = 0.40
BASKET_Y = 0.024


def make_model():
    grippers = []
    for name, y in (("right", -.15), ("left", .15)):
        fingers = [
            f'<geom name="{name}_{side}" type="box" pos="0 {side * .016} .012" '
            'size=".008 .003 .012" rgba=".21 .25 .30 1" contype="0" conaffinity="0"/>'
            for side in (-1, 1)
        ]
        grippers.append(f'<body name="{name}" mocap="true" pos="-.15 {y} .10">{"".join(fingers)}</body>')
    pins = [(CORNERS[0], "right"), (CORNERS[1], "left"), (LAYERS[0], "right"), (LAYERS[1], "right")]
    constraints = [
        f'<connect name="pin_{i}" body1="towel_{i}" body2="{jaw}" anchor="0 0 0" active="false" solref=".004 1"/>'
        for i, jaw in pins
    ]
    walls = [
        (0, 0, .002, .125, .165, .002), (-.122, 0, .044, .002, .164, .042),
        (.122, 0, .044, .002, .164, .042), (0, -.162, .044, .12, .002, .042),
        (0, .162, .044, .12, .002, .042),
    ]
    basket = [
        f'<geom type="box" pos="{BASKET_X+x} {BASKET_Y+y} {z}" size="{sx} {sy} {sz}" rgba=".38 .62 .42 1"/>'
        for x, y, z, sx, sy, sz in walls
    ]
    xml = f'''<mujoco>
    <option timestep=".002" integrator="implicitfast" solver="Newton" iterations="40" tolerance="1e-8"/>
    <size memory="128M"/><default><geom friction=".9 .005 .001" solref=".008 1"/></default>
    <worldbody><light pos="0 -.3 1.4"/><geom type="plane" size="1 1 .02" rgba=".80 .77 .70 1"/>
    {"".join(basket)}{"".join(grippers)}
    <flexcomp name="towel" dim="2" type="grid" count="{N} {N} 1" spacing=".03 .03 .02" pos="0 0 .008"
      radius=".0022" mass=".070" rgba=".30 .48 .80 1">
      <edge equality="true" solref=".006 1"/>
      <elasticity young="30000" poisson=".25" thickness=".002" elastic2d="bend"/>
      <contact selfcollide="auto" friction=".95 .005 .001" solref=".008 1" condim="3"/>
    </flexcomp></worldbody><equality>{"".join(constraints)}</equality></mujoco>'''
    return mujoco.MjModel.from_xml_string(xml), pins


class Trial:
    def __init__(self, output, video):
        self.model, self.pins = make_model()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.attachments = []
        self.samples = []
        self.writer = None
        if video:
            import cv2
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.camera = mujoco.MjvCamera()
            self.camera.lookat[:] = [.15, 0, .15]
            self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.12, 219, -38
            self.opt = mujoco.MjvOption()
            self.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = 0
            self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = 0
            self.writer = cv2.VideoWriter(str(output / "trajectory.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 25, (640, 480))
            if not self.writer.isOpened():
                raise RuntimeError("Could not open MJPEG output")

    def move(self, targets, seconds):
        start = self.data.mocap_pos.copy()
        count = round(seconds / self.model.opt.timestep)
        for tick in range(1, count + 1):
            u = tick / count
            self.data.mocap_pos[:] = start + (targets - start) * u**2 * (3 - 2*u)
            mujoco.mj_step(self.model, self.data)
            self.steps += 1
            if self.writer and self.steps % 20 == 0:
                self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.opt)
                self.writer.write(self.renderer.render()[:, :, ::-1].copy())

    def hold(self, seconds):
        self.move(self.data.mocap_pos.copy(), seconds)

    def jaw(self, name, nodes=()):
        jaw_id = self.model.body(name).mocapid[0]
        for i, (node, hand) in enumerate(self.pins):
            if hand != name:
                continue
            self.data.eq_active[i] = 0
            if node in nodes:
                delta = self.data.flexvert_xpos[node] - self.data.mocap_pos[jaw_id]
                distance = float(np.linalg.norm(delta))
                if distance > CAPTURE_RADIUS:
                    raise ValueError(f"{name} jaw missed vertex {node} by {distance:.4f} m")
                self.model.eq_data[i, :3] = 0
                self.model.eq_data[i, 3:6] = delta
                self.data.eq_active[i] = 1
                self.attachments.append(dict(time_s=float(self.data.time), jaw=name, node=node, distance_m=distance))
        for side in (-1, 1):
            self.model.geom(f"{name}_{side}").pos[1] = side * (.005 if nodes else .016)

    def measure(self, phase):
        p = self.data.flexvert_xpos.copy()
        c = p[CORNERS]
        in_basket = ((abs(p[:, 0] - BASKET_X) < .12) & (abs(p[:, 1] - BASKET_Y) < .16)
                     & (p[:, 2] > .001) & (p[:, 2] < .086))
        active = [jaw for i, (_, jaw) in enumerate(self.pins) if self.data.eq_active[i]]
        row = dict(phase=phase, time_s=round(self.data.time, 4), active_jaws=sorted(set(active)),
                   pinned_nodes=len(active), extent_m=np.ptp(p, axis=0).tolist(),
                   near_far_gap_m=np.linalg.norm(c[:2] - c[2:], axis=1).tolist(),
                   lowest_z_m=float(p[:, 2].min()), inside_vertices=int(in_basket.sum()),
                   total_vertices=len(p), inside_fraction=float(in_basket.mean()))
        self.samples.append(row)
        return row

    def run(self):
        self.hold(.35)
        c = self.data.flexvert_xpos[CORNERS].copy()
        near, far = c[:2], c[2:]
        self.move(near, .40)
        self.jaw("right", CORNERS[:1])
        self.jaw("left", CORNERS[1:2])
        for theta in np.linspace(0, np.pi, 45)[1:]:
            u = (1-np.cos(theta))/2
            point = (1-u)*near + u*far
            point[:, 2] += .122*np.sin(theta) + .006*u
            self.move(point, .035)
            if np.isclose(theta, np.pi/2):
                self.measure("lift_near_edge")
        self.jaw("left")
        self.jaw("right")
        self.move(self.data.mocap_pos + [0, 0, .09], .30)
        self.hold(.75)
        folded = self.measure("fold_released")

        right = self.model.body("right").mocapid[0]
        left = self.model.body("left").mocapid[0]
        targets = self.data.mocap_pos.copy()
        targets[left] = [-.17, .24, .14]
        targets[right] = self.data.flexvert_xpos[LAYERS].mean(axis=0) + [0, 0, .06]
        self.move(targets, .65)
        targets[right] = self.data.flexvert_xpos[LAYERS].mean(axis=0)
        self.move(targets, .35)
        self.jaw("right", LAYERS)
        targets[right, 2] += .45
        self.move(targets, 1.15)
        lift = self.measure("single_gripper_lift")
        targets[right, :2] = [BASKET_X, .115]
        self.move(targets, 1.35)
        self.hold(.55)
        self.measure("over_basket")
        # First settle the free portion on the basket floor, then release the two layers.
        targets[right, 2] = .24
        self.move(targets, .85)
        targets[right] += [.008, -.245, -.22]
        self.move(targets, 1.9)
        self.jaw("right")
        targets[right, 2] += .14
        self.move(targets, .55)
        self.hold(.95)
        final = self.measure("basket_released")
        basket_ok = final["inside_fraction"] >= .85 and not final["active_jaws"]
        return dict(
            fold_ok=(not folded["active_jaws"] and max(folded["near_far_gap_m"]) < .05
                     and .10 < folded["extent_m"][0] < .20 and .25 < folded["extent_m"][1] < .34
                     and folded["extent_m"][2] < .05),
            single_gripper_lift_ok=(lift["active_jaws"] == ["right"] and lift["lowest_z_m"] > .086),
            basket_ok=basket_ok,
            fold_retained_in_basket=(basket_ok and max(final["near_far_gap_m"]) < .05
                                     and .10 < final["extent_m"][0] < .20
                                     and .25 < final["extent_m"][1] < .34),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trial = Trial(args.out, args.video)
    started = time.monotonic()
    try:
        checks = trial.run()
    finally:
        if trial.writer:
            trial.writer.release()
            trial.renderer.close()
    metrics = dict(mujoco_version=mujoco.__version__, checks=checks, steps=trial.steps,
                   time_step_s=trial.model.opt.timestep, wall_s=time.monotonic()-started,
                   rendered=args.video, capture_radius_m=CAPTURE_RADIUS, attachments=trial.attachments, phases=trial.samples)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
