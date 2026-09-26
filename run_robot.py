"""Simulate, inspect, and walk the compact quadruped. Run --help for options."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from queue import SimpleQueue
import struct
import time
import zlib

import mujoco
import numpy as np

from build_robot import CAD_DIR, CONFIG, LEGS, ROOT, load_config, mesh_assets, model_xml


class Robot:
    def __init__(self, cfg=None, cad_dir=CAD_DIR):
        self.cfg = load_config() if cfg is None else cfg
        # Build in memory every launch, so stale generated XML cannot override JSON.
        self.model = mujoco.MjModel.from_xml_string(model_xml(self.cfg), assets=mesh_assets(self.cfg, cad_dir))
        self.data = mujoco.MjData(self.model)
        self.jids = np.array([self.model.joint(f"{name}_{kind}").id
                              for name, _, _ in LEGS for kind in ("swing", "lift")])
        self.qadr = self.model.jnt_qposadr[self.jids]
        self.vadr = self.model.jnt_dofadr[self.jids]
        self.foot_ids = {self.model.geom(f"{name}_foot").id for name, _, _ in LEGS}
        self.ground_id = self.model.geom("ground").id
        self.body_id = self.model.body("chassis").id
        self.neutral = np.tile([0., math.radians(self.cfg["leg"]["stance_deg"])], 4)
        self.target = self.neutral.copy()
        self.mode = "stand"
        self.mode_start = 0.
        self.transition_from = self.neutral.copy()
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.mode = "stand"
        self.mode_start = 0.
        self.target = self.neutral.copy()
        self.transition_from = self.neutral.copy()
        mujoco.mj_forward(self.model, self.data)

    def set_mode(self, mode):
        if mode not in ("stand", "walk"):
            raise ValueError("Mode must be stand or walk")
        self.transition_from = self.target.copy()
        self.mode, self.mode_start = mode, self.data.time

    def targets(self):
        g = self.cfg["gait"]
        elapsed = max(0., self.data.time - self.mode_start)
        target = self.neutral.copy()
        if self.mode == "walk":
            # One foot swings at a time: FL -> RR -> FR -> RL.
            offsets = (0., .5, .75, .25)
            swing_fraction = 1 - g["duty_factor"]
            for i, (_, _, side) in enumerate(LEGS):
                phase = (elapsed / g["period_s"] - offsets[i]) % 1.
                if phase < swing_fraction:
                    u = phase / swing_fraction
                    forward = -math.cos(math.pi * u)
                    lift = math.sin(math.pi * u) ** 2
                else:
                    u = (phase - swing_fraction) / g["duty_factor"]
                    forward = 1 - 2 * u
                    lift = 0.
                target[2 * i] = -side * math.radians(g["sweep_deg"]) * forward
                target[2 * i + 1] -= math.radians(g["lift_deg"]) * lift
        blend = min(1., elapsed / g["ramp_s"])
        blend = blend * blend * (3 - 2 * blend)
        return (1 - blend) * self.transition_from + blend * target

    def step(self):
        self.target = self.targets()
        q, v = self.data.qpos[self.qadr], self.data.qvel[self.vadr]
        sim, servo = self.cfg["simulation"], self.cfg["servo"]
        desired = sim["kp"] * (self.target - q) - sim["kd"] * v
        # Linear torque-speed envelope when driving; full bounded braking torque.
        limit = np.full(8, servo["torque_limit_nm"])
        driving = desired * v > 0
        limit[driving] *= np.clip(1 - np.abs(v[driving]) / math.radians(servo["speed_deg_s"]), 0, 1)
        self.data.ctrl[:] = np.clip(desired, -limit, limit)
        mujoco.mj_step(self.model, self.data)

    def ground_contacts(self):
        feet, other = set(), set()
        for c in self.data.contact:
            if self.ground_id in (c.geom1, c.geom2):
                gid = c.geom2 if c.geom1 == self.ground_id else c.geom1
                (feet if gid in self.foot_ids else other).add(gid)
        return feet, other


def trial(cfg, mode, seconds):
    robot = Robot(cfg)
    robot.set_mode(mode)
    start = robot.data.qpos[:3].copy()
    max_torque = max_tilt = max_penetration = 0.
    minimum_height = float(start[2])
    foot_steps = {name: 0 for name, _, _ in LEGS}
    nonfoot_steps = 0
    finite = True
    max_tracking_error = 0.
    for _ in range(math.ceil(seconds / robot.model.opt.timestep)):
        robot.step()
        if not np.isfinite(robot.data.qpos).all() or not np.isfinite(robot.data.qvel).all():
            finite = False
            break
        max_torque = max(max_torque, float(np.max(np.abs(robot.data.ctrl))))
        up = robot.data.xmat[robot.body_id].reshape(3, 3)[2, 2]
        max_tilt = max(max_tilt, math.degrees(math.acos(float(np.clip(up, -1, 1)))))
        minimum_height = min(minimum_height, float(robot.data.qpos[2]))
        feet, other = robot.ground_contacts()
        nonfoot_steps += bool(other)
        for name in foot_steps:
            foot_steps[name] += robot.model.geom(f"{name}_foot").id in feet
        for c in robot.data.contact:
            max_penetration = max(max_penetration, float(-c.dist))
        max_tracking_error = max(max_tracking_error, float(np.max(np.abs(robot.target - robot.data.qpos[robot.qadr]))))
    delta = robot.data.qpos[:3] - start
    return {
        "mode": mode, "seconds": float(robot.data.time), "finite": finite,
        "forward_m": float(delta[0]), "sideways_m": float(delta[1]),
        "minimum_body_height_m": minimum_height, "maximum_tilt_deg": max_tilt,
        "maximum_torque_nm": max_torque, "maximum_penetration_m": max_penetration,
        "maximum_tracking_error_deg": math.degrees(max_tracking_error),
        "nonfoot_ground_contact_steps": nonfoot_steps,
        "foot_contact_steps": foot_steps, "warnings": robot.data.warning.number.tolist(),
    }


def check(cfg):
    robot = Robot(cfg)
    if robot.model.nu != 8 or robot.model.njnt != 9:
        raise AssertionError("Expected eight driven joints and one free base")
    # Hold the base for the geometry sweep only; trials use unconstrained physics.
    robot.set_mode("walk")
    collisions = set()
    fixed = ["deck", "battery_and_controller"] + [f"{name}_hip_servo" for name, _, _ in LEGS]
    moving = [f"{name}_{part}" for name, _, _ in LEGS for part in ("lift_servo", "cradle", "shin", "foot")]
    for t in np.linspace(cfg["gait"]["ramp_s"], cfg["gait"]["ramp_s"] + cfg["gait"]["period_s"], 241):
        robot.data.time = t
        target = robot.targets()
        if np.any(target < robot.model.jnt_range[robot.jids, 0]) or np.any(target > robot.model.jnt_range[robot.jids, 1]):
            raise AssertionError("Gait target outside joint range")
        robot.data.qpos[robot.qadr] = target
        mujoco.mj_forward(robot.model, robot.data)
        for c in robot.data.contact:
            if robot.ground_id not in (c.geom1, c.geom2) and c.dist < -.0005:
                collisions.add(tuple(sorted((robot.model.geom(c.geom1).name, robot.model.geom(c.geom2).name))))
        # MuJoCo normally filters parent-child contacts. Explicitly inspect the
        # moving envelopes against the chassis too; intentional horn joints excluded.
        for a in moving:
            for b in fixed:
                distance = mujoco.mj_geomDistance(robot.model, robot.data,
                                                  robot.model.geom(a).id, robot.model.geom(b).id, .01, None)
                if distance < -.0005:
                    collisions.add(tuple(sorted((a, b))))
    cases = [trial(cfg, "stand", 6), trial(cfg, "walk", 20)]
    failures = []
    if collisions:
        failures.append("Self collisions in commanded gait")
    for case in cases:
        if not case["finite"] or any(case["warnings"]):
            failures.append(f"{case['mode']}: invalid physics state")
        if case["maximum_torque_nm"] > cfg["servo"]["torque_limit_nm"] + 1e-8:
            failures.append(f"{case['mode']}: torque limit exceeded")
        if case["maximum_tilt_deg"] > 35 or case["nonfoot_ground_contact_steps"]:
            failures.append(f"{case['mode']}: fell or dragged non-foot geometry")
        if not all(case["foot_contact_steps"].values()):
            failures.append(f"{case['mode']}: one or more feet never supported the robot")
    if cases[1]["forward_m"] < .03:
        failures.append("Walking did not advance at least 30 mm in 20 s")
    report = {"passed": not failures, "failures": failures, "servo_count": robot.model.nu,
              "cad_mesh_count": robot.model.nmesh,
              "estimated_mass_kg": float(robot.model.body_mass.sum()),
              "self_collisions": sorted(collisions), "trials": cases}
    path = ROOT / "results" / "check.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return not failures


def snapshot(robot, path):
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, .04]
    camera.distance, camera.azimuth, camera.elevation = .62, 135, -28
    with mujoco.Renderer(robot.model, height=900, width=1200) as renderer:
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0
        renderer.update_scene(robot.data, camera=camera, scene_option=options)
        pixels = renderer.render()
    # Standard-library PNG writer keeps rendering free of additional dependencies.
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    h, w, _ = pixels.shape
    raw = b"".join(b"\0" + row.tobytes() for row in pixels)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    print(f"Saved {path}")


def interactive(robot):
    import glfw
    import mujoco.viewer
    commands = SimpleQueue()
    # Passive-viewer callbacks do not consume MuJoCo's built-in shortcuts.
    # Up/Down have no simulation or rendering binding in MuJoCo 3.3.7.
    print("Up: walk | Down: stand | R: reset | C: collision overlay | Space: pause | Esc: close")
    paused = False
    with mujoco.viewer.launch_passive(robot.model, robot.data, key_callback=commands.put) as viewer:
        viewer.opt.geomgroup[3] = 0
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = .62, 135, -28
        substeps = max(1, round(1 / (60 * robot.model.opt.timestep)))
        frame_seconds = substeps * robot.model.opt.timestep
        while viewer.is_running():
            began = time.perf_counter()
            while not commands.empty():
                key = commands.get()
                if key == glfw.KEY_UP:
                    robot.set_mode("walk")
                elif key == glfw.KEY_DOWN:
                    robot.set_mode("stand")
                elif key == ord("R"):
                    robot.reset()
                elif key == ord("C"):
                    with viewer.lock():
                        viewer.opt.geomgroup[3] = 1 - viewer.opt.geomgroup[3]
                elif key == ord(" "):
                    paused = not paused
                elif key == 256:
                    return
            with viewer.lock():
                if not paused:
                    for _ in range(substeps):
                        robot.step()
                viewer.cam.lookat[:] = robot.data.xpos[robot.body_id]
            viewer.sync()
            time.sleep(max(0., frame_seconds - (time.perf_counter() - began)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--walk", action="store_true", help="Start the basic crawl")
    parser.add_argument("--check", action="store_true", help="Check geometry, standing, and walking without a window")
    parser.add_argument("--headless", action="store_true", help="Run physics without opening a window")
    parser.add_argument("--seconds", type=float, default=12., help="Duration for --headless")
    parser.add_argument("--snapshot", type=Path, help="Save a PNG of the standing design, then exit")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    cfg = load_config(args.config)
    if args.check:
        raise SystemExit(0 if check(cfg) else 1)
    if args.headless:
        print(json.dumps(trial(cfg, "walk" if args.walk else "stand", args.seconds), indent=2))
        return
    robot = Robot(cfg)
    if args.snapshot:
        snapshot(robot, args.snapshot)
        return
    if args.walk:
        robot.set_mode("walk")
    interactive(robot)


if __name__ == "__main__":
    main()
