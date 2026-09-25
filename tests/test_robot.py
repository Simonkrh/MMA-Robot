"""Behavior checks for configuration, controls, and the free physical model."""
import copy
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import tempfile
import unittest

import mujoco
import numpy as np

from build_robot import ROOT, load_config
from run_robot import Robot


class RobotTests(unittest.TestCase):
    def test_exported_parts_are_closed_and_connected(self):
        for part in ("deck", "carrier", "leg", "foot"):
            with self.subTest(part=part):
                text = (ROOT / "cad" / f"{part}.stl").read_text(encoding="utf-8")
                vertices = [tuple(float(v) for v in line.split())
                            for line in re.findall(r"vertex\s+([^\n]+)", text)]
                self.assertGreater(len(vertices), 0)
                self.assertEqual(len(vertices) % 3, 0)
                edges, graph = Counter(), defaultdict(set)
                for i in range(0, len(vertices), 3):
                    a, b, c = vertices[i:i + 3]
                    for u, v in ((a, b), (b, c), (c, a)):
                        edges[tuple(sorted((u, v)))] += 1
                        graph[u].add(v)
                        graph[v].add(u)
                self.assertTrue(all(count == 2 for count in edges.values()))
                remaining = set(graph)
                queue = [remaining.pop()]
                while queue:
                    for v in graph[queue.pop()]:
                        if v in remaining:
                            remaining.remove(v)
                            queue.append(v)
                self.assertFalse(remaining, "Part contains disconnected pieces")

    def test_rejects_impossible_geometry_and_invalid_physics(self):
        changes = [
            ("body", "width_mm", 50),
            ("body", "payload_size_mm", [70, 100, 22]),
            ("leg", "length_mm", 30),
            ("leg", "hip_x_mm", 20),
            ("gait", "lift_deg", 70),
            ("gait", "duty_factor", .5),
            ("simulation", "timestep_s", .05),
            ("servo", "torque_limit_nm", 99),
            ("servo", "mass_g", float("nan")),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for group, key, value in changes:
                with self.subTest(group=group, key=key):
                    cfg = load_config()
                    cfg[group][key] = value
                    path.write_text(json.dumps(cfg), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_body_really_falls_without_motor_control(self):
        robot = Robot()
        robot.data.qpos[2] += .2
        mujoco.mj_forward(robot.model, robot.data)
        initial_height = robot.data.qpos[2]
        for _ in range(100):
            mujoco.mj_step(robot.model, robot.data)
        self.assertLess(robot.data.qpos[2], initial_height - .1)
        self.assertEqual(robot.model.nu, 8)
        self.assertEqual(robot.model.neq, 0)  # No hidden support constraints.

    def test_crawl_lifts_only_one_leg_at_a_time_and_is_continuous(self):
        robot = Robot()
        robot.set_mode("walk")
        g = robot.cfg["gait"]
        start = g["ramp_s"] + g["period_s"]
        airborne_legs = set()
        for t in np.linspace(start, start + g["period_s"], 501):
            robot.data.time = t
            q = robot.targets()
            lifted = np.flatnonzero(q[1::2] < robot.neutral[1::2] - 1e-6)
            self.assertLessEqual(len(lifted), 1)
            airborne_legs.update(lifted)
            robot.data.time = t + 1e-7
            self.assertLess(float(np.max(np.abs(robot.targets() - q))), 1e-5)
        self.assertEqual(airborne_legs, {0, 1, 2, 3})

    def test_stop_blends_from_current_pose_and_reset_clears_motion(self):
        robot = Robot()
        robot.set_mode("walk")
        for _ in range(1700):
            robot.step()
        before = robot.target.copy()
        robot.set_mode("stand")
        np.testing.assert_allclose(robot.targets(), before, atol=1e-10)
        for _ in range(1200):
            robot.step()
        np.testing.assert_allclose(robot.target, robot.neutral, atol=1e-10)
        self.assertFalse(robot.ground_contacts()[1])
        robot.reset()
        self.assertEqual(robot.mode, "stand")
        self.assertEqual(robot.data.time, 0)
        np.testing.assert_array_equal(robot.data.qvel, 0)
        np.testing.assert_array_equal(robot.data.ctrl, 0)

    def test_config_changes_affect_model_without_rebuilding(self):
        cfg = load_config()
        altered = copy.deepcopy(cfg)
        altered["body"]["payload_mass_g"] += 25
        original, changed = Robot(cfg), Robot(altered)
        self.assertAlmostEqual(changed.model.body_mass.sum() - original.model.body_mass.sum(), .025)

    def test_torque_limit_holds_during_large_tracking_error(self):
        robot = Robot()
        robot.data.qpos[robot.qadr] = 0
        robot.data.qvel[robot.vadr] = np.tile([20., -20.], 4)
        robot.step()
        self.assertLessEqual(float(np.max(np.abs(robot.data.ctrl))), robot.cfg["servo"]["torque_limit_nm"])


if __name__ == "__main__":
    unittest.main()
