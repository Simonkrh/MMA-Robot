"""Behavior checks for configuration, controls, and the free physical model."""
import copy
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import mujoco
import numpy as np

from build_robot import CAD_PARTS, LEGS, ROOT, binary_stl, build, export_cad, load_config, model_xml, read_stl
from run_robot import Robot


class RobotTests(unittest.TestCase):
    def test_exported_parts_are_closed_and_connected(self):
        for part in ("deck", "carrier", "leg", "foot"):
            with self.subTest(part=part):
                triangles = read_stl(ROOT / "cad" / f"{part}.stl")
                vertices = [tuple(v) for v in triangles.reshape(-1, 3)]
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

    @staticmethod
    def visual_vertices(robot, geom_name, world=False):
        """Recover vertices after MuJoCo's internal mesh centering and rotation."""
        model, data = robot.model, robot.data
        gid = model.geom(geom_name).id
        mid = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        vertices = model.mesh_vert[start:start + count]
        points = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        if world:
            return points
        bid = model.geom_bodyid[gid]
        return (points - data.xpos[bid]) @ data.xmat[bid].reshape(3, 3)

    def test_meshes_are_attached_at_correct_scale_and_joint_frames(self):
        robot = Robot()
        self.assertEqual(robot.model.nmesh, 4)
        self.assertEqual(np.count_nonzero(robot.model.geom_group == 2), 13)
        np.testing.assert_allclose(
            [self.visual_vertices(robot, "deck_visual").min(axis=0),
             self.visual_vertices(robot, "deck_visual").max(axis=0)],
            [[-.068, -.054, -.003], [.068, .054, 0]], atol=2e-7)
        for swing, lift in ((0., 60.), (20., 35.), (-20., 80.)):
            robot.data.qpos[robot.qadr] = np.tile(np.radians([swing, lift]), 4)
            mujoco.mj_forward(robot.model, robot.data)
            for name, _, _ in LEGS:
                with self.subTest(leg=name, swing=swing, lift=lift):
                    points = self.visual_vertices(robot, f"{name}_leg_visual")
                    np.testing.assert_allclose([points.min(axis=0), points.max(axis=0)],
                                               [[-.010, .002, -.010], [.095, .006, .010]], atol=2e-7)
                    gid = robot.model.geom(f"{name}_leg_visual").id
                    self.assertEqual(robot.model.geom_bodyid[gid], robot.model.jnt_bodyid[robot.model.joint(f"{name}_lift").id])
                    carrier = robot.model.geom(f"{name}_carrier_visual").id
                    self.assertEqual(robot.model.geom_bodyid[carrier], robot.model.jnt_bodyid[robot.model.joint(f"{name}_swing").id])
                    # Carrier shaft mounting face remains aligned with the lift joint.
                    carrier_points = self.visual_vertices(robot, f"{name}_carrier_visual")
                    self.assertAlmostEqual(float(carrier_points[:, 2].min()), .002, places=6)
                    self.assertAlmostEqual(float(carrier_points[:, 2].max()), .03615, places=6)
                    foot = self.visual_vertices(robot, f"{name}_foot_visual", world=True)
                    centre = robot.data.site(f"{name}_toe").xpos
                    # The sphere's surface is centred on the toe, despite its print-bed offset.
                    radii = np.linalg.norm(foot - centre, axis=1)
                    self.assertLessEqual(float(radii.max()), .008001)
                    self.assertGreater(float(radii.max()), .0079)

    def test_visual_meshes_do_not_change_physics(self):
        robot = Robot()
        original = mujoco.MjModel.from_xml_string(model_xml(robot.cfg, cad_visuals=False))
        for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "jnt_pos", "jnt_axis", "jnt_range"):
            np.testing.assert_allclose(getattr(robot.model, field), getattr(original, field), atol=1e-12)
        visuals = robot.model.geom_group == 2
        np.testing.assert_array_equal(robot.model.geom_contype[visuals], 0)
        np.testing.assert_array_equal(robot.model.geom_conaffinity[visuals], 0)

    def test_replacing_source_stl_is_visible_on_next_load(self):
        with tempfile.TemporaryDirectory() as folder:
            cad_dir = Path(folder)
            for part in CAD_PARTS:
                shutil.copyfile(ROOT / "cad" / f"{part}.stl", cad_dir / f"{part}.stl")
            before = Robot(cad_dir=cad_dir)
            triangles = read_stl(cad_dir / "leg.stl")
            triangles[:, :, 1] *= 1.5  # Print Y becomes assembled Z: wider leg, same shaft centre.
            (cad_dir / "leg.stl").write_bytes(binary_stl(triangles))
            after = Robot(cad_dir=cad_dir)
            old_span = np.ptp(self.visual_vertices(before, "FL_leg_visual"), axis=0)
            new_span = np.ptp(self.visual_vertices(after, "FL_leg_visual"), axis=0)
            self.assertAlmostEqual(new_span[2] / old_span[2], 1.5, places=5)
            np.testing.assert_allclose(before.model.body_mass, after.model.body_mass)
            np.testing.assert_allclose(before.model.jnt_pos, after.model.jnt_pos)

    def test_saved_xml_and_meshes_load_from_a_different_directory(self):
        cfg = load_config()
        with tempfile.TemporaryDirectory(prefix="MMA robot ") as folder:
            output = build(cfg, Path(folder) / "models" / "robot.xml")
            model = mujoco.MjModel.from_xml_path(str(output))
            self.assertEqual(model.nmesh, 4)
            self.assertEqual(model.nu, 8)
            np.testing.assert_allclose(model.body_inertia, Robot(cfg).model.body_inertia)
            self.assertNotIn(str(ROOT), output.read_text(encoding="utf-8"))

    def test_refreshing_dimensions_preserves_edited_cad_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "robot.scad"
            source.write_text("// My custom leg design\n", encoding="utf-8")
            export_cad(load_config(), Path(folder) / "parameters.scad")
            self.assertEqual(source.read_text(encoding="utf-8"), "// My custom leg design\n")

    def test_missing_or_invalid_mesh_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "deck.stl"
            with self.assertRaisesRegex(FileNotFoundError, "Missing CAD mesh.*deck.stl"):
                Robot(cad_dir=folder)
            for data in (b"", b"solid bad\nvertex nan 0 0\nvertex 0 1 0\nvertex 0 0 1\nendsolid", b"invalid mesh"):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    read_stl(path)

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
