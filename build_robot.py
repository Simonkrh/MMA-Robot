"""Build the compact quadruped from robot.json. No original CAD files required."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "robot.json"
LEGS = (("FL", 1, 1), ("FR", 1, -1), ("RL", -1, 1), ("RR", -1, -1))
CAD_PARTS = ("deck", "carrier", "leg", "foot")
CAD_DIR = ROOT / "cad"


def read_stl(path):
    """Read ASCII or binary STL triangles in the source file's units."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing CAD mesh: {path}. Restore it or export with build_robot.py --stl.")
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else 0
    if count and len(data) == 84 + 50 * count:
        triangles = np.array([struct.unpack_from("<12fH", data, 84 + 50 * i)[3:12]
                              for i in range(count)], dtype=float).reshape(-1, 3, 3)
    else:
        try:
            text = data.decode("utf-8-sig")
            rows = re.findall(r"^\s*vertex\s+([^\r\n]+)", text, flags=re.MULTILINE)
            triangles = np.array([[float(v) for v in row.split()] for row in rows], dtype=float).reshape(-1, 3, 3)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid STL mesh: {path}") from exc
    if not triangles.size or not np.isfinite(triangles).all():
        raise ValueError(f"Empty or non-finite STL mesh: {path}")
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if np.any(area <= 1e-12):
        raise ValueError(f"STL contains zero-area triangles: {path}")
    return triangles


def binary_stl(triangles):
    """Encode triangles for MuJoCo without changing their coordinates or units."""
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    records = [struct.pack("<12fH", *normal, *triangle.ravel(), 0)
               for normal, triangle in zip(normals, triangles)]
    return b"MMA robot simulation mesh".ljust(80, b"\0") + struct.pack("<I", len(triangles)) + b"".join(records)


def mesh_assets(cfg, cad_dir=CAD_DIR):
    """Undo print-bed transforms; return assembly-local, millimetre binary meshes.

    Source STLs are never modified. Assets are freshly read on each invocation.
    These inverses match the per-part export transforms at the end of robot.scad.
    """
    assets = {}
    for part in CAD_PARTS:
        triangles = read_stl(Path(cad_dir) / f"{part}.stl")
        if part == "deck":
            triangles[:, :, 2] -= cfg["body"]["plate_mm"]
        elif part == "carrier":
            triangles[:, :, 2] += 2
        elif part == "leg":
            print_y = triangles[:, :, 1].copy()
            triangles[:, :, 1] = triangles[:, :, 2] + 2
            triangles[:, :, 2] = -print_y
        elif part == "foot":
            triangles[:, :, 2] -= cfg["leg"]["foot_radius_mm"]
        assets[f"meshes/{part}.stl"] = binary_stl(triangles)
    return assets


def load_config(path=CONFIG):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for group, keys in {
        "servo": ("length_mm", "width_mm", "height_mm", "shaft_offset_mm", "mass_g", "rated_torque_nm", "torque_limit_nm", "speed_deg_s"),
        "body": ("length_mm", "width_mm", "plate_mm", "clearance_mm", "mass_g", "payload_mass_g"),
        "leg": ("hip_x_mm", "hip_y_mm", "reach_mm", "lift_height_mm", "length_mm", "foot_radius_mm", "bracket_mass_g", "shin_mass_g", "foot_mass_g", "yaw_limit_deg", "lift_min_deg", "lift_max_deg"),
        "gait": ("period_s", "sweep_deg", "lift_deg", "ramp_s"),
        "simulation": ("timestep_s", "kp", "kd", "friction"),
    }.items():
        for key in keys:
            value = cfg[group][key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{group}.{key} must be finite and positive")
    s, b, leg, g = (cfg[k] for k in ("servo", "body", "leg", "gait"))
    payload = b["payload_size_mm"]
    if len(payload) != 3 or any(not math.isfinite(v) or v <= 0 for v in payload):
        raise ValueError("body.payload_size_mm must contain three finite positive dimensions")
    if payload[0] + 2 * b["clearance_mm"] > b["length_mm"]:
        raise ValueError("Payload is too long for the body")
    if s["torque_limit_nm"] > s["rated_torque_nm"]:
        raise ValueError("Torque limit exceeds the manufacturer's rated torque")
    if cfg["simulation"]["timestep_s"] > .005:
        raise ValueError("Use a simulation timestep of 0.005 s or less")
    if not all(math.isfinite(v) for v in (g["duty_factor"], leg["stance_deg"])):
        raise ValueError("Gait duty factor and stance angle must be finite")
    if not 0.75 <= g["duty_factor"] < 0.95:
        raise ValueError("gait.duty_factor must be in [0.75, 0.95) for a crawl")
    if not 20 <= leg["stance_deg"] <= 80:
        raise ValueError("leg.stance_deg must be between 20 and 80 degrees")
    if not 0 < s["shaft_offset_mm"] < s["length_mm"] / 2:
        raise ValueError("servo shaft offset must lie within its length")
    case_x = leg["hip_x_mm"] - s["shaft_offset_mm"]
    if case_x < s["length_mm"] / 2 + b["clearance_mm"]:
        raise ValueError("Front and rear hip servo envelopes overlap")
    if case_x + s["length_mm"] / 2 + b["clearance_mm"] > b["length_mm"] / 2:
        raise ValueError("Body too short for the hip servos")
    if leg["hip_y_mm"] + s["width_mm"] / 2 + b["clearance_mm"] > b["width_mm"] / 2:
        raise ValueError("Body too narrow for the hip servos")
    if b["payload_size_mm"][1] / 2 + b["clearance_mm"] > leg["hip_y_mm"] - s["width_mm"] / 2:
        raise ValueError("Payload is too wide for the central bay")
    if g["sweep_deg"] >= leg["yaw_limit_deg"] or leg["stance_deg"] - g["lift_deg"] <= leg["lift_min_deg"]:
        raise ValueError("Gait exceeds joint travel")
    if leg["stance_deg"] >= leg["lift_max_deg"]:
        raise ValueError("Standing angle exceeds joint travel")
    if standing_height(cfg) * 1000 < s["height_mm"] + b["plate_mm"] + 5:
        raise ValueError("Insufficient ground clearance below the body servos")
    if standing_height(cfg) * 1000 < payload[2] + b["plate_mm"] + 5:
        raise ValueError("Insufficient ground clearance below the payload")
    if leg["lift_height_mm"] < s["width_mm"] / 2 + 5:
        raise ValueError("Lift servo carrier is too low above the deck")
    return cfg


def standing_height(cfg):
    l = cfg["leg"]
    return (l["length_mm"] * math.sin(math.radians(l["stance_deg"]))
            + l["foot_radius_mm"] - l["lift_height_mm"]) / 1000


def values(seq):
    return " ".join(f"{x:.8g}" for x in seq)


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: values(v) if isinstance(v, (tuple, list)) else str(v)
                                    for k, v in attrs.items()})


def model_xml(cfg, *, cad_visuals=True):
    s, b, l, sim = (cfg[k] for k in ("servo", "body", "leg", "simulation"))
    sl, sw, sh = (s[k] / 1000 for k in ("length_mm", "width_mm", "height_mm"))
    offset = s["shaft_offset_mm"] / 1000
    reach, rise, length, radius = (l[k] / 1000 for k in ("reach_mm", "lift_height_mm", "length_mm", "foot_radius_mm"))
    plate = b["plate_mm"] / 1000
    root = ET.Element("mujoco", model=cfg["name"])
    add(root, "compiler", angle="degree", autolimits="true")
    add(root, "option", timestep=sim["timestep_s"], integrator="implicitfast", gravity="0 0 -9.81")
    visual = add(root, "visual")
    add(visual, "global", offwidth=1280, offheight=960)
    add(visual, "headlight", diffuse="0.7 0.7 0.7", ambient="0.35 0.35 0.35")
    add(visual, "map", znear="0.01")
    default = add(root, "default")
    add(default, "joint", damping="0.015", armature="0.00002")
    add(default, "geom", friction=[sim["friction"], 0.01, 0.001], condim=3,
        solref="0.006 1", solimp="0.95 0.99 0.001")
    asset = add(root, "asset")
    if cad_visuals:
        for part in CAD_PARTS:
            add(asset, "mesh", name=f"{part}_mesh", file=f"meshes/{part}.stl", scale="0.001 0.001 0.001")
    add(asset, "texture", name="grid", type="2d", builtin="checker", rgb1="0.15 0.19 0.23", rgb2="0.19 0.23 0.27", width=512, height=512)
    add(asset, "material", name="floor", texture="grid", texrepeat="16 16", reflectance="0.05")
    world = add(root, "worldbody")
    add(world, "light", pos="0 -1 2", dir="0 0 -1", directional="true")
    add(world, "geom", name="ground", type="plane", size="2 2 0.05", material="floor")
    body = add(world, "body", name="chassis", pos=[0, 0, standing_height(cfg) + 0.001])
    add(body, "freejoint", name="floating_base")
    # The deck is a simplified solid collision envelope. CAD export includes pockets.
    add(body, "geom", name="deck", type="box", size=[b["length_mm"] / 2000, b["width_mm"] / 2000, plate / 2],
        pos=[0, 0, -plate / 2], mass=b["mass_g"] / 1000, rgba="0.04 0.63 0.65 1")
    payload = [v / 1000 for v in b["payload_size_mm"]]
    add(body, "geom", name="battery_and_controller", type="box", size=[v / 2 for v in payload],
        pos=[0, 0, -plate - payload[2] / 2], mass=b["payload_mass_g"] / 1000, rgba="0.17 0.21 0.27 1")
    add(body, "geom", name="front_marker", type="box", pos=[b["length_mm"] / 2000 - .005, 0, .001],
        size="0.004 0.014 0.001", mass="0", contype=0, conaffinity=0, rgba="1 0.65 0.18 1")
    for name, front, side in LEGS:
        x, y = front * l["hip_x_mm"] / 1000, side * l["hip_y_mm"] / 1000
        add(body, "geom", name=f"{name}_hip_servo", type="box", size=[sl / 2, sw / 2, sh / 2],
            pos=[x - front * offset, y, -plate - sh / 2], mass=s["mass_g"] / 1000, rgba="0.13 0.15 0.18 1")
        heading = math.degrees(math.atan2(side, front))
        hip = add(body, "body", name=f"{name}_hip", pos=[x, y, 0], euler=[0, 0, heading])
        add(hip, "joint", name=f"{name}_swing", axis="0 0 1", range=[-l["yaw_limit_deg"], l["yaw_limit_deg"]])
        add(hip, "geom", name=f"{name}_riser", type="capsule", fromto=[0, 0, .004, 0, 0, rise], size="0.006", mass="0.004", rgba="0.04 0.63 0.65 1")
        add(hip, "geom", name=f"{name}_bracket", type="box", pos=[reach / 2, -.005, rise], size=[reach / 2 + .009, .003, .003], mass=l["bracket_mass_g"] / 2000, rgba="0.04 0.63 0.65 1")
        add(hip, "geom", name=f"{name}_cradle", type="box", pos=[reach - offset, -sh / 2 - .006, rise - sw / 2 - .0015],
            size=[sl / 2 + .002, sh / 2 + .002, .0015], mass=l["bracket_mass_g"] / 2000, rgba="0.04 0.63 0.65 1")
        add(hip, "geom", name=f"{name}_cradle_wall", type="box", pos=[reach - offset, -.0045, rise - .0015],
            size=[sl / 2 + .002, .0015, sw / 2 + .0015], mass=0, rgba="0.04 0.63 0.65 1")
        # Lift shaft is tangent to the body; its casing sits behind the horn plane.
        add(hip, "geom", name=f"{name}_lift_servo", type="box", size=[sl / 2, sh / 2, sw / 2],
            pos=[reach - offset, -sh / 2 - .006, rise], mass=s["mass_g"] / 1000, rgba="0.13 0.15 0.18 1")
        shin = add(hip, "body", name=f"{name}_leg", pos=[reach, 0, rise])
        add(shin, "joint", name=f"{name}_lift", axis="0 1 0", range=[l["lift_min_deg"], l["lift_max_deg"]])
        add(shin, "geom", name=f"{name}_shin", type="capsule", fromto=[0, .004, 0, length, .004, 0], size="0.004", mass=l["shin_mass_g"] / 1000, rgba="0.93 0.68 0.25 1")
        add(shin, "geom", name=f"{name}_foot", type="sphere", pos=[length, .004, 0], size=radius,
            mass=l["foot_mass_g"] / 1000, rgba="0.12 0.14 0.17 1")
        add(shin, "site", name=f"{name}_toe", pos=[length, .004, 0], size="0.002", rgba="1 0.3 0.2 1")
    actuator = add(root, "actuator")
    for name, _, _ in LEGS:
        for joint in ("swing", "lift"):
            add(actuator, "motor", name=f"{name}_{joint}", joint=f"{name}_{joint}",
                ctrllimited="true", ctrlrange=[-s["torque_limit_nm"], s["torque_limit_nm"]])
    keyframe = add(root, "keyframe")
    qpos = [0, 0, standing_height(cfg) + .001, 1, 0, 0, 0]
    for _ in LEGS:
        qpos += [0, math.radians(l["stance_deg"])]
    add(keyframe, "key", name="stand", qpos=qpos)
    if cad_visuals:
        attach_cad_visuals(root, cfg)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


def attach_cad_visuals(root, cfg):
    """Visuals follow existing rigid bodies; hidden proxies still own all physics."""
    def attach(body_name, part, name, proxies, color, pos=(0, 0, 0)):
        body = root.find(f".//body[@name='{body_name}']")
        for proxy in proxies:
            body.find(f"geom[@name='{proxy}']").set("group", "3")
        add(body, "geom", name=name, type="mesh", mesh=f"{part}_mesh", pos=pos,
            mass=0, contype=0, conaffinity=0, group=2, rgba=color)

    attach("chassis", "deck", "deck_visual", ["deck"], "0.04 0.63 0.65 1")
    for name, _, _ in LEGS:
        attach(f"{name}_hip", "carrier", f"{name}_carrier_visual",
               [f"{name}_{part}" for part in ("riser", "bracket", "cradle", "cradle_wall")], "0.04 0.63 0.65 1")
        attach(f"{name}_leg", "leg", f"{name}_leg_visual", [f"{name}_shin"], "0.93 0.68 0.25 1")
        attach(f"{name}_leg", "foot", f"{name}_foot_visual", [f"{name}_foot"], "0.12 0.14 0.17 1",
               pos=(cfg["leg"]["length_mm"] / 1000, .004, 0))


def build(cfg=None, output=ROOT / "models" / "robot.xml", cad_dir=CAD_DIR):
    cfg = load_config() if cfg is None else cfg
    assets = mesh_assets(cfg, cad_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for filename, data in assets.items():
        destination = output.parent / filename
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(data)
    output.write_text(model_xml(cfg), encoding="utf-8")
    return output


def export_cad(cfg, output=CAD_DIR / "parameters.scad"):
    """Refresh shared dimensions only; robot.scad is a user-editable source file."""
    s, b, l = (cfg[k] for k in ("servo", "body", "leg"))
    parameters = {
        "sl": s["length_mm"], "sw": s["width_mm"], "sh": s["height_mm"],
        "shaft_offset": s["shaft_offset_mm"], "bl": b["length_mm"], "bw": b["width_mm"],
        "plate": b["plate_mm"], "hx": l["hip_x_mm"], "hy": l["hip_y_mm"],
        "reach": l["reach_mm"], "rise": l["lift_height_mm"], "leg_length": l["length_mm"],
        "stance": l["stance_deg"], "foot_radius": l["foot_radius_mm"],
    }
    text = "// Generated dimensions from robot.json. Edit geometry in robot.scad.\n"
    text += "\n".join(f"{key} = {value};" for key, value in parameters.items()) + "\n"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def export_stl(scad, executable=None):
    executable = executable or shutil.which("openscad")
    if not executable:
        candidate = Path("C:/Program Files/OpenSCAD/openscad.com")
        if candidate.exists():
            executable = str(candidate)
    if not executable:
        raise RuntimeError("STL export needs OpenSCAD. Install it or pass --openscad PATH.")
    for part in CAD_PARTS:
        destination = scad.with_name(f"{part}.stl")
        subprocess.run([str(executable), "-o", str(destination), "-D", f'part="{part}"', str(scad)], check=True)
        print(f"Exported {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--stl", action="store_true", help="Also export prototype STLs using OpenSCAD")
    parser.add_argument("--openscad", type=Path, help="Optional path to OpenSCAD executable")
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"Updated CAD dimensions: {export_cad(config)}")
    if args.stl:
        export_stl(CAD_DIR / "robot.scad", args.openscad)
    print(f"Built {build(config)}")
