"""Build the compact quadruped from robot.json. No original CAD files required."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "robot.json"
LEGS = (("FL", 1, 1), ("FR", 1, -1), ("RL", -1, 1), ("RR", -1, -1))


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


def model_xml(cfg):
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
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


def build(cfg=None, output=ROOT / "models" / "robot.xml"):
    cfg = load_config() if cfg is None else cfg
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(model_xml(cfg), encoding="utf-8")
    return output


def cad_source(cfg):
    """Editable prototype parts in millimetres, driven by the same JSON as physics.

    Case envelopes are deliberately conservative. Horn slots and strap mounts avoid
    pretending the manufacturer's overview dimensions define a mounting drawing.
    """
    s, b, l = (cfg[k] for k in ("servo", "body", "leg"))
    parameters = {
        "sl": s["length_mm"], "sw": s["width_mm"], "sh": s["height_mm"],
        "shaft_offset": s["shaft_offset_mm"], "bl": b["length_mm"], "bw": b["width_mm"],
        "plate": b["plate_mm"], "hx": l["hip_x_mm"], "hy": l["hip_y_mm"],
        "reach": l["reach_mm"], "rise": l["lift_height_mm"], "leg_length": l["length_mm"],
        "stance": l["stance_deg"], "foot_radius": l["foot_radius_mm"],
    }
    header = "// Generated by build_robot.py from robot.json. Units: mm.\n"
    header += "// Prototype: measure shaft position, horn and fasteners before printing.\n"
    header += "\n".join(f"{k} = {v};" for k, v in parameters.items()) + "\n"
    return header + r'''
// Select assembly, deck, carrier, leg, or foot; assembly includes servo envelopes.
part = "assembly";
$fn = 48;

module slot(length, width, height) {
    hull() for (x = [-1, 1]) translate([x*(length-width)/2, 0, 0])
        cylinder(d=width, h=height, center=true);
}
module rounded_plate(length, width, height, radius=5) {
    hull() for (x=[-1,1], y=[-1,1])
        translate([x*(length/2-radius), y*(width/2-radius), 0])
            cylinder(r=radius, h=height, center=true);
}
module deck() {
    difference() {
        translate([0,0,-plate/2]) rounded_plate(bl,bw,plate);
        for (front=[-1,1], side=[-1,1]) {
            // Shaft/horn access hole. Hip servo is strapped against deck underside.
            translate([front*hx,side*hy,-plate/2]) cylinder(d=18,h=plate+2,center=true);
            for (x=[-1,1], y=[-1,1])
                translate([front*(hx-shaft_offset)+x*sl/4,
                           side*hy+y*(sw/2+3),-plate/2])
                    slot(6,2.5,plate+2);
        }
        // Battery straps and controller cable access.
        for (x=[-24,24], y=[-22,22]) translate([x,y,-plate/2]) slot(12,3,plate+2);
        translate([0,0,-plate/2]) slot(20,8,plate+2);
    }
}
module carrier() {
    tray_x = reach-shaft_offset;
    tray_y = -sh/2-6;
    bottom = rise-sw/2-3;
    difference() {
        union() {
            translate([0,0,3.5]) cylinder(r=10,h=3,center=true);
            translate([0,0,(rise+4)/2]) cylinder(r=6,h=rise-4,center=true);
            translate([reach/2,-5,rise]) cube([reach+18,6,6],center=true);
            translate([tray_x,tray_y,bottom+1.5]) cube([sl+4,sh+4,3],center=true);
            // Front wall joins cradle, horn bridge, and base; rear wall retains servo.
            for (y=[-4.5,-sh-7.5]) translate([tray_x,y,rise-1.5])
                cube([sl+4,3,sw+3],center=true);
        }
        // Horn centre access and two adjustable attachment slots.
        translate([0,0,rise/2]) cylinder(d=3.2,h=rise+12,center=true);
        for (x=[-7,7]) translate([x,0,3.5]) slot(5,2.6,5);
        // Clearance around horizontal lift shaft.
        translate([reach,-4.5,rise]) rotate([90,0,0]) cylinder(d=14,h=14,center=true);
        // Two strap passages through the carrier floor.
        for (x=[-sl/4,sl/4]) translate([tray_x+x,tray_y,bottom+1.5])
            cube([3,sh-4,5],center=true);
    }
}
module leg() {
    difference() {
        hull() for (x=[0,leg_length]) translate([x,4,0])
            rotate([90,0,0]) cylinder(r=x==0 ? 10 : 5,h=4,center=true);
        translate([0,4,0]) rotate([90,0,0]) cylinder(d=3.2,h=6,center=true);
        for (x=[-7,7]) translate([x,4,0]) rotate([90,0,0]) slot(5,2.6,6);
        translate([leg_length,4,0]) rotate([90,0,0]) cylinder(d=3.2,h=6,center=true);
    }
}
module foot() {
    // Flexible-material cap; slit fits the flat leg tip. Fit must be tested.
    difference() {
        sphere(r=foot_radius);
        translate([-1.5,0,0]) cube([2*foot_radius-1,4.4,10.4],center=true);
        rotate([90,0,0]) cylinder(d=3.2,h=2*foot_radius+2,center=true);
    }
}
module assembly() {
    color([0.04,0.63,0.65]) deck();
    for (front=[-1,1], side=[-1,1]) {
        color([0.13,0.15,0.18])
            translate([front*(hx-shaft_offset),side*hy,-plate-sh/2])
                cube([sl,sw,sh],center=true);
        translate([front*hx,side*hy,0]) rotate([0,0,atan2(side,front)]) {
            color([0.04,0.63,0.65]) carrier();
            color([0.13,0.15,0.18]) translate([reach-shaft_offset,-sh/2-6,rise])
                cube([sl,sh,sw],center=true);
            translate([reach,0,rise]) rotate([0,stance,0]) {
                color([0.93,0.68,0.25]) leg();
                color([0.12,0.14,0.17]) translate([leg_length,4,0]) foot();
            }
        }
    }
}
if (part=="assembly") assembly();
else if (part=="deck") translate([0,0,plate]) deck();
else if (part=="carrier") translate([0,0,-2]) carrier();
else if (part=="leg") translate([0,0,-2]) rotate([90,0,0]) leg();
else if (part=="foot") translate([0,0,foot_radius]) foot();
else assert(false,"Unknown part selection");
'''


def export_cad(cfg, output=ROOT / "cad" / "robot.scad"):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(cad_source(cfg), encoding="utf-8")
    return output


def export_stl(scad, executable=None):
    executable = executable or shutil.which("openscad")
    if not executable:
        candidate = Path("C:/Program Files/OpenSCAD/openscad.com")
        if candidate.exists():
            executable = str(candidate)
    if not executable:
        raise RuntimeError("STL export needs OpenSCAD. Install it or pass --openscad PATH.")
    for part in ("deck", "carrier", "leg", "foot"):
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
    print(f"Built {build(config)}")
    scad = export_cad(config)
    print(f"Generated prototype CAD: {scad}")
    if args.stl:
        export_stl(scad, args.openscad)
