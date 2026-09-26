# MMA robot

## Setup

Install Python 3.11 or 3.12. Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe run_robot.py
```

Or press **Ctrl+Shift+B** in VS Code.

Click the simulation window: **Up arrow** walks, **Down arrow** stands, **R** resets, **C** toggles collision shapes, **Space** pauses, and **Esc** closes.



Edit `robot.json` to change the robot settings. See [design notes](docs/DESIGN.md) for CAD and hardware details.

The simulation displays the parts from `cad/*.stl`. Replace a part and restart to see it.
To edit the supplied CAD, change `cad/robot.scad`, then export with OpenSCAD installed:

```powershell
.\.venv\Scripts\python.exe build_robot.py --stl
```

This also updates `models/robot.xml`. STL changes update appearance; physics dimensions and masses still need matching settings.
