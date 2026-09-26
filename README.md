# MMA robot

## Setup

Install Python 3.11 or 3.12. Run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe run_robot.py
```

Click the simulation window: **Up arrow** walks, **Down arrow** stands, **R** resets, **C** toggles collision shapes, **Space** pauses, and **Esc** closes.



Edit `robot.json` to change the robot settings. See [design notes](docs/DESIGN.md) for CAD and hardware details.

The simulation displays the parts from `cad/*.stl`. Replace a part and restart to see it.
Edit parts and export replacement STLs in millimetres. Preserve the filenames and coordinate conventions described in the design notes. To refresh the saved simulation XML and its mesh assets, run:

```powershell
.\.venv\Scripts\python.exe build_robot.py
```

This updates `models/robot.xml` and `models/meshes/` without changing the source STLs. STL changes update appearance; physics dimensions and masses still need matching settings.
