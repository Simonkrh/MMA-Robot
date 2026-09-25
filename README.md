# MMA robot

A compact four-legged robot simulation with eight servos.

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

Click the simulation window: **W** walks, **S** stands, **R** resets, **Space** pauses, and **Esc** closes.

To start walking immediately:

```powershell
.\.venv\Scripts\python.exe run_robot.py --walk
```

Edit `robot.json` to change the robot settings. See [design notes](docs/DESIGN.md) for CAD and hardware details.
