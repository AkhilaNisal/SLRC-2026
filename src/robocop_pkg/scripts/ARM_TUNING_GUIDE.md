# Robot Arm Tuning Guide

## Overview

The arm has 4 revolute joints + 1 gripper joint, all driven by PCA9685 servos via a socket bridge.
Commands flow: **MoveIt (Python) → JointTrajectoryController → rpi_arm_hardware (C++) → socket → pca9685_bridge.py → servos**

All arm poses are defined as joint angle dictionaries (in **radians**) inside `robot_arm_action_server.py`.

---

## Joint Reference

| # | Joint Name | Axis | Range | Effect |
|---|---|---|---|---|
| 0 | `base_rotating_waste_joint` | ≈ +Z (yaw) | ±90° | Rotates whole arm left/right |
| 1 | `rotating_waste_arm1_joint` | ≈ +Y (shoulder) | ±90° | Tilts arm forward/backward |
| 2 | `arm1_arm2_joint` | +Y (elbow) | −135° to +45° | Bends elbow |
| 3 | `arm2_gripper_base_joint` | **−Y** (wrist) | ±90° | Tilts gripper — **sign is inverted** in FK |
| 4 | `gripper_base_left_joint` | −X | −75° to +34° | Opens/closes gripper |

> **Important:** The wrist joint (J3) rotates around the **−Y axis**. This means a positive J3 value tilts the gripper **backward**, and a negative value tilts it **forward**. Keep this in mind when tuning wrist angles.

### Degrees ↔ Radians quick reference

| Degrees | Radians |
|---|---|
| ±90° | ±1.5708 |
| ±60° | ±1.0472 |
| ±45° | ±0.7854 |
| ±30° | ±0.5236 |
| ±15° | ±0.2618 |

Formula: `rad = deg × π/180`  or  `deg = rad × 180/π`

---

## Current Pose Library

All poses are declared as ROS2 parameters in `robot_arm_action_server.py`.

| Pose | J0 | J1 | J2 | J3 | Purpose |
|---|---|---|---|---|---|
| `home` | 20° | 37° | −72° | 53° | Safe resting/transit position |
| `grab` | 5° | 56° | 40° | −79° | Gripper at box level (≈15.5cm fwd, 10.6cm up) |
| `lift_1` | 5° | 52.5° | 40° | −79.5° | Step 1 of vertical lift (+9mm up, −1mm drift) |
| `lift_2` | 5° | 49° | 40° | −80° | Step 2 of vertical lift (+9mm up, −2mm drift) |
| `lift` | 5° | 45° | 40° | −81° | Final lift (≈14.9cm fwd, 13.5cm up) |
| `place1–3` | varies | — | — | — | Deposit positions for waste bin |
| `restore` | −5° | 76° | 45° | 35° | Pick-back position for restore sequence |

---

## How the Pick Sequence Works

```
gripper_open
    ↓
grab            ← arm moves to box level, gripper open
    ↓
gripper_close   ← grip the box
    ↓
lift_1          ← step up ~9mm (near-vertical, <2mm lateral drift)
    ↓
lift_2          ← step up another ~9mm
    ↓
lift            ← final lift position
    ↓
move_base_backward  ← robot reverses 5cm to clear the box area
    ↓
place1/2/3      ← arm swings to deposit position
    ↓
gripper_open    ← release
    ↓
home            ← return to safe position
```

The **3-step lift** (lift_1 → lift_2 → lift) replaces the old single grab→lift move.
Each small step forces OMPL to plan a short, direct trajectory instead of a potentially wild arc.

---

## Adding or Modifying a Pose

### Step 1 — Calculate in degrees using the FK tool

Edit the `POSES` dict in `scripts/arm_fk_tool.py` and add your candidate pose:

```python
POSES = {
    ...
    "my_new_pose": (J0_deg, J1_deg, J2_deg, J3_deg),
}
```

Run it:
```bash
python3 src/robocop_pkg/scripts/arm_fk_tool.py
```

The output shows:
- Gripper **x** (forward distance from robot base in cm)
- Gripper **z** (height above base_link in cm)
- **dz / dx** deltas from the previous pose
- **LIMIT!** warning if any joint exceeds its hardware limit

### Step 2 — Convert to radians and add to the action server

In `robot_arm_action_server.py`, declare parameters alongside the other poses:

```python
# my_new_pose = [J0_deg, J1_deg, J2_deg, J3_deg]
self.declare_parameter("my_new_pose.base_rotating_waste_joint",  J0_rad)
self.declare_parameter("my_new_pose.rotating_waste_arm1_joint",  J1_rad)
self.declare_parameter("my_new_pose.arm1_arm2_joint",            J2_rad)
self.declare_parameter("my_new_pose.arm2_gripper_base_joint",    J3_rad)
```

Then read it in the sequence method with:
```python
my_joints = self.get_arm_joint_dict("my_new_pose")
```

And execute it with:
```python
if not self.move_arm_joint_values_with_retry(my_joints, "my_new_pose"):
    return self.fail_result(goal_handle, "Failed at step: my_new_pose")
```

### Step 3 — Override at runtime without recompiling (optional)

You can override any parameter at launch without touching the code:

```bash
ros2 run robocop_pkg robot_arm_action_server \
  --ros-args \
  -p "lift_1.rotating_waste_arm1_joint:=0.95" \
  -p "lift_1.arm2_gripper_base_joint:=-1.41"
```

This is useful for quick testing on the robot before locking in values.

---

## Tuning Tips

### Vertical lift (keeping the box from sliding)
- Keep **J2 (elbow) constant** during the lift — all lifting should come from J1 (shoulder).
- Adjust J3 (wrist) slightly as J1 changes to compensate and keep gripper angle stable.
- Rule of thumb: for every **~7° decrease in J1**, increase J3 by **~1°** (more negative value).
- Use the FK tool to verify the horizontal drift stays under ~3mm per step.

### Avoiding collisions during place
- Always go through `lift` before swinging to a `place` position.
- If the arm grazes the robot chassis during the swing, increase J2 (elbow) by 5–10° in `lift` to bring the box higher before rotating.

### Gripper timing
- `step_pause_sec` (default 0.5s) is the wait after each arm move.
- `plan_settle_sec` (default 0.2s) is the wait before planning.
- If the arm arrives but the controller hasn't fully settled, increase `step_pause_sec`.

### Planning failures
- `retry_count` (default 3): number of OMPL re-plan attempts per step.
- If a specific step fails repeatedly, the joint target may be near a singularity — try shifting J1 or J2 by ±5° and re-check with the FK tool.

---

## FK Tool Reference

```
src/robocop_pkg/scripts/arm_fk_tool.py
```

**What it does:** Runs simplified forward kinematics in the arm's XZ plane (side view).
It does not simulate J0 (base yaw) — that only affects which direction the arm points, not height or forward reach.

**Limitations:**
- Ignores small Y-offsets between joints (error < 5mm in practice).
- Does not model collisions or reachability beyond joint limits.
- Gripper tip position is approximate (L3 = 30mm from wrist to grip point).

To add new poses for evaluation, edit the `POSES` dict at the top of the script — no ROS installation needed, plain Python only.

---

## Quick Debugging Checklist

| Symptom | Likely cause | Fix |
|---|---|---|
| Arm swings wildly between two poses | OMPL found a distant path in joint space | Add an intermediate waypoint between the two poses |
| Box tips during lift | Too much horizontal drift | Add more lift steps; reduce J1 change per step to ≤5° |
| Box slides during grab→close | Gripper not fully at box level | Decrease J1 in `grab` by 2–3° (lowers gripper slightly) |
| Planning fails repeatedly | Near a singularity | Shift J1 or J2 by ±5° in the target pose |
| Gripper overshoots on close | Servo speed too high | Reduce velocity scaling in `joint_limits.yaml` (currently 0.1) |
| Arm doesn't reach full extension | Joint limit hit | Check FK tool output for LIMIT! warnings |
