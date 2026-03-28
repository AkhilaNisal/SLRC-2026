#!/usr/bin/env python3
"""
Quick FK tool for robot_arm_v2.
Prints gripper tip position in the XZ plane (side view) for any joint config.
Usage:
    python3 arm_fk_tool.py
    # or edit POSES dict below and re-run
"""
import math

# Link lengths from URDF joint origins (meters)
L1 = 0.089852   # shoulder -> elbow (arm1_link)
L2 = 0.07865    # elbow -> wrist   (arm2_link)
L3 = 0.0302     # wrist -> gripper tip (approximate)
Z_SHOULDER = 0.05 + 0.043622  # height of shoulder joint above base_link

# ── Joint limits (radians) ────────────────────────────────────────────────────
LIMITS = {
    "base_rotating_waste_joint":  (-1.57, 1.57),
    "rotating_waste_arm1_joint":  (-1.57, 1.57),
    "arm1_arm2_joint":            (-2.35, 0.78),
    "arm2_gripper_base_joint":    (-1.57, 1.57),
}

# ── Poses to evaluate ─────────────────────────────────────────────────────────
# Format: (j0_deg, j1_deg, j2_deg, j3_deg)
# j0 = base yaw  (does not affect XZ FK, only changes which side the arm faces)
# j1 = shoulder  (around +Y — positive tilts arm forward)
# j2 = elbow     (around +Y — positive bends arm forward)
# j3 = wrist     (around -Y — sign is FLIPPED in FK: effective = -j3)
POSES = {
    "home":   (20,   37,  -72,   53),
    "grab":   ( 5,   56,   40,  -79),
    "lift_1": ( 5,   52.5, 40,  -79.5),   # intermediate step 1
    "lift_2": ( 5,   49,   40,  -80),     # intermediate step 2
    "lift":   ( 5,   45,   40,  -81),
    "place1": (-16,  12, -100,   67),
    "place2": ( 23,  -4,  -92,   75),
    "place3": ( 60,  16,  -79,   81),
}


def fk(j1_deg, j2_deg, j3_deg):
    """Return (x_tip, z_tip, x_wrist, z_wrist) in meters above base_link origin."""
    j1 = math.radians(j1_deg)
    j2 = math.radians(j2_deg)
    j3 = math.radians(j3_deg)

    # Elbow position
    x_e = L1 * math.sin(j1)
    z_e = Z_SHOULDER + L1 * math.cos(j1)

    # Wrist position  (arm1 + arm2 both rotate around +Y)
    angle_w = j1 + j2
    x_w = x_e + L2 * math.sin(angle_w)
    z_w = z_e + L2 * math.cos(angle_w)

    # Gripper tip  (wrist joint is around -Y, so contribution is -j3)
    angle_tip = j1 + j2 + (-j3)
    x_tip = x_w + L3 * math.sin(angle_tip)
    z_tip = z_w + L3 * math.cos(angle_tip)

    return x_tip, z_tip, x_w, z_w


def check_limits(j0_deg, j1_deg, j2_deg, j3_deg):
    joints = {
        "base_rotating_waste_joint": math.radians(j0_deg),
        "rotating_waste_arm1_joint": math.radians(j1_deg),
        "arm1_arm2_joint":           math.radians(j2_deg),
        "arm2_gripper_base_joint":   math.radians(j3_deg),
    }
    warnings = []
    for name, val in joints.items():
        lo, hi = LIMITS[name]
        if not (lo <= val <= hi):
            warnings.append(f"  !! {name} = {math.degrees(val):.1f}° out of range [{math.degrees(lo):.0f}, {math.degrees(hi):.0f}]")
    return warnings


def main():
    print(f"{'Pose':<10} {'J0':>6} {'J1':>6} {'J2':>6} {'J3':>6}  |  "
          f"{'x (cm)':>8} {'z (cm)':>8}  | notes")
    print("-" * 80)

    prev_x, prev_z = None, None
    for name, (j0, j1, j2, j3) in POSES.items():
        x, z, xw, zw = fk(j1, j2, j3)
        delta = ""
        if prev_x is not None:
            dz = (z - prev_z) * 1000
            dx = (x - prev_x) * 1000
            delta = f"  dz={dz:+.1f}mm  dx={dx:+.1f}mm"
        warnings = check_limits(j0, j1, j2, j3)
        warn_str = " LIMIT!" if warnings else ""
        print(f"{name:<10} {j0:>6.1f} {j1:>6.1f} {j2:>6.1f} {j3:>6.1f}  |  "
              f"{x*100:>8.1f} {z*100:>8.1f}{delta}{warn_str}")
        for w in warnings:
            print(w)
        prev_x, prev_z = x, z

    print()
    print("Notes:")
    print("  x = forward distance from robot base (cm)")
    print("  z = height above base_link (cm)")
    print("  J0=base yaw does not change gripper x/z in this simplified model")
    print()
    print("To convert degrees to radians for the action server:")
    print("  rad = deg * pi/180  (or use math.radians(deg))")
    for name, (j0, j1, j2, j3) in POSES.items():
        print(f"  {name}: [{math.radians(j0):.4f}, {math.radians(j1):.4f}, "
              f"{math.radians(j2):.4f}, {math.radians(j3):.4f}]")


if __name__ == "__main__":
    main()
