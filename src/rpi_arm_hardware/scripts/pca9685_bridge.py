#!/usr/bin/env python3

import math
import socket
import time
from adafruit_servokit import ServoKit

NUM_JOINTS = 4

# PCA9685 channels used for your 4 arm joints
SERVO_CHANNELS = [0, 1, 2, 3]

# ---------------- Joint limits in RAD ----------------
joint_lower_rad = [-1.57, -1.57, -0.972, -1.57]
joint_upper_rad = [ 1.57,  1.57,  2.172,  1.57]

# If reversed, min/max are swapped
servo_min_deg = [0, 180, 0, 180]
servo_max_deg = [180, 0, 180, 0]

# Zero trim in servo degrees
zero_offset_deg = [0, 1, -10, 0]

# Smoothing and update timing
SERVO_PERIOD_SEC = 0.02      # 50 Hz
ALPHA = 0.18
DEADBAND_DEG = 1
FEEDBACK_PERIOD_SEC = 0.05   # 20 Hz

HOST = "127.0.0.1"
PORT = 9999

kit = ServoKit(channels=16)

current_pos = [0, 0, 0, 0]      # integer servo degrees actually written
target_pos = [0, 0, 0, 0]       # integer servo target degrees
filtered_pos = [0.0, 0.0, 0.0, 0.0]


def clampf(x, lo, hi):
    return max(lo, min(hi, x))


def mapf(x, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return out_min
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def map_joint_to_servo(joint_index, joint_rad):
    jr = clampf(joint_rad, joint_lower_rad[joint_index], joint_upper_rad[joint_index])

    sd = mapf(
        jr,
        joint_lower_rad[joint_index], joint_upper_rad[joint_index],
        float(servo_min_deg[joint_index]), float(servo_max_deg[joint_index])
    )

    sd += float(zero_offset_deg[joint_index])

    smin = min(servo_min_deg[joint_index], servo_max_deg[joint_index])
    smax = max(servo_min_deg[joint_index], servo_max_deg[joint_index])
    sd = clampf(sd, smin, smax)

    return int(sd + 0.5)


def map_servo_to_joint(joint_index, servo_deg):
    sd = servo_deg - zero_offset_deg[joint_index]

    smin = min(servo_min_deg[joint_index], servo_max_deg[joint_index])
    smax = max(servo_min_deg[joint_index], servo_max_deg[joint_index])

    if sd < smin:
        sd = smin
    if sd > smax:
        sd = smax

    jr = mapf(
        float(sd),
        float(servo_min_deg[joint_index]), float(servo_max_deg[joint_index]),
        joint_lower_rad[joint_index], joint_upper_rad[joint_index]
    )

    return jr


def init_home():
    for i in range(NUM_JOINTS):
        home = map_joint_to_servo(i, 0.0)
        current_pos[i] = home
        target_pos[i] = home
        filtered_pos[i] = float(home)

        ch = SERVO_CHANNELS[i]
        kit.servo[ch].set_pulse_width_range(500, 2500)
        kit.servo[ch].angle = current_pos[i]


def parse_positions(line):
    if not line.startswith("J:"):
        return

    body = line[2:].strip()
    parts = body.split(",")

    for i in range(min(NUM_JOINTS, len(parts))):
        try:
            cmd_rad = float(parts[i])
            target_pos[i] = map_joint_to_servo(i, cmd_rad)
        except ValueError:
            pass


def send_positions(conn):
    msg = "S:" + ",".join(f"{map_servo_to_joint(i, current_pos[i]):.3f}" for i in range(NUM_JOINTS)) + "\n"
    conn.sendall(msg.encode("utf-8"))


def servo_update():
    for i in range(NUM_JOINTS):
        filtered_pos[i] = filtered_pos[i] + ALPHA * (float(target_pos[i]) - filtered_pos[i])
        cmd_deg = int(filtered_pos[i] + 0.5)

        if abs(cmd_deg - current_pos[i]) <= DEADBAND_DEG:
            continue

        current_pos[i] = cmd_deg
        ch = SERVO_CHANNELS[i]
        angle = max(0, min(180, current_pos[i]))
        kit.servo[ch].angle = angle


def run_server():
    init_home()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"PCA9685 bridge listening on {HOST}:{PORT}")

    while True:
        conn, addr = server.accept()
        print(f"Client connected: {addr}")

        conn.settimeout(0.001)
        rx_buffer = ""

        last_servo = time.monotonic()
        last_feedback = time.monotonic()

        try:
            while True:
                now = time.monotonic()

                # Read incoming data
                try:
                    data = conn.recv(256)
                    if not data:
                        break
                    rx_buffer += data.decode("utf-8", errors="ignore")
                except socket.timeout:
                    pass

                # Process complete lines
                while "\n" in rx_buffer:
                    line, rx_buffer = rx_buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        parse_positions(line)

                # Servo update at fixed rate
                if now - last_servo >= SERVO_PERIOD_SEC:
                    servo_update()
                    last_servo = now

                # Feedback at fixed rate
                if now - last_feedback >= FEEDBACK_PERIOD_SEC:
                    send_positions(conn)
                    last_feedback = now

                time.sleep(0.001)

        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            conn.close()
            print("Client disconnected")


if __name__ == "__main__":
    run_server()