#!/usr/bin/env python3

import math
import time
import threading

import gpiod
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class StepperControlNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_stepper_node')

        # =========================
        # ROS parameters
        # =========================
        self.declare_parameter('wheel_radius', 0.065)              # meters
        self.declare_parameter('wheel_base', 0.20)                 # meters
        self.declare_parameter('steps_per_rev', 200)               # full steps/rev of motor
        self.declare_parameter('microsteps', 16)                   # 1,2,4,8,16 (match A4988 switches)
        self.declare_parameter('max_steps_per_sec', 4000.0)

        self.declare_parameter('accel_steps_per_sec2', 3500.0)
        self.declare_parameter('decel_steps_per_sec2', 3500.0)
        self.declare_parameter('cmd_vel_timeout', 0.2)

        self.declare_parameter('chip_name', 'gpiochip4')

        # Left motor pins
        self.declare_parameter('left_en_pin', 22)
        self.declare_parameter('left_dir_pin', 23)
        self.declare_parameter('left_step_pin', 24)

        # Right motor pins
        self.declare_parameter('right_en_pin', 12)
        self.declare_parameter('right_dir_pin', 5)
        self.declare_parameter('right_step_pin', 6)

        self.declare_parameter('enable_active_low', True)

        # Direction inversion
        self.declare_parameter('left_dir_inverted', False)
        self.declare_parameter('right_dir_inverted', True)

        # Topics
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        # =========================
        # Load parameters
        # =========================
        self.wheel_radius = float(self.get_parameter('wheel_radius').value)
        self.wheel_base = float(self.get_parameter('wheel_base').value)
        self.steps_per_rev = int(self.get_parameter('steps_per_rev').value)
        self.microsteps = int(self.get_parameter('microsteps').value)
        self.max_steps_per_sec = float(self.get_parameter('max_steps_per_sec').value)

        self.accel_steps_per_sec2 = float(self.get_parameter('accel_steps_per_sec2').value)
        self.decel_steps_per_sec2 = float(self.get_parameter('decel_steps_per_sec2').value)
        self.cmd_vel_timeout = float(self.get_parameter('cmd_vel_timeout').value)

        self.chip_name = str(self.get_parameter('chip_name').value)

        self.left_en_pin = int(self.get_parameter('left_en_pin').value)
        self.left_dir_pin = int(self.get_parameter('left_dir_pin').value)
        self.left_step_pin = int(self.get_parameter('left_step_pin').value)

        self.right_en_pin = int(self.get_parameter('right_en_pin').value)
        self.right_dir_pin = int(self.get_parameter('right_dir_pin').value)
        self.right_step_pin = int(self.get_parameter('right_step_pin').value)

        self.enable_active_low = bool(self.get_parameter('enable_active_low').value)

        self.left_dir_inverted = bool(self.get_parameter('left_dir_inverted').value)
        self.right_dir_inverted = bool(self.get_parameter('right_dir_inverted').value)

        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)

        valid_microsteps = [1, 2, 4, 8, 16]
        if self.microsteps not in valid_microsteps:
            self.get_logger().warn(
                f'Invalid microsteps={self.microsteps}. Using 16 instead. '
                f'Valid values: {valid_microsteps}'
            )
            self.microsteps = 16

        # =========================
        # GPIO setup
        # =========================
        self.chip = gpiod.Chip(self.chip_name)

        self.left_en = self.chip.get_line(self.left_en_pin)
        self.left_dir = self.chip.get_line(self.left_dir_pin)
        self.left_step = self.chip.get_line(self.left_step_pin)

        self.right_en = self.chip.get_line(self.right_en_pin)
        self.right_dir = self.chip.get_line(self.right_dir_pin)
        self.right_step = self.chip.get_line(self.right_step_pin)

        self.left_en.request(
            consumer='stepper_control',
            type=gpiod.LINE_REQ_DIR_OUT,
            default_vals=[1]
        )
        self.left_dir.request(
            consumer='stepper_control',
            type=gpiod.LINE_REQ_DIR_OUT,
            default_vals=[0]
        )
        self.left_step.request(
            consumer='stepper_control',
            type=gpiod.LINE_REQ_DIR_OUT,
            default_vals=[0]
        )

        self.right_en.request(
            consumer='stepper_control',
            type=gpiod.LINE_REQ_DIR_OUT,
            default_vals=[1]
        )
        self.right_dir.request(
            consumer='stepper_control',
            type=gpiod.LINE_REQ_DIR_OUT,
            default_vals=[0]
        )
        self.right_step.request(
            consumer='stepper_control',
            type=gpiod.LINE_REQ_DIR_OUT,
            default_vals=[0]
        )

        self.enable_drivers(True)

        # =========================
        # Motion state
        # =========================
        self.target_left_sps = 0.0
        self.target_right_sps = 0.0
        self.current_left_sps = 0.0
        self.current_right_sps = 0.0

        self.last_cmd_time = time.monotonic()
        self.lock = threading.Lock()
        self.running = True

        # =========================
        # ROS interfaces
        # =========================
        self.subscription = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_vel_callback,
            10
        )

        self.watchdog_timer = self.create_timer(0.05, self.watchdog_callback)

        # =========================
        # Motor threads
        # =========================
        self.left_thread = threading.Thread(
            target=self.motor_loop,
            args=('left',),
            daemon=True
        )
        self.right_thread = threading.Thread(
            target=self.motor_loop,
            args=('right',),
            daemon=True
        )

        self.left_thread.start()
        self.right_thread.start()

        self.get_logger().info('Stepper control node started.')
        self.get_logger().info(
            f'Using microsteps={self.microsteps}, '
            f'steps_per_mech_rev={self.steps_per_rev * self.microsteps}'
        )

    def enable_drivers(self, enable: bool):
        if self.enable_active_low:
            value = 0 if enable else 1
        else:
            value = 1 if enable else 0

        self.left_en.set_value(value)
        self.right_en.set_value(value)

    def cmd_vel_callback(self, msg: Twist):
        linear_x = float(msg.linear.x)
        angular_z = float(msg.angular.z)

        # Differential drive wheel linear velocities
        v_left = linear_x - (angular_z * self.wheel_base / 2.0)
        v_right = linear_x + (angular_z * self.wheel_base / 2.0)

        wheel_circumference = 2.0 * math.pi * self.wheel_radius
        left_rev_per_sec = v_left / wheel_circumference
        right_rev_per_sec = v_right / wheel_circumference

        # IMPORTANT:
        # This is where microsteps affects the required pulse rate.
        steps_per_mech_rev = self.steps_per_rev * self.microsteps
        left_sps = left_rev_per_sec * steps_per_mech_rev
        right_sps = right_rev_per_sec * steps_per_mech_rev

        left_sps = max(-self.max_steps_per_sec, min(self.max_steps_per_sec, left_sps))
        right_sps = max(-self.max_steps_per_sec, min(self.max_steps_per_sec, right_sps))

        with self.lock:
            self.target_left_sps = left_sps
            self.target_right_sps = right_sps
            self.last_cmd_time = time.monotonic()

    def watchdog_callback(self):
        if time.monotonic() - self.last_cmd_time > self.cmd_vel_timeout:
            with self.lock:
                self.target_left_sps = 0.0
                self.target_right_sps = 0.0

    def ramp_toward(self, current: float, target: float, dt: float) -> float:
        delta = target - current

        if abs(target) < abs(current):
            max_delta = self.decel_steps_per_sec2 * dt
        else:
            max_delta = self.accel_steps_per_sec2 * dt

        if delta > max_delta:
            return current + max_delta
        if delta < -max_delta:
            return current - max_delta
        return target

    def set_direction(self, dir_line, positive_direction: bool, inverted: bool):
        gpio_value = 1 if positive_direction else 0
        if inverted:
            gpio_value = 0 if gpio_value == 1 else 1
        dir_line.set_value(gpio_value)

    def motor_loop(self, side: str):
        last_time = time.monotonic()

        if side == 'left':
            dir_line = self.left_dir
            step_line = self.left_step
            dir_inverted = self.left_dir_inverted
        else:
            dir_line = self.right_dir
            step_line = self.right_step
            dir_inverted = self.right_dir_inverted

        while self.running:
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            with self.lock:
                if side == 'left':
                    self.current_left_sps = self.ramp_toward(
                        self.current_left_sps,
                        self.target_left_sps,
                        dt
                    )
                    current_sps = self.current_left_sps
                else:
                    self.current_right_sps = self.ramp_toward(
                        self.current_right_sps,
                        self.target_right_sps,
                        dt
                    )
                    current_sps = self.current_right_sps

            if abs(current_sps) < 1.0:
                time.sleep(0.002)
                continue

            positive_direction = current_sps >= 0.0
            self.set_direction(dir_line, positive_direction, dir_inverted)

            freq = abs(current_sps)
            period = 1.0 / freq
            half_period = period / 2.0

            step_line.set_value(1)
            time.sleep(half_period)
            step_line.set_value(0)
            time.sleep(half_period)

    def destroy_node(self):
        self.running = False
        time.sleep(0.05)

        try:
            self.enable_drivers(False)
        except Exception:
            pass

        for line in [
            self.left_step, self.left_dir, self.left_en,
            self.right_step, self.right_dir, self.right_en
        ]:
            try:
                line.release()
            except Exception:
                pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StepperControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()