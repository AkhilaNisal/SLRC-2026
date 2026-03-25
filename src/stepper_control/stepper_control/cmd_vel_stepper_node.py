#!/usr/bin/env python3

import math
import time
import threading

import gpiod
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray


class StepperControlNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_stepper_node')

        # =========================
        # ROS parameters
        # =========================
        self.declare_parameter('wheel_radius', 0.0325)          # meters
        self.declare_parameter('wheel_base', 0.20)              # meters
        self.declare_parameter('steps_per_rev', 200)            # full steps / motor rev
        self.declare_parameter('microsteps', 16)                # 1,2,4,8,16
        self.declare_parameter('max_steps_per_sec', 4000.0)

        self.declare_parameter('accel_steps_per_sec2', 3500.0)
        self.declare_parameter('decel_steps_per_sec2', 15000.0)
        self.declare_parameter('cmd_vel_timeout', 0.2)

        # Fixed stepping rate used during distance mode
        self.declare_parameter('distance_mode_sps', 800.0)

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
        self.declare_parameter('cmd_distance_topic', '/cmd_distance')

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
        self.distance_mode_sps = float(self.get_parameter('distance_mode_sps').value)

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
        self.cmd_distance_topic = str(self.get_parameter('cmd_distance_topic').value)

        valid_microsteps = [1, 2, 4, 8, 16]
        if self.microsteps not in valid_microsteps:
            self.get_logger().warn(
                f'Invalid microsteps={self.microsteps}. Using 16 instead. '
                f'Valid values: {valid_microsteps}'
            )
            self.microsteps = 16

        # =========================
        # Derived values
        # =========================
        self.steps_per_mech_rev = self.steps_per_rev * self.microsteps
        self.wheel_circumference = 2.0 * math.pi * self.wheel_radius
        self.meters_per_step = self.wheel_circumference / float(self.steps_per_mech_rev)

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
        # Modes:
        #   VEL  -> follow /cmd_vel
        #   DIST -> execute exact per-wheel distance command
        self.control_mode = 'VEL'

        # Velocity mode
        self.target_left_sps = 0.0
        self.target_right_sps = 0.0
        self.current_left_sps = 0.0
        self.current_right_sps = 0.0

        # Distance mode
        self.left_target_steps = 0
        self.right_target_steps = 0
        self.left_done_steps = 0
        self.right_done_steps = 0

        self.last_cmd_time = time.monotonic()
        self.lock = threading.Lock()
        self.running = True

        # =========================
        # ROS interfaces
        # =========================
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_vel_callback,
            10
        )

        self.cmd_distance_sub = self.create_subscription(
            Float32MultiArray,
            self.cmd_distance_topic,
            self.cmd_distance_callback,
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
            f'steps_per_mech_rev={self.steps_per_mech_rev}'
        )
        self.get_logger().info(
            f'wheel_circumference={self.wheel_circumference:.6f} m, '
            f'meters_per_step={self.meters_per_step:.8f} m'
        )
        self.get_logger().info(
            f'cmd_vel_topic={self.cmd_vel_topic}, cmd_distance_topic={self.cmd_distance_topic}'
        )

    def enable_drivers(self, enable: bool):
        if self.enable_active_low:
            value = 0 if enable else 1
        else:
            value = 1 if enable else 0

        self.left_en.set_value(value)
        self.right_en.set_value(value)

    def distance_m_to_steps(self, distance_m: float) -> int:
        return int(round(distance_m / self.meters_per_step))

    def cmd_vel_callback(self, msg: Twist):
        linear_x = float(msg.linear.x)
        angular_z = float(msg.angular.z)

        with self.lock:
            # Ignore /cmd_vel while exact distance motion is active
            if self.control_mode == 'DIST':
                return

        # Differential drive wheel linear velocities
        v_left = linear_x - (angular_z * self.wheel_base / 2.0)
        v_right = linear_x + (angular_z * self.wheel_base / 2.0)

        left_rev_per_sec = v_left / self.wheel_circumference
        right_rev_per_sec = v_right / self.wheel_circumference

        left_sps = left_rev_per_sec * self.steps_per_mech_rev
        right_sps = right_rev_per_sec * self.steps_per_mech_rev

        left_sps = max(-self.max_steps_per_sec, min(self.max_steps_per_sec, left_sps))
        right_sps = max(-self.max_steps_per_sec, min(self.max_steps_per_sec, right_sps))

        with self.lock:
            self.control_mode = 'VEL'
            self.target_left_sps = left_sps
            self.target_right_sps = right_sps
            self.last_cmd_time = time.monotonic()

    def cmd_distance_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn('cmd_distance requires [left_distance_m, right_distance_m]')
            return

        left_distance_m = float(msg.data[0])
        right_distance_m = float(msg.data[1])

        left_steps = self.distance_m_to_steps(left_distance_m)
        right_steps = self.distance_m_to_steps(right_distance_m)

        with self.lock:
            self.control_mode = 'DIST'

            # Stop velocity mode immediately
            self.target_left_sps = 0.0
            self.target_right_sps = 0.0
            self.current_left_sps = 0.0
            self.current_right_sps = 0.0

            # Set exact distance targets
            self.left_target_steps = left_steps
            self.right_target_steps = right_steps
            self.left_done_steps = 0
            self.right_done_steps = 0

            self.last_cmd_time = time.monotonic()

        self.get_logger().info(
            f'Received cmd_distance: '
            f'left={left_distance_m:.4f} m ({left_steps} steps), '
            f'right={right_distance_m:.4f} m ({right_steps} steps)'
        )

    def watchdog_callback(self):
        with self.lock:
            if self.control_mode == 'DIST':
                return

            if time.monotonic() - self.last_cmd_time > self.cmd_vel_timeout:
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

    def pulse_once(self, step_line, freq: float):
        if freq <= 0.0:
            return

        period = 1.0 / freq
        half_period = period / 2.0

        step_line.set_value(1)
        time.sleep(half_period)
        step_line.set_value(0)
        time.sleep(half_period)

    def finish_distance_mode_if_done(self):
        with self.lock:
            left_done = abs(self.left_done_steps) >= abs(self.left_target_steps)
            right_done = abs(self.right_done_steps) >= abs(self.right_target_steps)

            if left_done and right_done and self.control_mode == 'DIST':
                self.control_mode = 'VEL'
                self.target_left_sps = 0.0
                self.target_right_sps = 0.0
                self.current_left_sps = 0.0
                self.current_right_sps = 0.0

                left_dist = self.left_done_steps * self.meters_per_step
                right_dist = self.right_done_steps * self.meters_per_step

                self.get_logger().info(
                    f'Distance move complete: '
                    f'left={left_dist:.4f} m ({self.left_done_steps} steps), '
                    f'right={right_dist:.4f} m ({self.right_done_steps} steps)'
                )

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

            # =========================
            # Check current mode
            # =========================
            with self.lock:
                mode = self.control_mode

            # =========================
            # Distance mode
            # =========================
            if mode == 'DIST':
                with self.lock:
                    if side == 'left':
                        target_steps = self.left_target_steps
                        done_steps = self.left_done_steps
                    else:
                        target_steps = self.right_target_steps
                        done_steps = self.right_done_steps

                if abs(done_steps) >= abs(target_steps):
                    time.sleep(0.001)
                    self.finish_distance_mode_if_done()
                    continue

                positive_direction = target_steps >= 0
                self.set_direction(dir_line, positive_direction, dir_inverted)

                self.pulse_once(step_line, self.distance_mode_sps)

                with self.lock:
                    if side == 'left':
                        self.left_done_steps += 1 if positive_direction else -1
                    else:
                        self.right_done_steps += 1 if positive_direction else -1

                self.finish_distance_mode_if_done()
                continue

            # =========================
            # Velocity mode
            # =========================
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

            self.pulse_once(step_line, abs(current_sps))

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