#!/usr/bin/env python3
import os
import signal
import subprocess
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TaskManager(Node):
    def __init__(self):
        super().__init__('task_manager')

        # =========================
        # Parameters
        # =========================
        self.declare_parameter('task2_package', 'your_package_name')
        self.declare_parameter('task2_executable', 'task2_with_arm')

        self.declare_parameter('task3_package', 'your_package_name')
        self.declare_parameter('task3_executable', 'task3')

        self.declare_parameter('task2_status_topic', '/task2/status')
        self.declare_parameter('task3_status_topic', '/task3/status')

        self.declare_parameter('startup_delay_sec', 2.0)
        self.declare_parameter('shutdown_wait_sec', 3.0)
        self.declare_parameter('use_shell', False)

        self.task2_package = str(self.get_parameter('task2_package').value)
        self.task2_executable = str(self.get_parameter('task2_executable').value)

        self.task3_package = str(self.get_parameter('task3_package').value)
        self.task3_executable = str(self.get_parameter('task3_executable').value)

        self.task2_status_topic = str(self.get_parameter('task2_status_topic').value)
        self.task3_status_topic = str(self.get_parameter('task3_status_topic').value)

        self.startup_delay_sec = float(self.get_parameter('startup_delay_sec').value)
        self.shutdown_wait_sec = float(self.get_parameter('shutdown_wait_sec').value)
        self.use_shell = bool(self.get_parameter('use_shell').value)

        # =========================
        # State machine
        # =========================
        self.STAGE_START_TASK2 = 'START_TASK2'
        self.STAGE_WAIT_TASK2 = 'WAIT_TASK2'
        self.STAGE_STOP_TASK2 = 'STOP_TASK2'
        self.STAGE_START_TASK3 = 'START_TASK3'
        self.STAGE_WAIT_TASK3 = 'WAIT_TASK3'
        self.STAGE_STOP_TASK3 = 'STOP_TASK3'
        self.STAGE_FINISHED = 'FINISHED'

        self.stage = self.STAGE_START_TASK2

        # =========================
        # Process handles
        # =========================
        self.task2_proc = None
        self.task3_proc = None

        # =========================
        # Completion flags
        # =========================
        self.task2_done = False
        self.task3_done = False

        # =========================
        # Subscribers
        # =========================
        self.task2_status_sub = self.create_subscription(
            String,
            self.task2_status_topic,
            self.task2_status_cb,
            10
        )

        self.task3_status_sub = self.create_subscription(
            String,
            self.task3_status_topic,
            self.task3_status_cb,
            10
        )

        # =========================
        # Timer
        # =========================
        self.timer = self.create_timer(0.5, self.loop)

        self.get_logger().info("Task manager started.")
        self.get_logger().info(
            f"Task2: {self.task2_package}/{self.task2_executable}, "
            f"Task3: {self.task3_package}/{self.task3_executable}"
        )

    def task2_status_cb(self, msg: String):
        if msg.data.strip().upper() == 'DONE':
            if not self.task2_done:
                self.get_logger().info("Received Task 2 DONE.")
            self.task2_done = True

    def task3_status_cb(self, msg: String):
        if msg.data.strip().upper() == 'DONE':
            if not self.task3_done:
                self.get_logger().info("Received Task 3 DONE.")
            self.task3_done = True

    def start_node_process(self, package_name: str, executable_name: str, label: str):
        cmd = ['ros2', 'run', package_name, executable_name]
        self.get_logger().info(f"Starting {label}: {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid
            )
            return proc
        except Exception as e:
            self.get_logger().error(f"Failed to start {label}: {e}")
            return None

    def stop_node_process(self, proc, label: str):
        if proc is None:
            return

        if proc.poll() is not None:
            self.get_logger().info(f"{label} already stopped.")
            return

        self.get_logger().info(f"Stopping {label}...")

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except Exception as e:
            self.get_logger().warn(f"SIGINT failed for {label}: {e}")

        t0 = time.time()
        while time.time() - t0 < self.shutdown_wait_sec:
            if proc.poll() is not None:
                self.get_logger().info(f"{label} stopped cleanly.")
                return
            time.sleep(0.1)

        self.get_logger().warn(f"{label} did not stop after SIGINT. Sending SIGTERM...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception as e:
            self.get_logger().warn(f"SIGTERM failed for {label}: {e}")

        t0 = time.time()
        while time.time() - t0 < 2.0:
            if proc.poll() is not None:
                self.get_logger().info(f"{label} stopped after SIGTERM.")
                return
            time.sleep(0.1)

        self.get_logger().warn(f"{label} still alive. Sending SIGKILL...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception as e:
            self.get_logger().error(f"SIGKILL failed for {label}: {e}")

    def loop(self):
        if self.stage == self.STAGE_START_TASK2:
            self.task2_done = False
            self.task2_proc = self.start_node_process(
                self.task2_package,
                self.task2_executable,
                'task2'
            )

            if self.task2_proc is None:
                self.get_logger().error("Could not start task2.")
                self.stage = self.STAGE_FINISHED
                return

            time.sleep(self.startup_delay_sec)
            self.stage = self.STAGE_WAIT_TASK2
            self.get_logger().info("Waiting for Task 2 completion...")

        elif self.stage == self.STAGE_WAIT_TASK2:
            if self.task2_proc is not None and self.task2_proc.poll() is not None:
                self.get_logger().error("task2 exited unexpectedly.")
                self.stage = self.STAGE_FINISHED
                return

            if self.task2_done:
                self.stage = self.STAGE_STOP_TASK2

        elif self.stage == self.STAGE_STOP_TASK2:
            self.stop_node_process(self.task2_proc, 'task2')
            self.task2_proc = None
            self.stage = self.STAGE_START_TASK3

        elif self.stage == self.STAGE_START_TASK3:
            self.task3_done = False
            self.task3_proc = self.start_node_process(
                self.task3_package,
                self.task3_executable,
                'task3'
            )

            if self.task3_proc is None:
                self.get_logger().error("Could not start task3.")
                self.stage = self.STAGE_FINISHED
                return

            time.sleep(self.startup_delay_sec)
            self.stage = self.STAGE_WAIT_TASK3
            self.get_logger().info("Waiting for Task 3 completion...")

        elif self.stage == self.STAGE_WAIT_TASK3:
            if self.task3_proc is not None and self.task3_proc.poll() is not None:
                self.get_logger().error("task3 exited unexpectedly.")
                self.stage = self.STAGE_FINISHED
                return

            if self.task3_done:
                self.stage = self.STAGE_STOP_TASK3

        elif self.stage == self.STAGE_STOP_TASK3:
            self.stop_node_process(self.task3_proc, 'task3')
            self.task3_proc = None
            self.stage = self.STAGE_FINISHED
            self.get_logger().info("Task sequence completed successfully.")

        elif self.stage == self.STAGE_FINISHED:
            pass

    def destroy_node(self):
        try:
            self.stop_node_process(self.task2_proc, 'task2')
        except Exception:
            pass

        try:
            self.stop_node_process(self.task3_proc, 'task3')
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TaskManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()