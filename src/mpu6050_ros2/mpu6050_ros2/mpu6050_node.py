#!/usr/bin/env python3

import math
import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion

try:
    import smbus2 as smbus_lib
except ImportError:
    import smbus as smbus_lib


MPU6050_ADDR = 0x68

PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
INT_ENABLE = 0x38

ACCEL_XOUT_H = 0x3B
TEMP_OUT_H = 0x41
GYRO_XOUT_H = 0x43


def euler_to_quaternion(roll, pitch, yaw):
    q = Quaternion()

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy

    return q


@dataclass
class KalmanAngle:
    q_angle: float = 0.001
    q_bias: float = 0.003
    r_measure: float = 0.03

    angle: float = 0.0
    bias: float = 0.0
    rate: float = 0.0

    p00: float = 0.0
    p01: float = 0.0
    p10: float = 0.0
    p11: float = 0.0

    def update(self, new_angle, new_rate, dt):
        self.rate = new_rate - self.bias
        self.angle += dt * self.rate

        self.p00 += dt * (dt * self.p11 - self.p01 - self.p10 + self.q_angle)
        self.p01 -= dt * self.p11
        self.p10 -= dt * self.p11
        self.p11 += self.q_bias * dt

        s = self.p00 + self.r_measure
        k0 = self.p00 / s
        k1 = self.p10 / s

        y = new_angle - self.angle
        self.angle += k0 * y
        self.bias += k1 * y

        p00_temp = self.p00
        p01_temp = self.p01

        self.p00 -= k0 * p00_temp
        self.p01 -= k0 * p01_temp
        self.p10 -= k1 * p00_temp
        self.p11 -= k1 * p01_temp

        return self.angle


class MPU6050:
    def __init__(self, bus_num=1, address=MPU6050_ADDR):
        self.bus = smbus_lib.SMBus(bus_num)
        self.address = address

        self.accel_scale = 16384.0   # +/-2g
        self.gyro_scale = 131.0      # +/-250 deg/s

        self.gyro_bias_x = 0.0
        self.gyro_bias_y = 0.0
        self.gyro_bias_z = 0.0

    def write_byte(self, reg, value):
        self.bus.write_byte_data(self.address, reg, value)

    def read_i2c_word(self, reg):
        high = self.bus.read_byte_data(self.address, reg)
        low = self.bus.read_byte_data(self.address, reg + 1)
        value = (high << 8) | low
        if value >= 0x8000:
            value = -((65535 - value) + 1)
        return value

    def initialize(self):
        self.write_byte(PWR_MGMT_1, 0x00)
        time.sleep(0.1)

        self.write_byte(SMPLRT_DIV, 0x07)
        self.write_byte(CONFIG, 0x03)
        self.write_byte(GYRO_CONFIG, 0x00)
        self.write_byte(ACCEL_CONFIG, 0x00)
        self.write_byte(INT_ENABLE, 0x00)

    def read_raw(self):
        ax = self.read_i2c_word(ACCEL_XOUT_H)
        ay = self.read_i2c_word(ACCEL_XOUT_H + 2)
        az = self.read_i2c_word(ACCEL_XOUT_H + 4)

        gx = self.read_i2c_word(GYRO_XOUT_H)
        gy = self.read_i2c_word(GYRO_XOUT_H + 2)
        gz = self.read_i2c_word(GYRO_XOUT_H + 4)

        temp_raw = self.read_i2c_word(TEMP_OUT_H)

        return ax, ay, az, gx, gy, gz, temp_raw

    def read_scaled(self):
        ax, ay, az, gx, gy, gz, temp_raw = self.read_raw()

        ax_g = ax / self.accel_scale
        ay_g = ay / self.accel_scale
        az_g = az / self.accel_scale

        gx_dps = gx / self.gyro_scale - self.gyro_bias_x
        gy_dps = gy / self.gyro_scale - self.gyro_bias_y
        gz_dps = gz / self.gyro_scale - self.gyro_bias_z

        temp_c = (temp_raw / 340.0) + 36.53

        return ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, temp_c

    def calibrate_gyro(self, samples=1000, delay=0.002):
        print("Keep MPU6050 still. Calibrating gyro bias...")

        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0

        for _ in range(samples):
            _, _, _, gx, gy, gz, _ = self.read_raw()
            sum_x += gx / self.gyro_scale
            sum_y += gy / self.gyro_scale
            sum_z += gz / self.gyro_scale
            time.sleep(delay)

        self.gyro_bias_x = sum_x / samples
        self.gyro_bias_y = sum_y / samples
        self.gyro_bias_z = sum_z / samples

        print(f"Gyro bias X: {self.gyro_bias_x:.4f} dps")
        print(f"Gyro bias Y: {self.gyro_bias_y:.4f} dps")
        print(f"Gyro bias Z: {self.gyro_bias_z:.4f} dps")


class MPU6050Node(Node):
    def __init__(self):
        super().__init__('mpu6050_node')

        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('i2c_address', 0x68)
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('stationary_gyro_threshold_dps', 0.8)
        self.declare_parameter('stationary_accel_threshold_g', 0.08)
        self.declare_parameter('yaw_bias_adapt_alpha', 0.001)

        bus_num = self.get_parameter('i2c_bus').value
        addr = self.get_parameter('i2c_address').value
        self.frame_id = self.get_parameter('frame_id').value
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.stationary_gyro_threshold = float(
            self.get_parameter('stationary_gyro_threshold_dps').value
        )
        self.stationary_accel_threshold = float(
            self.get_parameter('stationary_accel_threshold_g').value
        )
        self.yaw_bias_adapt_alpha = float(
            self.get_parameter('yaw_bias_adapt_alpha').value
        )

        self.angle_pub = self.create_publisher(Float32, '/gyro_angle', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data_raw', 10)

        self.mpu = MPU6050(bus_num=bus_num, address=addr)
        self.mpu.initialize()
        time.sleep(0.5)
        self.mpu.calibrate_gyro(samples=1000, delay=0.002)

        self.kalman_roll = KalmanAngle()
        self.kalman_pitch = KalmanAngle()

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.last_time = time.time()

        period = 1.0 / self.publish_rate
        self.timer = self.create_timer(period, self.update)

        self.get_logger().info("MPU6050 node started")

    def update(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        if dt <= 0.0 or dt > 0.5:
            return

        try:
            ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, _ = self.mpu.read_scaled()
        except Exception as e:
            self.get_logger().error(f"MPU6050 read failed: {e}")
            return

        roll_acc = math.degrees(math.atan2(ay_g, az_g))
        pitch_acc = math.degrees(
            math.atan2(-ax_g, math.sqrt(ay_g * ay_g + az_g * az_g))
        )

        roll_deg = self.kalman_roll.update(roll_acc, gx_dps, dt)
        pitch_deg = self.kalman_pitch.update(pitch_acc, gy_dps, dt)

        self.roll = math.radians(roll_deg)
        self.pitch = math.radians(pitch_deg)

        accel_mag = math.sqrt(ax_g * ax_g + ay_g * ay_g + az_g * az_g)
        is_stationary = (
            abs(gx_dps) < self.stationary_gyro_threshold and
            abs(gy_dps) < self.stationary_gyro_threshold and
            abs(gz_dps) < self.stationary_gyro_threshold and
            abs(accel_mag - 1.0) < self.stationary_accel_threshold
        )

        if is_stationary:
            self.mpu.gyro_bias_z = (
                (1.0 - self.yaw_bias_adapt_alpha) * self.mpu.gyro_bias_z
                + self.yaw_bias_adapt_alpha * (self.mpu.gyro_bias_z + gz_dps)
            )
            gz_dps = 0.0

        self.yaw += math.radians(gz_dps) * dt

        while self.yaw > math.pi:
            self.yaw -= 2.0 * math.pi
        while self.yaw < -math.pi:
            self.yaw += 2.0 * math.pi

        yaw_deg = math.degrees(self.yaw)

        angle_msg = Float32()
        angle_msg.data = float(yaw_deg)
        self.angle_pub.publish(angle_msg)

        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = self.frame_id

        imu_msg.orientation = euler_to_quaternion(self.roll, self.pitch, self.yaw)
        imu_msg.orientation_covariance = [
            0.02, 0.0, 0.0,
            0.0, 0.02, 0.0,
            0.0, 0.0, 0.15
        ]

        imu_msg.linear_acceleration.x = ax_g * 9.80665
        imu_msg.linear_acceleration.y = ay_g * 9.80665
        imu_msg.linear_acceleration.z = az_g * 9.80665
        imu_msg.linear_acceleration_covariance = [
            0.2, 0.0, 0.0,
            0.0, 0.2, 0.0,
            0.0, 0.0, 0.2
        ]

        imu_msg.angular_velocity.x = math.radians(gx_dps)
        imu_msg.angular_velocity.y = math.radians(gy_dps)
        imu_msg.angular_velocity.z = math.radians(gz_dps)
        imu_msg.angular_velocity_covariance = [
            0.02, 0.0, 0.0,
            0.0, 0.02, 0.0,
            0.0, 0.0, 0.03
        ]

        self.imu_pub.publish(imu_msg)

    def destroy_node(self):
        try:
            self.mpu.bus.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MPU6050Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()