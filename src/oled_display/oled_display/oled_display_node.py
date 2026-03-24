#!/usr/bin/env python3

import textwrap

import board
import busio
import adafruit_ssd1306

from PIL import Image, ImageDraw, ImageFont

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class OledDisplayNode(Node):
    def __init__(self):
        super().__init__('oled_display_node')

        # Parameters
        self.declare_parameter('topic_name', '/oled_text')
        self.declare_parameter('i2c_address', 0x3C)
        self.declare_parameter('width', 128)
        self.declare_parameter('height', 64)
        self.declare_parameter('font_size', 12)
        self.declare_parameter('lines', 4)

        topic_name = self.get_parameter('topic_name').value
        i2c_address = self.get_parameter('i2c_address').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        self.font_size = self.get_parameter('font_size').value
        self.max_lines = self.get_parameter('lines').value

        # ROS subscriber
        self.subscription = self.create_subscription(
            String,
            topic_name,
            self.text_callback,
            10
        )

        # OLED init
        try:
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.oled = adafruit_ssd1306.SSD1306_I2C(
                width, height, self.i2c, addr=i2c_address
            )
            self.oled.fill(0)
            self.oled.show()
        except Exception as e:
            self.get_logger().error(f'Failed to initialize OLED: {e}')
            raise

        # Image buffer
        self.width = width
        self.height = height
        self.image = Image.new("1", (self.width, self.height))
        self.draw = ImageDraw.Draw(self.image)

        try:
            self.font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                self.font_size
            )
        except Exception:
            self.font = ImageFont.load_default()

        self.display_text("OLED node ready")

        self.get_logger().info(f'OLED display node started. Subscribed to: {topic_name}')

    def display_text(self, text: str):
        self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

        wrapped = textwrap.wrap(text, width=20)
        wrapped = wrapped[:self.max_lines]

        y = 0
        line_height = self.font_size + 2

        for line in wrapped:
            self.draw.text((0, y), line, font=self.font, fill=255)
            y += line_height

        self.oled.image(self.image)
        self.oled.show()

    def text_callback(self, msg: String):
        self.get_logger().info(f'Received: "{msg.data}"')
        self.display_text(msg.data)


def main(args=None):
    rclpy.init(args=args)
    node = OledDisplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.display_text("Shutting down")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()