
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from adafruit_servokit import ServoKit
import time


class ServoControlNode(Node):
    def __init__(self):
        super().__init__('servo_control_node')
        
        # Parameters
        self.declare_parameter('num_channels', 16)
        self.declare_parameter('servo_topic', '/servo_angles')
        
        self.num_channels = self.get_parameter('num_channels').value
        self.servo_topic = self.get_parameter('servo_topic').value
        
        # Initialize ServoKit
        try:
            self.kit = ServoKit(channels=self.num_channels)
            self.get_logger().info(f'✅ ServoKit initialized with {self.num_channels} channels')
        except Exception as e:
            self.get_logger().error(f'❌ Failed to initialize ServoKit: {e}')
            raise
        
        # Subscriber for servo commands
        self.subscription = self.create_subscription(
            Float32MultiArray,
            self.servo_topic,
            self.servo_callback,
            10
        )
        
        self.get_logger().info(f'✅ Subscribed to: {self.servo_topic}')
        self.get_logger().info('✅ Servo control node ready!')
    
    def servo_callback(self, msg: Float32MultiArray):
        """Set servo angles from received message"""
        try:
            for ch, angle in enumerate(msg.data):
                if ch < self.num_channels:
                    # Clamp angle to 0-180
                    angle = max(0, min(180, angle))
                    self.kit.servo[ch].angle = angle
            
            self.get_logger().debug(f'Set servo angles: {list(msg.data[:self.num_channels])}')
            
        except Exception as e:
            self.get_logger().error(f'Error setting servo angles: {e}')


def main(args=None):
    """Main entry point for ROS 2 node"""
    rclpy.init(args=args)
    node = ServoControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()