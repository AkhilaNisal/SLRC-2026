import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist

MAX_RANGE = 0.15

class ObstacleSeeker(Node):
    def __init__(self):
        super().__init__('obstacle_seeker')

        self.__publisher = self.create_publisher(Twist, 'cmd_vel', 1)

        self.create_subscription(Range, 'left_sensor', self.__left_sensor_callback, 1)
        self.create_subscription(Range, 'right_sensor', self.__right_sensor_callback, 1)
        
        self.__left_sensor_value = MAX_RANGE
        self.__right_sensor_value = MAX_RANGE

    def __left_sensor_callback(self, message):
        self.__left_sensor_value = message.range

    def __right_sensor_callback(self, message):
        self.__right_sensor_value = message.range

        command_message = Twist()

        # Drive forward primarily
        command_message.linear.x = 0.1

        # If left sensor detects something closer, turn towards left (+ angular Z)
        # If right sensor detects something closer, turn towards right (- angular Z)
        # We can steer proportionally to the difference.
        
        diff = self.__left_sensor_value - self.__right_sensor_value
        
        if self.__left_sensor_value < MAX_RANGE or self.__right_sensor_value < MAX_RANGE:
            # Turn towards the closer obstacle
            # Example: if left is 0.05 and right is 0.15, diff = -0.10, we want positive angular.z (left turn)
            command_message.angular.z = -5.0 * diff

        self.__publisher.publish(command_message)

def main(args=None):
    rclpy.init(args=args)
    seeker = ObstacleSeeker()
    rclpy.spin(seeker)
    seeker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
