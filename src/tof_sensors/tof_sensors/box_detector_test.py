#!/usr/bin/env python3

from collections import deque

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Range
from std_msgs.msg import Bool, Float32, String


class SimpleKalmanFilter:
    def __init__(self, process_variance: float, measurement_variance: float):
        self.q = process_variance
        self.r = measurement_variance
        self.x = None   # estimated value
        self.p = 1.0    # estimation error covariance

    def update(self, measurement: float) -> float:
        if self.x is None:
            self.x = measurement
            return self.x

        # Predict
        self.p = self.p + self.q

        # Update
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1.0 - k) * self.p

        return self.x


class EdgeBoxDetector:
    def __init__(self, name: str, node: Node):
        self.name = name
        self.node = node

        # Parameters
        self.drop_threshold = float(node.get_parameter('drop_threshold_m').value)
        self.rise_threshold = float(node.get_parameter('rise_threshold_m').value)
        self.plateau_tolerance = float(node.get_parameter('plateau_tolerance_m').value)
        self.min_plateau_samples = int(node.get_parameter('min_plateau_samples').value)
        self.max_gap_samples = int(node.get_parameter('max_gap_samples').value)
        self.min_valid_range = float(node.get_parameter('min_valid_range_m').value)
        self.max_valid_range = float(node.get_parameter('max_valid_range_m').value)
        self.prev_window_size = int(node.get_parameter('prev_window_size').value)
        self.curr_window_size = int(node.get_parameter('curr_window_size').value)
        self.median_window_size = int(node.get_parameter('median_window').value)
        self.min_event_samples = int(node.get_parameter('min_event_samples').value)

        # Kalman parameters
        self.kalman_process_variance = float(node.get_parameter('kalman_process_variance').value)
        self.kalman_measurement_variance = float(node.get_parameter('kalman_measurement_variance').value)

        # Raw and filtered history
        self.raw_window = deque(maxlen=self.median_window_size)
        self.filtered_history = deque(maxlen=100)

        # Kalman filter
        self.kalman = SimpleKalmanFilter(
            process_variance=self.kalman_process_variance,
            measurement_variance=self.kalman_measurement_variance
        )

        # Detector state
        self.state = 'IDLE'
        self.low_count = 0
        self.edge_wait_count = 0
        self.event_sample_count = 0
        self.low_reference = None

        # Debug values
        self.last_median = None
        self.last_filtered = None
        self.last_drop = 0.0
        self.last_rise = 0.0
        self.last_variation = 0.0
        self.last_prev_mean = None
        self.last_curr_mean = None

        # Public status
        self.detected = False
        self.last_event_text = "none"

    def median_filter(self, value: float) -> float:
        self.raw_window.append(value)
        values = sorted(self.raw_window)
        return values[len(values) // 2]

    @staticmethod
    def mean(values):
        if not values:
            return None
        return sum(values) / len(values)

    def get_window_stats(self):
        """
        Returns:
            prev_mean, curr_mean, variation, drop

        variation = curr_mean - prev_mean
            negative => recent values are closer
            positive => recent values are farther

        drop = prev_mean - curr_mean
            positive => recent values are closer
        """
        needed = self.prev_window_size + self.curr_window_size
        if len(self.filtered_history) < needed:
            return None, None, None, None

        hist = list(self.filtered_history)
        prev_vals = hist[-needed:-self.curr_window_size]
        curr_vals = hist[-self.curr_window_size:]

        prev_mean = self.mean(prev_vals)
        curr_mean = self.mean(curr_vals)

        if prev_mean is None or curr_mean is None:
            return None, None, None, None

        variation = curr_mean - prev_mean
        drop = prev_mean - curr_mean
        return prev_mean, curr_mean, variation, drop

    def get_window_rise_from_low(self):
        if self.low_reference is None:
            return None

        if len(self.filtered_history) < self.curr_window_size:
            return None

        curr_vals = list(self.filtered_history)[-self.curr_window_size:]
        curr_mean = self.mean(curr_vals)
        if curr_mean is None:
            return None

        return curr_mean - self.low_reference

    def reset_candidate(self):
        self.state = 'IDLE'
        self.low_count = 0
        self.edge_wait_count = 0
        self.event_sample_count = 0
        self.low_reference = None
        self.detected = False

    def update(self, distance_m: float):
        """
        Returns:
            event (str or None):
                'candidate_start'
                'box_detected'
                'candidate_reset'
        """
        if distance_m < self.min_valid_range or distance_m > self.max_valid_range:
            return None

        median_value = self.median_filter(distance_m)
        filtered = self.kalman.update(median_value)

        self.last_median = median_value
        self.last_filtered = filtered
        self.filtered_history.append(filtered)

        prev_mean, curr_mean, variation, drop = self.get_window_stats()
        rise = self.get_window_rise_from_low()

        self.last_prev_mean = prev_mean
        self.last_curr_mean = curr_mean
        self.last_variation = 0.0 if variation is None else variation
        self.last_drop = 0.0 if drop is None else drop
        self.last_rise = 0.0 if rise is None else rise

        event = None

        if self.state == 'IDLE':
            self.detected = False

            if drop is not None and drop >= self.drop_threshold:
                recent_vals = list(self.filtered_history)[-self.curr_window_size:]
                self.low_reference = self.mean(recent_vals)
                self.low_count = 1
                self.edge_wait_count = 0
                self.event_sample_count = 1
                self.state = 'LOW_CONFIRM'
                self.last_event_text = "candidate_start"
                event = 'candidate_start'

        elif self.state == 'LOW_CONFIRM':
            self.edge_wait_count += 1
            self.event_sample_count += 1

            if self.low_reference is None:
                self.reset_candidate()
                self.last_event_text = "candidate_reset"
                return 'candidate_reset'

            if abs(filtered - self.low_reference) <= self.plateau_tolerance:
                self.low_count += 1

                recent_vals = list(self.filtered_history)[-self.curr_window_size:]
                recent_mean = self.mean(recent_vals)
                if recent_mean is not None:
                    self.low_reference = 0.8 * self.low_reference + 0.2 * recent_mean
            else:
                if self.low_count < self.min_plateau_samples:
                    self.reset_candidate()
                    self.last_event_text = "candidate_reset"
                    return 'candidate_reset'

            if self.low_count >= self.min_plateau_samples:
                self.state = 'WAIT_RISE'

            if self.edge_wait_count > self.max_gap_samples:
                self.reset_candidate()
                self.last_event_text = "candidate_reset"
                event = 'candidate_reset'

        elif self.state == 'WAIT_RISE':
            self.edge_wait_count += 1
            self.event_sample_count += 1

            if rise is not None and rise >= self.rise_threshold:
                if self.event_sample_count >= self.min_event_samples:
                    self.detected = True
                    self.last_event_text = "box_detected"
                    event = 'box_detected'
                else:
                    self.last_event_text = "candidate_reset"
                    event = 'candidate_reset'

                self.reset_candidate()

            elif self.edge_wait_count > self.max_gap_samples:
                self.reset_candidate()
                self.last_event_text = "candidate_reset"
                event = 'candidate_reset'

        return event


class BoxDetectorTestNode(Node):
    def __init__(self):
        super().__init__('box_detector_test')

        # Topics
        self.declare_parameter('left_topic', '/robocop/ds_left')
        self.declare_parameter('right_topic', '/robocop/ds_right')

        # Signal processing
        self.declare_parameter('median_window', 5)
        self.declare_parameter('prev_window_size', 4)
        self.declare_parameter('curr_window_size', 4)

        # Kalman filter parameters
        self.declare_parameter('kalman_process_variance', 0.0005)
        self.declare_parameter('kalman_measurement_variance', 0.0025)

        # Detector thresholds
        self.declare_parameter('drop_threshold_m', 0.05)
        self.declare_parameter('rise_threshold_m', 0.05)
        self.declare_parameter('plateau_tolerance_m', 0.015)
        self.declare_parameter('min_plateau_samples', 5)
        self.declare_parameter('max_gap_samples', 10)
        self.declare_parameter('min_event_samples', 5)

        # Range validity
        self.declare_parameter('min_valid_range_m', 0.03)
        self.declare_parameter('max_valid_range_m', 1.50)

        # Side detectors
        self.left_detector = EdgeBoxDetector('left', self)
        self.right_detector = EdgeBoxDetector('right', self)

        # Subscribers
        left_topic = self.get_parameter('left_topic').value
        right_topic = self.get_parameter('right_topic').value

        self.create_subscription(Range, left_topic, self.left_cb, 10)
        self.create_subscription(Range, right_topic, self.right_cb, 10)

        # Publishers
        self.left_box_pub = self.create_publisher(Bool, '/robocop/box_left_detected', 10)
        self.right_box_pub = self.create_publisher(Bool, '/robocop/box_right_detected', 10)

        self.left_drop_pub = self.create_publisher(Float32, '/robocop/box_left_drop', 10)
        self.right_drop_pub = self.create_publisher(Float32, '/robocop/box_right_drop', 10)

        self.left_rise_pub = self.create_publisher(Float32, '/robocop/box_left_rise', 10)
        self.right_rise_pub = self.create_publisher(Float32, '/robocop/box_right_rise', 10)

        self.left_variation_pub = self.create_publisher(Float32, '/robocop/box_left_variation', 10)
        self.right_variation_pub = self.create_publisher(Float32, '/robocop/box_right_variation', 10)

        self.left_filtered_pub = self.create_publisher(Float32, '/robocop/box_left_filtered', 10)
        self.right_filtered_pub = self.create_publisher(Float32, '/robocop/box_right_filtered', 10)

        self.left_median_pub = self.create_publisher(Float32, '/robocop/box_left_median', 10)
        self.right_median_pub = self.create_publisher(Float32, '/robocop/box_right_median', 10)

        self.left_state_pub = self.create_publisher(String, '/robocop/box_left_state', 10)
        self.right_state_pub = self.create_publisher(String, '/robocop/box_right_state', 10)

        self.get_logger().info('Kalman-based box detector test node started.')

    def publish_debug(
        self,
        detector: EdgeBoxDetector,
        detected_pub,
        drop_pub,
        rise_pub,
        variation_pub,
        filtered_pub,
        median_pub,
        state_pub
    ):
        detected_msg = Bool()
        detected_msg.data = detector.detected
        detected_pub.publish(detected_msg)

        drop_msg = Float32()
        drop_msg.data = float(detector.last_drop)
        drop_pub.publish(drop_msg)

        rise_msg = Float32()
        rise_msg.data = float(detector.last_rise)
        rise_pub.publish(rise_msg)

        variation_msg = Float32()
        variation_msg.data = float(detector.last_variation)
        variation_pub.publish(variation_msg)

        filtered_msg = Float32()
        filtered_msg.data = 0.0 if detector.last_filtered is None else float(detector.last_filtered)
        filtered_pub.publish(filtered_msg)

        median_msg = Float32()
        median_msg.data = 0.0 if detector.last_median is None else float(detector.last_median)
        median_pub.publish(median_msg)

        state_msg = String()
        state_msg.data = detector.state
        state_pub.publish(state_msg)

    def left_cb(self, msg: Range):
        event = self.left_detector.update(msg.range)
        self.publish_debug(
            self.left_detector,
            self.left_box_pub,
            self.left_drop_pub,
            self.left_rise_pub,
            self.left_variation_pub,
            self.left_filtered_pub,
            self.left_median_pub,
            self.left_state_pub
        )

        if event == 'candidate_start':
            self.get_logger().info(
                f'[LEFT] candidate start | median={self.left_detector.last_median:.3f} m '
                f'| filtered={self.left_detector.last_filtered:.3f} m '
                f'| variation={self.left_detector.last_variation:.3f} m '
                f'| drop={self.left_detector.last_drop:.3f} m'
            )
        elif event == 'box_detected':
            self.get_logger().info(
                f'[LEFT] BOX DETECTED | rise={self.left_detector.last_rise:.3f} m'
            )

    def right_cb(self, msg: Range):
        event = self.right_detector.update(msg.range)
        self.publish_debug(
            self.right_detector,
            self.right_box_pub,
            self.right_drop_pub,
            self.right_rise_pub,
            self.right_variation_pub,
            self.right_filtered_pub,
            self.right_median_pub,
            self.right_state_pub
        )

        if event == 'candidate_start':
            self.get_logger().info(
                f'[RIGHT] candidate start | median={self.right_detector.last_median:.3f} m '
                f'| filtered={self.right_detector.last_filtered:.3f} m '
                f'| variation={self.right_detector.last_variation:.3f} m '
                f'| drop={self.right_detector.last_drop:.3f} m'
            )
        elif event == 'box_detected':
            self.get_logger().info(
                f'[RIGHT] BOX DETECTED | rise={self.right_detector.last_rise:.3f} m'
            )


def main(args=None):
    rclpy.init(args=args)
    node = BoxDetectorTestNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()