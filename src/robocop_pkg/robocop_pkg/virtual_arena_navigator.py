#!/usr/bin/env python3
"""Virtual Arena Navigator for SLRC 2026.

Navigates the simulated robot (Ares) to a target grid cell in the virtual arena
by communicating with the REST API at http://localhost:8000.

The virtual arena is a 25x25 grid of 0.4 m cells (total 10 m x 10 m).
Cell coordinates are integers in the range 0–24 for both X and Y axes.

Usage (one-shot via ROS 2 parameters)::

    ros2 run robocop_pkg virtual_arena_navigator \
        --ros-args -p cell_x:=5 -p cell_y:=10

Usage (continuous via ROS 2 topic)::

    # In one terminal, launch the node:
    ros2 run robocop_pkg virtual_arena_navigator

    # In another terminal, send a target cell (format: "cell_x,cell_y"):
    ros2 topic pub --once /virtual_arena/target_cell std_msgs/msg/String "data: '5,10'"

API endpoints used (all on http://localhost:8000):
  GET  /arena/metadata       – grid cell_size, grid_span, start_cell
  GET  /start_coordinate     – Ares world-frame start position {x, y}
  GET  /odometry             – current pose {pose: {x, y, yaw}}
  POST /move_relative        – execute {distance, rotation} with trapezoidal profile
  POST /stop                 – emergency stop
"""

import math
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

_API_DEFAULT = 'http://localhost:8000'
_MOVE_TIMEOUT = 60.0    # seconds – generous timeout for move_relative (blocks until done)
_HTTP_TIMEOUT = 5.0     # seconds – timeout for non-blocking API requests
_STOP_TIMEOUT = 2.0     # seconds – timeout for the emergency-stop request
_SETTLE_DELAY = 0.3     # seconds – brief pause after rotation so the robot fully settles


class VirtualArenaNavigator(Node):
    """ROS 2 node that navigates Ares to a target cell in the virtual arena.

    Cell coordinates are converted to world Cartesian coordinates using the
    arena metadata and the robot's known starting world position, then the
    robot is driven using rotate-then-forward motion via the /move_relative
    endpoint.
    """

    def __init__(self):
        super().__init__('virtual_arena_navigator')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('api_base', _API_DEFAULT)
        self.declare_parameter('cell_x', -1)
        self.declare_parameter('cell_y', -1)
        self.declare_parameter('arrival_tolerance', 0.15)   # metres
        self.declare_parameter('rotation_tolerance', 0.05)  # radians

        self._api_base = str(self.get_parameter('api_base').value)
        self._arrival_tol = float(self.get_parameter('arrival_tolerance').value)
        self._rotation_tol = float(self.get_parameter('rotation_tolerance').value)

        # ── Cached arena metadata ────────────────────────────────────────────
        self._cell_size: Optional[float] = None
        self._origin_x: Optional[float] = None
        self._origin_y: Optional[float] = None

        # ── Topic subscription ───────────────────────────────────────────────
        self._target_sub = self.create_subscription(
            String,
            '/virtual_arena/target_cell',
            self._target_cell_cb,
            10,
        )

        self.get_logger().info(
            f'Virtual Arena Navigator ready. API: {self._api_base}'
        )
        self.get_logger().info(
            'Listening on /virtual_arena/target_cell for "cell_x,cell_y" messages.'
        )

        # ── One-shot navigation from parameters ──────────────────────────────
        cell_x = int(self.get_parameter('cell_x').value)
        cell_y = int(self.get_parameter('cell_y').value)
        if cell_x >= 0 and cell_y >= 0:
            self.get_logger().info(
                f'One-shot navigation to cell ({cell_x}, {cell_y}) from parameters.'
            )
            # Use a short-delay timer so the node is fully initialised first.
            self._oneshot_args = (cell_x, cell_y)
            self._oneshot_timer = self.create_timer(0.5, self._oneshot_cb)

    # ── Timer callbacks ──────────────────────────────────────────────────────

    def _oneshot_cb(self):
        """Fire once then cancel: navigate to cell from ROS 2 parameters."""
        self._oneshot_timer.cancel()
        self.navigate_to_cell(*self._oneshot_args)

    # ── Subscription callback ────────────────────────────────────────────────

    def _target_cell_cb(self, msg: String):
        """Handle target cell messages formatted as ``"cell_x,cell_y"``."""
        raw = msg.data.strip()
        parts = raw.split(',')
        if len(parts) != 2:
            self.get_logger().error(
                f"Invalid target cell format: '{raw}'. Expected 'cell_x,cell_y'."
            )
            return
        try:
            cell_x = int(parts[0].strip())
            cell_y = int(parts[1].strip())
        except ValueError as exc:
            self.get_logger().error(
                f"Could not parse target cell '{raw}': {exc}"
            )
            return
        self.get_logger().info(f'Received target cell: ({cell_x}, {cell_y})')
        self.navigate_to_cell(cell_x, cell_y)

    # ── Arena metadata helpers ────────────────────────────────────────────────

    def _fetch_arena_metadata(self):
        """Return raw arena metadata dict from the API, or *None* on failure."""
        if not _REQUESTS_AVAILABLE:
            self.get_logger().error(
                "'requests' package not available. "
                "Install it with: pip install requests"
            )
            return None
        try:
            r = requests.get(
                f'{self._api_base}/arena/metadata', timeout=_HTTP_TIMEOUT
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            self.get_logger().error(f'Failed to get arena metadata: {exc}')
            return None

    def _fetch_start_coordinate(self):
        """Return the Ares start world coordinate ``{x, y}``, or *None*."""
        try:
            r = requests.get(
                f'{self._api_base}/start_coordinate', timeout=_HTTP_TIMEOUT
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            self.get_logger().error(f'Failed to get start coordinate: {exc}')
            return None

    def _ensure_arena_origin(self) -> bool:
        """Compute and cache the world-frame origin of cell (0, 0).

        Uses the arena metadata and the robot's start world coordinate to
        derive the mapping between grid indices and world Cartesian positions.

        Returns True if the origin is already cached or was successfully
        computed, False otherwise.
        """
        if self._origin_x is not None:
            return True

        meta = self._fetch_arena_metadata()
        if meta is None:
            return False

        start_coord = self._fetch_start_coordinate()
        if start_coord is None:
            return False

        try:
            cell_size: float = meta['cell_size']
            start_cell: list = meta['start_cell']   # [cell_x, cell_y]
        except KeyError as exc:
            self.get_logger().error(
                f'Arena metadata is missing expected key: {exc}. '
                f'Received: {meta}'
            )
            return False
        start_wx: float = start_coord['x']
        start_wy: float = start_coord['y']

        # The center of start_cell in world frame is (start_wx, start_wy).
        # Center of cell (cx, cy) = (origin_x + cx*cell_size + cell_size/2,
        #                            origin_y + cy*cell_size + cell_size/2)
        # Solving for origin:
        self._cell_size = cell_size
        self._origin_x = start_wx - start_cell[0] * cell_size - cell_size / 2.0
        self._origin_y = start_wy - start_cell[1] * cell_size - cell_size / 2.0

        self.get_logger().info(
            f'Arena origin cached: ({self._origin_x:.3f}, {self._origin_y:.3f}), '
            f'cell_size: {self._cell_size:.3f} m'
        )
        return True

    # ── Public conversion helper ─────────────────────────────────────────────

    def cell_to_world(self, cell_x: int, cell_y: int) -> Tuple[Optional[float], Optional[float]]:
        """Convert grid cell indices to world Cartesian coordinates (cell centre).

        Args:
            cell_x: Grid X index (0–24).
            cell_y: Grid Y index (0–24).

        Returns:
            Tuple ``(world_x, world_y)`` in metres, or ``(None, None)`` on
            failure.
        """
        if not self._ensure_arena_origin():
            return None, None
        world_x = self._origin_x + cell_x * self._cell_size + self._cell_size / 2.0
        world_y = self._origin_y + cell_y * self._cell_size + self._cell_size / 2.0
        return world_x, world_y

    # ── Odometry helper ──────────────────────────────────────────────────────

    def _get_odometry(self):
        """Return ``(x, y, yaw)`` from the odometry endpoint, or *None*."""
        try:
            r = requests.get(
                f'{self._api_base}/odometry', timeout=_HTTP_TIMEOUT
            )
            r.raise_for_status()
            data = r.json()
            x = float(data['pose']['x'])
            y = float(data['pose']['y'])
            yaw = float(data['pose']['yaw'])
            return x, y, yaw
        except Exception as exc:
            self.get_logger().error(f'Failed to get odometry: {exc}')
            return None

    # ── Motion helpers ───────────────────────────────────────────────────────

    def _move_relative(self, distance: float, rotation: float) -> bool:
        """Send a relative move command.

        Args:
            distance: Forward displacement in metres (positive = forward).
            rotation: Heading change in radians (positive = CCW).

        Returns:
            True if the API accepted the command, False otherwise.
        """
        try:
            r = requests.post(
                f'{self._api_base}/move_relative',
                json={'distance': distance, 'rotation': rotation},
                timeout=_MOVE_TIMEOUT,
            )
            r.raise_for_status()
            return True
        except Exception as exc:
            self.get_logger().error(f'move_relative failed: {exc}')
            return False

    def _stop(self):
        """Send an emergency stop to Ares."""
        try:
            requests.post(
                f'{self._api_base}/stop', json={}, timeout=_STOP_TIMEOUT
            )
        except Exception:
            pass

    # ── Main navigation logic ─────────────────────────────────────────────────

    def navigate_to_cell(self, cell_x: int, cell_y: int) -> bool:
        """Navigate Ares to the centre of the specified grid cell.

        The robot first rotates to face the target, then drives straight toward
        it.  Both steps use the ``/move_relative`` endpoint which executes a
        smooth trapezoidal velocity profile and blocks until complete.

        Args:
            cell_x: Target cell X index (valid range 0–24).
            cell_y: Target cell Y index (valid range 0–24).

        Returns:
            True if navigation completed successfully, False on any error.
        """
        if not _REQUESTS_AVAILABLE:
            self.get_logger().error(
                "'requests' package is not installed. "
                "Run: pip install requests"
            )
            return False

        self.get_logger().info(f'Navigating to cell ({cell_x}, {cell_y})…')

        # ── Resolve target world position ────────────────────────────────────
        target_wx, target_wy = self.cell_to_world(cell_x, cell_y)
        if target_wx is None:
            self.get_logger().error(
                'Could not resolve arena metadata. Navigation aborted.'
            )
            return False

        self.get_logger().info(
            f'Target world position: ({target_wx:.3f}, {target_wy:.3f}) m'
        )

        # ── Get current pose ─────────────────────────────────────────────────
        odom = self._get_odometry()
        if odom is None:
            self.get_logger().error('Cannot read odometry. Navigation aborted.')
            return False

        curr_x, curr_y, curr_yaw = odom
        self.get_logger().info(
            f'Current pose: ({curr_x:.3f}, {curr_y:.3f}) m, '
            f'yaw={math.degrees(curr_yaw):.1f}°'
        )

        # ── Compute required motion ──────────────────────────────────────────
        dx = target_wx - curr_x
        dy = target_wy - curr_y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < self._arrival_tol:
            self.get_logger().info(
                f'Already within tolerance of cell ({cell_x}, {cell_y}). '
                f'Distance: {distance:.3f} m.'
            )
            return True

        target_angle = math.atan2(dy, dx)
        rotation = target_angle - curr_yaw
        # Normalise to (-π, π]
        rotation = (rotation + math.pi) % (2.0 * math.pi) - math.pi

        self.get_logger().info(
            f'Required: rotate {math.degrees(rotation):.1f}°, '
            f'drive {distance:.3f} m'
        )

        # ── Step 1: Rotate to face the target ────────────────────────────────
        if abs(rotation) > self._rotation_tol:
            self.get_logger().info(f'Rotating {math.degrees(rotation):.1f}°…')
            if not self._move_relative(0.0, rotation):
                return False
            time.sleep(_SETTLE_DELAY)  # allow the robot to fully settle after rotation

        # ── Step 2: Drive forward to target ─────────────────────────────────
        self.get_logger().info(f'Driving forward {distance:.3f} m…')
        if not self._move_relative(distance, 0.0):
            return False

        self.get_logger().info(
            f'Arrived at cell ({cell_x}, {cell_y}). '
            f'Target world pos: ({target_wx:.3f}, {target_wy:.3f}) m.'
        )
        return True


def main(args=None):
    rclpy.init(args=args)
    node = VirtualArenaNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
