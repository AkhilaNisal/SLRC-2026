# Task 2: Collecting Parts from Inventory & Coordinate Extraction — Comprehensive Plan

## 1. Task Summary

The robot must:
1. Navigate from the maze (Task 1 area) to the **Task 2 designated area**
2. Traverse along the platforms (placed on left/right wall)
3. **Detect** each red cube (5×5×5 cm) on a grey platform (10×10×5 cm)
4. **Stop** aligned with the platform
5. **Pick** the red cube using the 4-DOF arm + gripper
6. **Store** the cube in an onboard location (one of 6 `place` poses)
7. **Detect & decode** the AprilTag (6×6 cm, 1 cm above platform top) revealed behind the cube
8. **Illuminate green LED** on successful tag read
9. Repeat for all **6 platforms**

### Arena Constraints
- Platforms along left **or** right wall (unknown until match)
- Minimum 20 cm clearance from corners
- Platforms have unique color+shape markers (triangle/circle/pentagon × blue/green/orange) on 3 visible sides — helpful for identification but not required for scoring
- AprilTag is **hidden behind the cube** and only visible after the cube is picked

---

## 2. Available Hardware

| Component | Details | ROS2 Topic / Interface |
|---|---|---|
| **Front webcam** | USB, 640×480 @ 30 FPS | `/camera/image/image_color` |
| **ToF left** | VL53L0X, 30–2000 mm | `/robocop/ds_left` |
| **ToF front** | VL53L0X, 30–2000 mm | `/robocop/ds_front` |
| **ToF right** | VL53L0X, 30–2000 mm | `/robocop/ds_right` |
| **Robot arm** | 4-DOF + parallel gripper, MoveIt2 | `/pick_box` action (PickBox) |
| **Stepper motors** | Differential drive, A4988 | `/cmd_vel` |
| **IMU** | MPU6050 | `/imu` |
| **OLED** | SSD1306 128×64 | `/oled_text` |
| **Green LED** | GPIO (to be wired) | Custom GPIO publisher |

---

## 3. Existing Code We Will Reuse

| Module | Reuse For |
|---|---|
| `camera_feed_node` | Camera image stream — already works |
| `tof_node` | 3× ToF distance readings — already works |
| `cmd_vel_stepper_node` | Motor control via `/cmd_vel` — already works |
| `robot_arm_action_server` | Pick & place sequences — reuse `do_pick_place_sequence()` + poses |
| `line_detection_utils` / `build_white_mask()` | White line detection — already calibrated |
| `task2.py` (current) | FSM skeleton: approach, turn, follow line, detect boxes — **extend this** |
| `mpu6050_node` | IMU for heading during turns — available |
| `oled_display_node` | Display status messages — available |

---

## 4. New Capabilities Needed

### 4.1 AprilTag Detection
**Library:** `dt-apriltags` (pip install)
- Pure Python bindings for AprilTag3 library
- Supports `tag36h11` family (most common for SLRC)
- Returns: `tag_id`, `center`, `corners`, `decision_margin`
- Lightweight, runs on RPi at ~15-20 FPS for 640×480

**Installation:**
```bash
pip install dt-apriltags
```

**Integration plan:**
- Create a new ROS2 node: `apriltag_detector_node`
- Subscribes to `/camera/image/image_color`
- Publishes detected tags to `/apriltag/detections` (custom message or JSON string)
- On detection → publish tag_id, trigger green LED, run decode pipeline

### 4.2 AprilTag Decoding Pipeline
Per Task 1 specification:
```
Raw Tag ID → Decoded value (0–48713)
  → First digit: Key ID (selects 1 of 5 keys + decode function)
  → Remaining 4 digits: Payload

After decode function applied:
  → 2-digit Order ID (1–14)
  → 2-digit X coordinate (0–24)
  → 2-digit Y coordinate (0–24)
```

**Implementation:** A pure function `decode_tag(raw_id, keys)` → `(order_id, x, y)`
- Keys and decode functions obtained from pre-competition puzzles
- Pluggable: load keys from a YAML/JSON config at runtime

### 4.3 Red Cube Detection (Vision)
**Already exists** in `red_box_seeker` and `red_box_perpendicular_seeker`:
- HSV red: H[0–10] ∪ H[170–180], S≥120, V≥70
- Contour area filtering, centroid tracking

**Enhancement for Task 2:**
- Use the **same HSV red detection** to confirm cube presence on platform
- Combine with **ToF side sensor** distance for robust detection
- Optional: detect the grey platform color as secondary confirmation

### 4.4 Platform Alignment (Precise Stopping)
Critical for arm reach. The arm has a fixed `grab` pose that assumes the cube is at a known position relative to the robot.

**Approach strategy (two-stage):**
1. **Coarse detection** via side ToF: detect distance drop when passing a platform
2. **Fine alignment** via camera: center the red cube in the camera frame, then creep forward until cube is at the target pixel row (arm reach distance)

### 4.5 Green LED Indicator
- GPIO output pin on RPi
- Simple ROS2 node or service call to toggle on/off
- Light up on successful AprilTag read, turn off before moving to next

---

## 5. System Architecture

```
                    ┌────────────────────┐
                    │   task2_main_node   │  ← NEW master FSM
                    │  (orchestrator)     │
                    └────┬───┬───┬───┬───┘
                         │   │   │   │
          ┌──────────────┘   │   │   └──────────────┐
          ▼                  ▼   ▼                   ▼
  ┌───────────────┐ ┌──────────────┐ ┌──────────────────┐ ┌──────────┐
  │ camera_feed   │ │  tof_node    │ │ robot_arm_action  │ │ stepper  │
  │ _node         │ │ (3× VL53L0X)│ │ _server (MoveIt)  │ │ _node    │
  └───────┬───────┘ └──────┬──────┘ └──────────────────┘ └──────────┘
          │                │
          ▼                ▼
  ┌───────────────┐  Distance data
  │ apriltag_     │  for platform
  │ detector_node │  detection
  └───────────────┘
```

---

## 6. State Machine Design

The new `task2_collector` node replaces/extends the existing `task2.py` with a richer FSM:

```
STATES:
  1. ENTER_TASK2_AREA       — Navigate from maze exit to Task 2 zone
  2. DETERMINE_WALL_SIDE    — Use ToF left/right to detect which wall has platforms
  3. FOLLOW_LINE            — Follow white line along the corridor
  4. DETECT_PLATFORM        — Side ToF detects distance drop → platform nearby
  5. ALIGN_TO_CUBE          — Camera-based: center red cube in frame, approach to arm range
  6. STOP_AND_PICK          — Stop, call /pick_box action → arm grabs cube & stores
  7. DETECT_APRILTAG        — Camera reads AprilTag now visible behind removed cube
  8. DECODE_AND_STORE       — Decode tag → extract (order_id, x, y) → store in memory
  9. RESUME_LINE_FOLLOW     — Move forward, return to FOLLOW_LINE
  10. TASK_COMPLETE          — All 6 cubes collected → signal completion
```

### Detailed State Transitions

```
ENTER_TASK2_AREA
  │ (arrive at Task 2 zone — white line detected after turn)
  ▼
DETERMINE_WALL_SIDE
  │ (compare left_tof vs right_tof — closer side has platforms)
  ▼
FOLLOW_LINE ◄──────────────────────────────────────┐
  │ (side ToF detects close object < threshold)     │
  ▼                                                 │
DETECT_PLATFORM                                     │
  │ (red blob confirmed in camera)                  │
  ▼                                                 │
ALIGN_TO_CUBE                                       │
  │ (cube centered + at correct distance)           │
  ▼                                                 │
STOP_AND_PICK                                       │
  │ (arm action completes → cube stored)            │
  ▼                                                 │
DETECT_APRILTAG                                     │
  │ (tag detected + read within timeout)            │
  ▼                                                 │
DECODE_AND_STORE                                    │
  │ (green LED → log/store decoded coords)          │
  ▼                                                 │
RESUME_LINE_FOLLOW ─── (cube_count < 6) ────────────┘
  │
  │ (cube_count == 6)
  ▼
TASK_COMPLETE
```

---

## 7. Detailed Algorithm for Each State

### 7.1 ENTER_TASK2_AREA
- Reuse existing logic: follow white line, handle junctions
- Same as current `task2.py` states: APPROACH_LINE → TURN_LEFT_90 → POST_TURN_WAIT
- After turn is complete, transition to DETERMINE_WALL_SIDE

### 7.2 DETERMINE_WALL_SIDE
```python
# Sample left and right ToF over N frames
# The side with consistently shorter distance has the wall + platforms
if avg(left_tof) < avg(right_tof):
    platform_side = "LEFT"
    platform_tof_topic = "/robocop/ds_left"
else:
    platform_side = "RIGHT"
    platform_tof_topic = "/robocop/ds_right"
```
- Timeout: 2 seconds of sampling, then decide
- Fallback: use a pre-configured parameter if ToF is ambiguous

### 7.3 FOLLOW_LINE
- **Same proportional controller** as existing code:
  - White line HSV mask → centroid → P-control steering
  - `kp=0.004`, `max_angular=1.2 rad/s`, `linear_speed=0.12 m/s`
- While following, continuously check the **platform-side ToF**:
  - If `side_tof < PLATFORM_DETECT_THRESHOLD` (e.g., 0.35 m) for N consecutive frames → platform detected
  - Transition to DETECT_PLATFORM

### 7.4 DETECT_PLATFORM
- **Slow down** to creep speed (0.04 m/s)
- Run **red color detection** on camera:
  - HSV red mask: H[0–10] ∪ H[170–180], S≥100, V≥70
  - Find largest contour → compute bounding box and centroid
- If red blob area > MIN_AREA → confirm cube presence → transition to ALIGN_TO_CUBE
- If no red blob after timeout → false positive, resume FOLLOW_LINE

### 7.5 ALIGN_TO_CUBE
**Goal:** Position the robot so the cube is within the arm's grab reach.

**Two-axis alignment:**

1. **Lateral (angular) alignment:**
   - Compute horizontal pixel offset of red blob centroid from frame center
   - P-controller: `angular_z = -Kp_align * pixel_error`
   - Target: centroid within ±15 px of center

2. **Distance (linear) alignment:**
   - Use **front ToF** to measure distance to wall/platform
   - OR use red blob **vertical position** in frame (lower = closer)
   - Creep forward until blob's bottom edge reaches target row (calibrated to arm reach)
   - Alternative: front ToF reaches target distance (e.g., 0.18–0.22 m depending on arm calibration)

3. **Stop condition:**
   - Lateral error < threshold AND distance within range
   - Hold position, send zero velocity
   - Transition to STOP_AND_PICK

```python
# Alignment pseudocode
red_cx, red_cy, red_area = detect_red_blob(frame)
lateral_error = red_cx - frame_center_x
distance_error = TARGET_Y_ROW - red_cy  # or use front_tof

if abs(lateral_error) < 15 and abs(distance_error) < 10:
    # Aligned! Stop and pick
    cmd_vel = Twist()  # zero
    transition(STOP_AND_PICK)
else:
    cmd_vel.angular.z = -Kp_lateral * lateral_error
    cmd_vel.linear.x = Kp_distance * distance_error  # small creep
```

### 7.6 STOP_AND_PICK
- Publish zero velocity
- Determine which side the cube is on (from DETERMINE_WALL_SIDE):
  - `side = "LEFT"` or `"RIGHT"`
- Call `/pick_box` action with the determined side
- Wait for action result (MoveIt plans and executes):
  1. Open gripper
  2. Move to `grab` pose (configured for the side)
  3. Close gripper
  4. Move to `placeN` pose (N = cube count)
  5. Open gripper (release in storage)
  6. Return to `home` pose
- Increment `cube_count`
- Transition to DETECT_APRILTAG

### 7.7 DETECT_APRILTAG
Now that the cube is removed, an AprilTag (6×6 cm) is visible on the wall behind where the cube was.

```python
# AprilTag detection
from dt_apriltags import Detector

detector = Detector(
    families='tag36h11',
    nthreads=2,
    quad_decimate=1.0,     # Full resolution for 6cm tag at close range
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25,
)

# In detection loop:
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
tags = detector.detect(gray)

for tag in tags:
    tag_id = tag.tag_id
    center = tag.center
    corners = tag.corners
    # → proceed to decode
```

**Detection strategy:**
- Robot is already stopped at arm-reach distance from the platform
- Camera should see the AprilTag clearly (6 cm tag at ~20 cm distance)
- Apply **CLAHE** (histogram equalization) for consistent detection under varying light
- Retry for up to 3 seconds (90 frames) before declaring failure
- If tag not in frame, tilt camera slightly or drive forward slightly

**Optimization for close range:**
- `quad_decimate=1.0` — no decimation (tag is small in frame)
- `refine_edges=True` — better accuracy
- Consider ROI cropping (center portion of frame) to reduce processing time

### 7.8 DECODE_AND_STORE
```python
def decode_tag(raw_tag_id, keys_config):
    """
    Decode an AprilTag ID per SLRC competition rules.
    
    raw_tag_id → decoded_value (0–48713) via one of 5 key/function pairs
    decoded_value structure: KPPPP
      K = Key ID (1 digit)
      PPPP = Payload (4 digits)
    
    After applying decode function:
      Result: OOXXYYY
        OO = Order ID (1–14)
        XX = X coordinate (0–24)
        YY = Y coordinate (0–24)
    """
    decoded_value = apply_key_decode(raw_tag_id, keys_config)
    
    decoded_str = str(decoded_value).zfill(5)
    key_id = int(decoded_str[0])
    payload = decoded_str[1:5]
    
    # Apply corresponding decode function using key_id
    result = apply_decode_function(key_id, payload, keys_config)
    
    result_str = str(result).zfill(6)
    order_id = int(result_str[0:2])
    x_coord = int(result_str[2:4])
    y_coord = int(result_str[4:6])
    
    return order_id, x_coord, y_coord

# Store the decoded coordinate
coordinates.append({
    'order_id': order_id,
    'x': x_coord,
    'y': y_coord,
    'raw_tag_id': raw_tag_id,
    'platform_index': cube_count,
})
```

- **Green LED on** after successful decode
- Display decoded info on OLED
- Publish decoded data to `/task2/decoded_tags` topic
- Short pause (0.5s) for visual confirmation
- **Green LED off**
- Transition to RESUME_LINE_FOLLOW

### 7.9 RESUME_LINE_FOLLOW
- Move forward briefly (0.3s at 0.08 m/s) to clear the platform
- Transition back to FOLLOW_LINE

### 7.10 TASK_COMPLETE
- All 6 cubes collected and 6 AprilTags decoded
- Stop all motion
- Display collected coordinates on OLED
- Publish `/task2/status` = "COMPLETE"
- Sort decoded coordinates by `order_id` for Task 3 use

---

## 8. AprilTag Detection — Best Practices for This Setup

### 8.1 Library Choice: `dt-apriltags`
- **Why:** Pure Python bindings, ARM support, lightweight, well-tested on Raspberry Pi with Duckiebot
- **Alternative:** `pupil-apriltags` (also good, faster on some platforms)
- **Tag family:** Most likely `tag36h11` (confirm from competition rules)

### 8.2 Camera Calibration
For accurate tag pose estimation (optional but useful):
```python
# Camera intrinsic parameters [fx, fy, cx, cy]
# Calibrate using cv2.calibrateCamera() with a checkerboard
# Store in YAML config file
camera_params = [fx, fy, cx, cy]
tag_size = 0.06  # 6 cm in meters

tags = detector.detect(gray, estimate_tag_pose=True, 
                       camera_params=camera_params, tag_size=tag_size)
```

### 8.3 Lighting Compensation
```python
# CLAHE for consistent detection
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = clahe.apply(gray)
```

### 8.4 Detection Confidence
- Check `decision_margin` > 30 (reject low-confidence detections)
- Require **2 consecutive identical reads** before accepting
- Timeout: max 3 seconds per tag, then log failure and move on

---

## 9. Red Cube Detection — Best Practices

### 9.1 HSV Ranges (already calibrated in codebase)
```python
# Red wraps around HSV hue channel
lower_red_1 = np.array([0, 100, 70])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([170, 100, 70])
upper_red_2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
red_mask = mask1 | mask2
```

### 9.2 Morphological Cleaning
```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
```

### 9.3 Contour-Based Cube Localization
```python
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area > MIN_CUBE_AREA:
        x, y, w, h = cv2.boundingRect(largest)
        cx = x + w // 2
        cy = y + h // 2
        # Use (cx, cy) for alignment
```

---

## 10. Platform Alignment Strategy — Detailed

### 10.1 Phase 1: Side ToF Detection (Coarse)
The robot is following the white line. The side ToF sensor sweeps along the wall.

```
Normal wall distance: ~0.5–0.8 m (varies)
Platform present:     ~0.3–0.4 m (platform sticks out 10cm from wall)

Detection: side_tof < WALL_DISTANCE - 0.05  for N consecutive frames
```

- When transitioning from "no platform" to "platform detected," we know we're **entering** a platform zone
- Start slowing down immediately

### 10.2 Phase 2: Camera Alignment (Fine)
- Switch from pure line-following to **dual-objective control**:
  - Primary: center red cube horizontally in frame
  - Secondary: approach until cube is at target distance

```python
# Alignment Controller
Kp_angular = 0.003   # rad/s per pixel error
Kp_linear = 0.0005   # m/s per pixel error (distance)

target_cx = frame_width // 2   # horizontal center
target_cy = int(frame_height * 0.7)  # lower portion = closer

angular_z = -Kp_angular * (cube_cx - target_cx)
linear_x = Kp_linear * (target_cy - cube_cy)
linear_x = clamp(linear_x, 0.0, 0.06)  # only forward, max creep speed
```

### 10.3 Phase 3: Final Positioning
- front ToF as safety stop: if `front_tof < 0.15 m` → emergency stop
- When aligned: publish zero velocity for 0.5s to ensure full stop before arm operation

---

## 11. Arm Pick Sequence — Modifications Needed

The existing `robot_arm_action_server` already handles pick & place. Modifications:

### 11.1 Side-Specific Grab Poses
Since platforms can be on LEFT or RIGHT wall, we need **two grab poses**:
- `grab_left`: arm rotated toward left side
- `grab_right`: arm rotated toward right side

The existing `grab` pose uses `base_rotating_waste_joint = -0.1222` (≈ -7°, slightly left).
We need to add a mirrored pose for right-side picking.

```yaml
# New poses to calibrate with pose_tuner
grab_left:
  base_rotating_waste_joint: -0.35    # rotated left ≈ -20°
  rotating_waste_arm1_joint: 1.4439
  arm1_arm2_joint: 0.0873
  arm2_gripper_base_joint: -1.3963

grab_right:
  base_rotating_waste_joint: 0.35     # rotated right ≈ +20°
  rotating_waste_arm1_joint: 1.4439
  arm1_arm2_joint: 0.0873
  arm2_gripper_base_joint: -1.3963
```

### 11.2 Storage Slot Management
6 cubes → 6 storage positions already defined (`place1`–`place6`).
The action server's `placed_box_count` auto-increments, selecting the next slot.

### 11.3 Timing
- Full pick-place cycle: ~8–12 seconds (MoveIt planning + execution)
- Robot must remain stationary throughout

---

## 12. Implementation Plan — Step by Step

### Phase 1: Foundation (New Nodes)

#### Step 1: AprilTag Detector Node
Create `src/robocop_pkg/robocop_pkg/apriltag_detector.py`:
- Subscribes: `/camera/image/image_color`
- Publishes: `/apriltag/detections` (String msg with JSON: `[{tag_id, center_x, center_y, decision_margin}]`)
- Service: `/apriltag/detect_once` → trigger single detection and return result
- Parameters: `tag_family`, `quad_decimate`, `nthreads`, `min_decision_margin`

#### Step 2: Tag Decode Module
Create `src/robocop_pkg/robocop_pkg/tag_decoder.py`:
- Pure functions, no ROS dependencies
- `load_keys(yaml_path)` → load 5 decode keys
- `decode_raw_tag(raw_id, keys)` → decoded value
- `extract_coordinates(decoded_value)` → `(order_id, x, y)`
- Unit tests with known values

#### Step 3: Green LED Node
Create `src/robocop_pkg/robocop_pkg/led_controller.py`:
- Subscribes: `/led/green` (Bool)
- GPIO output to green LED pin
- Simple on/off control

### Phase 2: Vision Processing

#### Step 4: Red Cube Detector (Enhanced)
Create `src/robocop_pkg/robocop_pkg/red_cube_detector.py`:
- Subscribes: `/camera/image/image_color`
- Publishes: `/red_cube/detection` (custom msg or JSON String with: `found`, `cx`, `cy`, `area`, `bbox`)
- HSV red detection with morphological cleanup
- Runs at camera frame rate

#### Step 5: Camera Calibration
- Run `cv2.calibrateCamera()` with a printed checkerboard
- Store intrinsics in `config/camera_calibration.yaml`
- Used by AprilTag detector for pose estimation

### Phase 3: Main Controller

#### Step 6: Task 2 Collector Node
Create `src/robocop_pkg/robocop_pkg/task2_collector.py`:
- Master FSM (10 states as described in Section 6)
- Subscribes to:
  - `/camera/image/image_color` (or relay via detection nodes)
  - `/robocop/ds_left`, `/robocop/ds_front`, `/robocop/ds_right`
  - `/apriltag/detections`
  - `/red_cube/detection`
- Publishes:
  - `/cmd_vel`
  - `/task2/status`
  - `/task2/decoded_tags`
  - `/oled_text`
  - `/led/green`
- Action client: `/pick_box`
- Data storage: list of decoded `(order_id, x, y)` tuples

### Phase 4: Integration & Tuning

#### Step 7: Launch File
Create `src/robocop_pkg/launch/task2_full.launch.py`:
```python
# Nodes to launch:
# 1. camera_feed_node
# 2. tof_node
# 3. cmd_vel_stepper_node
# 4. robot_arm_action_server (with MoveIt)
# 5. apriltag_detector_node
# 6. red_cube_detector_node (or inline in main node)
# 7. led_controller_node
# 8. oled_display_node
# 9. task2_collector_node (main FSM)
```

#### Step 8: Parameter Tuning
Key parameters to calibrate on the real robot:

| Parameter | Description | How to Tune |
|---|---|---|
| `platform_detect_distance` | Side ToF threshold for platform | Measure actual distance to platform |
| `cube_align_target_cy` | Pixel row for correct arm reach | Pick cube, check success, adjust |
| `grab_left` / `grab_right` poses | Arm joint angles for grab | Use `pose_tuner` interactively |
| `approach_stop_distance` | Front ToF distance to stop | Measure arm reach vs. front dist |
| `red_hsv_*` | Red color ranges | HSV tuner with actual cubes |
| `apriltag_min_margin` | Min detection confidence | Test with printed tags |

#### Step 9: Testing Protocol
1. **Unit test:** AprilTag detection on printed tags (on desk)
2. **Unit test:** Red cube detection in various lighting
3. **Unit test:** Arm pick sequence on a stationary cube
4. **Integration test:** Single platform pick + tag read
5. **Full run:** 6 platforms, end-to-end

---

## 13. Risk Mitigation

| Risk | Mitigation |
|---|---|
| AprilTag not detected (lighting/angle) | CLAHE preprocessing, multiple retries, adjust robot position |
| Cube drop during pick | Calibrate gripper close force, test with actual cubes |
| Platform on unexpected side | Auto-detect with ToF comparison; configurable parameter fallback |
| Arm can't reach cube | Multiple grab poses for different distances; use `pose_tuner` to calibrate |
| False platform detection | Require both ToF AND camera confirmation (red blob) |
| Missed platform (drove past) | Slow cruise speed (0.08 m/s near expected platform zones) |
| Tag family mismatch | Support multiple families; configurable parameter |
| Decode keys not yet available | Modular decode functions; swap in keys at competition time |

---

## 14. Timing Budget

| Phase | Estimated Duration |
|---|---|
| Enter Task 2 area | ~10s (line following + turn) |
| Determine wall side | ~2s |
| **Per platform (×6):** | |
| ├── Approach & detect | ~5–10s |
| ├── Align to cube | ~3–5s |
| ├── Arm pick & store | ~10–15s |
| ├── AprilTag detect + decode | ~2–3s |
| └── Resume | ~2s |
| **Total per platform** | **~22–35s** |
| **6 platforms total** | **~130–210s (2–3.5 min)** |
| **Grand total Task 2** | **~2.5–4 min** |

---

## 15. Data Flow Summary

```
Camera Frame
    │
    ├──► Red Cube Detector ──► cube position (cx, cy, area)
    │                              │
    │                              ▼
    │                      Task2 Collector FSM
    │                       │          │
    │                       │          ├──► /cmd_vel (steering)
    │                       │          ├──► /pick_box action
    │                       │          ├──► /oled_text
    │                       │          └──► /led/green
    │                       │
    ├──► AprilTag Detector ─┘──► tag_id
    │                              │
    │                              ▼
    │                       Tag Decoder
    │                              │
    │                              ▼
    │                       (order_id, x, y)
    │
ToF Sensors ──► distances ──► Task2 Collector FSM
                                (platform detection)
```

---

## 16. File Structure (New/Modified Files)

```
src/robocop_pkg/robocop_pkg/
    ├── apriltag_detector.py      # NEW: AprilTag detection node
    ├── tag_decoder.py            # NEW: Tag ID decode functions
    ├── red_cube_detector.py      # NEW: Red cube detection node (or inline)
    ├── led_controller.py         # NEW: Green LED GPIO controller
    ├── task2_collector.py        # NEW: Master FSM for Task 2
    └── task2.py                  # EXISTING: keep as reference/fallback

src/robocop_pkg/config/
    ├── decode_keys.yaml          # NEW: 5 decode keys + functions
    └── camera_calibration.yaml   # NEW: Camera intrinsics

src/robocop_pkg/launch/
    └── task2_full.launch.py      # NEW: Full Task 2 launch

src/robocop_pkg/test/
    ├── test_tag_decoder.py       # NEW: Unit tests for decode
    └── test_apriltag_detect.py   # NEW: Integration test for detection
```

---

## 17. Dependencies to Install

```bash
# AprilTag detection
pip install dt-apriltags

# Already available (verify):
pip install opencv-python numpy

# For LED GPIO control (RPi)
pip install gpiod   # already used by stepper_control

# For OLED (already used)
# adafruit-circuitpython-ssd1306
```

---

## 18. Quick Start Checklist

- [ ] Install `dt-apriltags` on the Raspberry Pi
- [ ] Print test AprilTags (tag36h11 family, various IDs)
- [ ] Calibrate camera intrinsics with checkerboard
- [ ] Calibrate `grab_left` and `grab_right` arm poses using `pose_tuner`
- [ ] Tune red HSV range with actual competition cubes
- [ ] Measure and set `platform_detect_distance` for side ToF
- [ ] Measure and set `approach_stop_distance` for front ToF
- [ ] Wire green LED to GPIO pin, update pin config
- [ ] Create decode keys config (from pre-competition puzzles)
- [ ] Run integration test: 1 platform → pick → tag read → decode
- [ ] Full run: 6 platforms end-to-end
