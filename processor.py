"""
processor.py
────────────
Core video processing pipeline:
  YOLO v8 detection → Deep SORT tracking → traffic‑light colour detection
  → queue estimation → violation logic → annotated video + CSV output.
"""

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import pandas as pd
import os

# ── Model & Tracker ──────────────────────────────────────────────────────────
model = YOLO("yolov8n.pt")

# Only track vehicle classes (COCO IDs: car=2, motorcycle=3, bus=5, truck=7)
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


# ── Traffic‑Light Colour Detection (HSV) ────────────────────────────────────
def detect_traffic_light_colour(frame, roi=None):
    """
    Detect whether the traffic light in *frame* is red, green, or yellow
    using HSV colour thresholding.

    Parameters
    ----------
    frame : np.ndarray   – BGR frame from OpenCV.
    roi   : tuple | None – (x1, y1, x2, y2) region of interest that contains
                           the traffic light.  If None the top‑right quarter
                           of the frame is used as a reasonable default for
                           many fixed‑camera setups.

    Returns
    -------
    str – "red", "green", "yellow", or "unknown"
    """
    if roi is not None:
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
    else:
        h, w = frame.shape[:2]
        crop = frame[0 : h // 3, w // 2 :]  # top‑right third

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # ── Red (wraps around hue 0) ──
    red_lo1 = np.array([0, 100, 100])
    red_hi1 = np.array([10, 255, 255])
    red_lo2 = np.array([160, 100, 100])
    red_hi2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, red_lo1, red_hi1) | cv2.inRange(hsv, red_lo2, red_hi2)

    # ── Green ──
    green_lo = np.array([40, 50, 50])
    green_hi = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, green_lo, green_hi)

    # ── Yellow ──
    yellow_lo = np.array([15, 100, 100])
    yellow_hi = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lo, yellow_hi)

    red_px = cv2.countNonZero(red_mask)
    green_px = cv2.countNonZero(green_mask)
    yellow_px = cv2.countNonZero(yellow_mask)

    MIN_PIXELS = 50  # ignore noise
    counts = {"red": red_px, "green": green_px, "yellow": yellow_px}
    best = max(counts, key=counts.get)

    if counts[best] < MIN_PIXELS:
        return "unknown"
    return best


# ── Queue Detection (stationary vehicles) ───────────────────────────────────
class QueueDetector:
    """
    A vehicle is "in queue" when it has been nearly stationary
    (pixel displacement < MOVE_THRESH) for at least STILL_FRAMES
    consecutive processed frames.
    """

    MOVE_THRESH = 8        # pixels
    STILL_FRAMES = 3       # consecutive frames to count as queued

    def __init__(self):
        self.history = {}  # track_id → {"prev_pos": (x,y), "still_count": int}

    def update(self, track_id, x_center, y_center):
        if track_id not in self.history:
            self.history[track_id] = {
                "prev_pos": (x_center, y_center),
                "still_count": 0,
            }
            return False

        rec = self.history[track_id]
        px, py = rec["prev_pos"]
        dist = np.hypot(x_center - px, y_center - py)

        if dist < self.MOVE_THRESH:
            rec["still_count"] += 1
        else:
            rec["still_count"] = 0

        rec["prev_pos"] = (x_center, y_center)
        return rec["still_count"] >= self.STILL_FRAMES


# ── Speed Estimator (pixel‑based, with smoothing) ──────────────────────────
class SpeedEstimator:
    """
    Estimates per‑vehicle speed in pixels/frame using an exponential
    moving average so a single jittery frame doesn't trigger a false
    "rash driving" flag.
    """

    ALPHA = 0.4            # EMA smoothing factor
    RASH_THRESH = 25       # px/frame — tune for your video

    def __init__(self):
        self.prev = {}     # track_id → (x, y)
        self.ema = {}      # track_id → smoothed speed

    def update(self, track_id, x_center, y_center):
        """Returns (smoothed_speed, is_rash)."""
        if track_id not in self.prev:
            self.prev[track_id] = (x_center, y_center)
            self.ema[track_id] = 0.0
            return 0.0, False

        px, py = self.prev[track_id]
        raw_speed = np.hypot(x_center - px, y_center - py)
        smoothed = self.ALPHA * raw_speed + (1 - self.ALPHA) * self.ema[track_id]

        self.prev[track_id] = (x_center, y_center)
        self.ema[track_id] = smoothed

        return smoothed, smoothed > self.RASH_THRESH


# ── Main Processing Pipeline ────────────────────────────────────────────────
def process_video(input_path, output_video_path, output_csv_path,
                  traffic_light_roi=None, frame_skip=2):
    """
    Run the full detection → tracking → analytics pipeline.

    Parameters
    ----------
    input_path         : str  – path to input .mp4
    output_video_path  : str  – path to write annotated .mp4
    output_csv_path    : str  – path to write per‑frame CSV
    traffic_light_roi  : tuple | None – (x1,y1,x2,y2) for the signal crop
    frame_skip         : int  – process every Nth frame (2‑3 recommended)
    """

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    # Work at a fixed processing size to keep inference fast
    PROC_W, PROC_H = 640, 480
    scale_x = PROC_W / orig_w
    scale_y = PROC_H / orig_h

    # Scale the traffic‑light ROI if one was provided at original resolution
    if traffic_light_roi is not None:
        rx1, ry1, rx2, ry2 = traffic_light_roi
        traffic_light_roi = (
            int(rx1 * scale_x), int(ry1 * scale_y),
            int(rx2 * scale_x), int(ry2 * scale_y),
        )

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps // frame_skip,
        (PROC_W, PROC_H),
    )

    tracker = DeepSort(max_age=30, n_init=3)
    queue_detector = QueueDetector()
    speed_estimator = SpeedEstimator()

    # Map track_id → vehicle class name (persists across frames)
    track_class_map = {}

    data = []
    frame_id = 0
    stop_line_y = int(PROC_H * 0.6)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── Frame skipping (gentler: every 2nd frame, not every 5th) ──
        if frame_id % frame_skip != 0:
            frame_id += 1
            continue

        frame = cv2.resize(frame, (PROC_W, PROC_H))

        # ── 1. Traffic‑light colour ─────────────────────────────────
        signal_colour = detect_traffic_light_colour(frame, traffic_light_roi)
        signal_red = signal_colour == "red"

        # ── 2. YOLO detection (vehicles only) ───────────────────────
        results = model(frame, verbose=False)
        detections = []
        det_class_names = []  # parallel list so we can map class → track later

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                if cls_id not in VEHICLE_CLASSES:
                    continue

                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w, h = x2 - x1, y2 - y1

                detections.append(([x1, y1, w, h], conf, cls_id))
                det_class_names.append(VEHICLE_CLASSES[cls_id])

        # ── 3. Deep SORT tracking ───────────────────────────────────
        tracks = tracker.update_tracks(detections, frame=frame)

        # Update class map: for each confirmed track, find the nearest
        # detection and assign its class (one‑time per new track).
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            if tid not in track_class_map and track.det_class is not None:
                cid = int(track.det_class)
                track_class_map[tid] = VEHICLE_CLASSES.get(cid, "vehicle")

        # ── 4. Draw stop line ───────────────────────────────────────
        line_colour = (0, 0, 255) if signal_red else (0, 255, 0)
        cv2.line(frame, (0, stop_line_y), (PROC_W, stop_line_y), line_colour, 2)

        # Signal label
        label = f"Signal: {signal_colour.upper()}"
        cv2.putText(frame, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_colour, 2)

        # ── 5. Per‑track analytics ──────────────────────────────────
        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            l, t, r, b = track.to_ltrb()
            x_center = (l + r) / 2
            y_center = (t + b) / 2

            vtype = track_class_map.get(tid, "vehicle")

            # Queue
            in_queue = queue_detector.update(tid, x_center, y_center)

            # Red‑light violation
            red_light_violation = signal_red and y_center < stop_line_y

            # Speed / rash driving
            speed, rash_driving = speed_estimator.update(tid, x_center, y_center)

            # Confidence — use the track's detection confidence if available
            conf = float(track.det_conf) if track.det_conf is not None else 0.0

            # ── Draw bounding box ──
            box_colour = (0, 255, 0)
            if red_light_violation:
                box_colour = (0, 0, 255)
            elif rash_driving:
                box_colour = (0, 165, 255)
            elif in_queue:
                box_colour = (255, 255, 0)

            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), box_colour, 2)
            cv2.putText(
                frame,
                f"ID:{tid} {vtype} {conf:.0%}",
                (int(l), int(t) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_colour, 1,
            )

            data.append({
                "frame": frame_id,
                "vehicle_id": tid,
                "vehicle_type": vtype,
                "confidence": round(conf, 3),
                "x_center": round(x_center, 1),
                "y_center": round(y_center, 1),
                "speed_px": round(speed, 1),
                "in_queue": in_queue,
                "red_light_violation": red_light_violation,
                "rash_driving": rash_driving,
                "signal_state": signal_colour,
            })

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    df = pd.DataFrame(data)
    if df.empty:
        # Write an empty CSV with the right columns so the dashboard doesn't crash
        df = pd.DataFrame(columns=[
            "frame", "vehicle_id", "vehicle_type", "confidence",
            "x_center", "y_center", "speed_px",
            "in_queue", "red_light_violation", "rash_driving", "signal_state",
        ])
    df.to_csv(output_csv_path, index=False)
    return df
