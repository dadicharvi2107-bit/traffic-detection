"""
processor.py
────────────
Core video processing pipeline:
  YOLO v8 detection → Deep SORT tracking → traffic‑light colour detection
  → queue estimation → violation logic → annotated video + CSV output.

OPTIMIZED: GPU auto-detect, progress logging, smaller processing size,
faster tracker, frame-count safety.
"""

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import pandas as pd
import os
import time
import torch

# ── Device Auto-Detection ────────────────────────────────────────────────────
DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_GPU = torch.cuda.is_available()
print(f"[processor] Using device: {'GPU (CUDA)' if USE_GPU else 'CPU'}", flush=True)

# ── Model & Tracker ──────────────────────────────────────────────────────────
model = YOLO("yolov8n.pt")

# Only track vehicle classes (COCO IDs: car=2, motorcycle=3, bus=5, truck=7)
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


# ── Traffic‑Light Colour Detection (HSV) ────────────────────────────────────
def detect_traffic_light_colour(frame, roi=None):
    """Detect red / green / yellow signal via HSV thresholding."""
    if roi is not None:
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
    else:
        h, w = frame.shape[:2]
        crop = frame[0 : h // 3, w // 2 :]  # top‑right third

    if crop.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Red (wraps around hue 0)
    red_mask = (
        cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        | cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    )
    green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))

    counts = {
        "red": cv2.countNonZero(red_mask),
        "green": cv2.countNonZero(green_mask),
        "yellow": cv2.countNonZero(yellow_mask),
    }
    best = max(counts, key=counts.get)

    MIN_PIXELS = 50
    if counts[best] < MIN_PIXELS:
        return "unknown"
    return best


# ── Queue Detection (stationary vehicles) ───────────────────────────────────
class QueueDetector:
    MOVE_THRESH = 8
    STILL_FRAMES = 3

    def __init__(self):
        self.history = {}

    def update(self, track_id, x_center, y_center):
        if track_id not in self.history:
            self.history[track_id] = {"prev_pos": (x_center, y_center), "still_count": 0}
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
    ALPHA = 0.4
    RASH_THRESH = 25

    def __init__(self):
        self.prev = {}
        self.ema = {}

    def update(self, track_id, x_center, y_center):
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
                  traffic_light_roi=None, frame_skip=2, progress_callback=None):
    """
    Run the full detection → tracking → analytics pipeline.

    progress_callback : optional fn(percent_float) for a Streamlit progress bar.
    """

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # ── SPEED FIX: smaller processing size (was 640×480) ──
    PROC_W, PROC_H = 480, 320
    scale_x = PROC_W / orig_w
    scale_y = PROC_H / orig_h

    if traffic_light_roi is not None:
        rx1, ry1, rx2, ry2 = traffic_light_roi
        traffic_light_roi = (
            int(rx1 * scale_x), int(ry1 * scale_y),
            int(rx2 * scale_x), int(ry2 * scale_y),
        )

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1, fps // frame_skip),
        (PROC_W, PROC_H),
    )

    # ── SPEED FIX: lighter tracker, GPU embedder if available ──
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        embedder_gpu=USE_GPU,
        half=USE_GPU,          # half precision only helps on GPU
    )

    queue_detector = QueueDetector()
    speed_estimator = SpeedEstimator()
    track_class_map = {}

    data = []
    frame_id = 0
    processed_count = 0
    stop_line_y = int(PROC_H * 0.6)
    start_time = time.time()

    print(f"[processor] Video: {orig_w}x{orig_h} @ {fps}fps, "
          f"{total_frames} frames, skip={frame_skip}", flush=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── Frame skipping ──
        if frame_id % frame_skip != 0:
            frame_id += 1
            continue

        frame = cv2.resize(frame, (PROC_W, PROC_H))

        # ── 1. Traffic‑light colour ──
        signal_colour = detect_traffic_light_colour(frame, traffic_light_roi)
        signal_red = signal_colour == "red"

        # ── 2. YOLO detection (GPU/CPU auto) ──
        results = model(frame, verbose=False, device=DEVICE)
        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                if cls_id not in VEHICLE_CLASSES:
                    continue
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, cls_id))

        # ── 3. Deep SORT tracking ──
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            if tid not in track_class_map and track.det_class is not None:
                cid = int(track.det_class)
                track_class_map[tid] = VEHICLE_CLASSES.get(cid, "vehicle")

        # ── 4. Draw stop line + signal label ──
        line_colour = (0, 0, 255) if signal_red else (0, 255, 0)
        cv2.line(frame, (0, stop_line_y), (PROC_W, stop_line_y), line_colour, 2)
        cv2.putText(frame, f"Signal: {signal_colour.upper()}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_colour, 2)

        # ── 5. Per‑track analytics ──
        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            l, t, r, b = track.to_ltrb()
            x_center = (l + r) / 2
            y_center = (t + b) / 2
            vtype = track_class_map.get(tid, "vehicle")

            in_queue = queue_detector.update(tid, x_center, y_center)
            red_light_violation = signal_red and y_center < stop_line_y
            speed, rash_driving = speed_estimator.update(tid, x_center, y_center)
            conf = float(track.det_conf) if track.det_conf is not None else 0.0

            box_colour = (0, 255, 0)
            if red_light_violation:
                box_colour = (0, 0, 255)
            elif rash_driving:
                box_colour = (0, 165, 255)
            elif in_queue:
                box_colour = (255, 255, 0)

            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), box_colour, 2)
            cv2.putText(frame, f"ID:{tid} {vtype} {conf:.0%}",
                        (int(l), int(t) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_colour, 1)

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
        processed_count += 1

        # ── PROGRESS LOGGING (so terminal never looks "frozen") ──
        if processed_count % 10 == 0:
            elapsed = time.time() - start_time
            fps_proc = processed_count / elapsed if elapsed > 0 else 0
            pct = (frame_id / total_frames * 100) if total_frames else 0
            print(f"[processor] frame {frame_id}/{total_frames} "
                  f"({pct:.1f}%) | {fps_proc:.1f} proc-fps", flush=True)
            if progress_callback and total_frames:
                progress_callback(min(frame_id / total_frames, 1.0))

        frame_id += 1

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    print(f"[processor] DONE in {elapsed:.1f}s | {processed_count} frames processed",
          flush=True)

    df = pd.DataFrame(data)
    if df.empty:
        df = pd.DataFrame(columns=[
            "frame", "vehicle_id", "vehicle_type", "confidence",
            "x_center", "y_center", "speed_px",
            "in_queue", "red_light_violation", "rash_driving", "signal_state",
        ])
    df.to_csv(output_csv_path, index=False)
    return df