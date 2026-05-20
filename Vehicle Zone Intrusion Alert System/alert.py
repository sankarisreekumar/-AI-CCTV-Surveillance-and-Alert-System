import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import os
import sys

# -------------------------------
# RESOURCE PATH (IMPORTANT FOR EXE)
# -------------------------------
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# -------------------------------
# SETTINGS
# -------------------------------
PRE_SECONDS = 2
POST_SECONDS = 58
OUTPUT_DIR = "alerts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use resource path for counter file
counter_file = resource_path("counter.txt")

# Create counter file if not exists
if not os.path.exists(counter_file):
    with open(counter_file, "w") as f:
        f.write("0")


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def run_detection(video_path, model_path):

    # Read counter
    with open(counter_file, "r") as f:
        alert_counter = int(f.read().strip())

    # Load model (IMPORTANT FIX)
    model = YOLO(resource_path(model_path))
    names = model.names

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Video not opened")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25

    buffer_size = int(PRE_SECONDS * fps)
    frame_buffer = deque(maxlen=buffer_size)

    recording = False
    record_frames_left = 0
    video_writer = None

    left_line = ((39, 474), (192, 213))
    right_line = ((281, 402), (308, 194))

    def extend_line_limited(p1, p2, height, y_top=120, y_bottom=None):
        if y_bottom is None:
            y_bottom = height - 20

        x1, y1 = p1
        x2, y2 = p2

        if x2 - x1 == 0:
            return (x1, y_top), (x1, y_bottom)

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        x_top = int((y_top - b) / m)
        x_bottom = int((y_bottom - b) / m)

        return (x_top, y_top), (x_bottom, y_bottom)

    prev_alert = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]

        # ---------------- LANE ----------------
        left_ext = extend_line_limited(left_line[0], left_line[1], h)
        right_ext = extend_line_limited(right_line[0], right_line[1], h)

        (x1_l, y1_l), (x2_l, y2_l) = left_ext
        left_ext = ((256, 123), (x2_l, y2_l))

        pts = np.array([
            left_ext[0],
            left_ext[1],
            right_ext[1],
            right_ext[0]
        ])

        lane_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(lane_mask, [pts], 255)

        # ---------------- YOLO ----------------
        results = model.predict(frame, conf=0.4, verbose=False)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        wall_in_lane = False
        vehicle_in_lane = False

        # ---------------- DETECTIONS ----------------
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls)

            roi = lane_mask[y1:y2, x1:x2]

            label = names[cls_id]
            color = (200, 200, 200)

            # WALL
            if cls_id == 0:
                color = (255, 0, 0)
                label = "WALL"

                cx = int((x1 + x2) / 2)
                cy = int(y2)

                cx = np.clip(cx, 0, w - 1)
                cy = np.clip(cy, 0, h - 1)

                cv2.circle(frame, (cx, cy), 6, color, -1)

                if lane_mask[cy, cx] > 0:
                    wall_in_lane = True

            # VEHICLE
            elif cls_id == 1:
                color = (0, 255, 0)
                label = "VEHICLE"

                overlap_ratio = 0
                if roi.size > 0:
                    lane_pixels = np.sum(roi > 0)
                    bbox_area = (x2 - x1) * (y2 - y1)
                    if bbox_area > 0:
                        overlap_ratio = lane_pixels / bbox_area

                if overlap_ratio >= 0.4:
                    vehicle_in_lane = True

                cv2.putText(frame, f"{overlap_ratio:.2f}",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

            # PERSON
            elif cls_id == 2:
                color = (0, 255, 255)
                label = "PERSON"

            # DRAW
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        # ---------------- ALERT ----------------
        alert = wall_in_lane and vehicle_in_lane

        overlay = frame.copy()
        lane_color = (0, 0, 255) if alert else (0, 255, 0)

        cv2.fillPoly(overlay, [pts], lane_color)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        cv2.polylines(frame, [pts], True, lane_color, 3)

        if alert:
            cv2.putText(frame, "EARLY ALERT",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        3)

        # ---------------- BUFFER ----------------
        frame_buffer.append(frame.copy())

        # ---------------- RECORD ----------------
        if alert and not prev_alert and not recording:
            recording = True
            record_frames_left = int(POST_SECONDS * fps)

            alert_counter += 1
            filename = os.path.join(OUTPUT_DIR, f"alert_{alert_counter}.mp4")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))

            print("Recording:", filename)

            for bf in frame_buffer:
                video_writer.write(bf)

        if recording:
            video_writer.write(frame)
            record_frames_left -= 1

            if record_frames_left <= 0:
                recording = False
                video_writer.release()

        prev_alert = alert

        cv2.imshow("Forklift Alert System", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Save counter
    with open(counter_file, "w") as f:
        f.write(str(alert_counter))

    cap.release()
    cv2.destroyAllWindows()