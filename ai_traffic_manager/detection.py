# detection.py
import os
import cv2
import numpy as np
from ultralytics import YOLO

class LaneDetector:
    """
    YOLOv8-based detector with safe lazy-loading and fallback.
    predict_frame(frame) -> (counts, ambulance_lane, annotated_frame)
    """

    def __init__(self, model_path="yolov8n.pt", allow_download=True):
        self.model_path = model_path
        self.model = None
        self.allow_download = allow_download
        self.vehicle_labels = ["car", "bus", "truck", "motorbike", "motorcycle", "bicycle"]
        self._mock = False  # switched true if model unavailable

    def _ensure_model(self):
        if self.model is not None or self._mock:
            return

        if not os.path.exists(self.model_path):
            if self.allow_download:
                print("[INFO] Model not found locally — attempting download...")
                try:
                    import urllib.request
                    urllib.request.urlretrieve(
                        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
                        self.model_path
                    )
                    print("[INFO] Model downloaded.")
                except Exception as e:
                    print("[WARN] Auto-download failed:", e)
            else:
                print("[WARN] Model not present and downloads disallowed.")

        try:
            self.model = YOLO(self.model_path)
            # do not force torch internals here; rely on ultralytics device args during predict
            print("[INFO] YOLO model initialized.")
        except Exception as e:
            print("[ERROR] Bad YOLO init:", e)
            self.model = None
            self._mock = True

    def predict_frame(self, frame):
        """Return (counts dict, ambulance_lane or None, annotated_frame)."""
        if frame is None:
            raise ValueError("Empty frame passed to detector.")

        h, w = frame.shape[:2]
        counts = {"N": 0, "E": 0, "S": 0, "W": 0}
        ambulance_lane = None
        annotated = frame.copy()

        self._ensure_model()

        if self.model is None:
            # fallback deterministic mock: use brightness to create repeatable counts
            avg = int(frame.mean() % 5)
            counts = {l: (avg + i) % 5 for i, l in enumerate(["N", "E", "S", "W"])}
            return counts, None, annotated

        try:
            # predict on CPU explicitly - avoid GPU initialization
            results = self.model.predict(frame, imgsz=640, device="cpu", verbose=False)[0]
            names = self.model.names

            for box in results.boxes:
                cls = int(box.cls)
                label = names.get(cls, str(cls)).lower()
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # simple region mapping (center-based)
                if cy < h // 2 and abs(cx - w // 2) < w // 4:
                    lane = "N"
                elif cy > h // 2 and abs(cx - w // 2) < w // 4:
                    lane = "S"
                elif cx < w // 2:
                    lane = "W"
                else:
                    lane = "E"

                if "ambulance" in label:
                    ambulance_lane = lane

                if any(v in label for v in self.vehicle_labels):
                    counts[lane] += 1

            # annotated frame returned by ultralytics
            try:
                annotated = results.plot()
            except Exception:
                annotated = frame.copy()

        except Exception as e:
            print("[ERROR] YOLO predict failed:", e)
            # fallback - don't crash
            counts = {l: 0 for l in counts}

        return counts, ambulance_lane, annotated
