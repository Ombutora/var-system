"""Player and ball detection with a pretrained YOLO model (COCO "person" and "sports ball").

Detection runs on a background thread so the review window never freezes. The
thread always serves the frame on screen first, and can optionally work through
the whole clip so playback and ball trails are instant afterwards.
"""

import threading
from dataclasses import dataclass

import cv2
import numpy as np

PERSON, SPORTS_BALL = 0, 32  # COCO class ids


@dataclass(frozen=True)
class Detection:
    kind: str  # "player" | "ball"
    box: tuple  # (x1, y1, x2, y2) in frame pixels
    conf: float

    @property
    def foot(self):
        """Bottom centre of the box: where the player touches the ground."""
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, y2)

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def contains(self, p):
        x1, y1, x2, y2 = self.box
        return x1 <= p[0] <= x2 and y1 <= p[1] <= y2


def parse_detections(xyxy, cls, conf, player_conf=0.35, ball_conf=0.15):
    """Turn raw model output into Detections, keeping only the most confident ball."""
    players, best_ball = [], None
    for box, c, p in zip(np.asarray(xyxy, float), np.asarray(cls, int), np.asarray(conf, float)):
        box = tuple(float(v) for v in box)
        if c == PERSON and p >= player_conf:
            players.append(Detection("player", box, float(p)))
        elif c == SPORTS_BALL and p >= ball_conf and (best_ball is None or p > best_ball.conf):
            best_ball = Detection("ball", box, float(p))
    return players + ([best_ball] if best_ball else [])


def snap_to_player(point, detections):
    """Foot point of the smallest player box containing `point`, or None."""
    hits = [d for d in detections if d.kind == "player" and d.contains(point)]
    if not hits:
        return None
    x = min(hits, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
    return x.foot


def ball_trail(results, idx, length=25):
    """Ball centres over the last `length` frames that have been analysed."""
    trail = []
    for i in range(max(0, idx - length), idx + 1):
        dets = results.get(i)
        if dets:
            balls = [d for d in dets if d.kind == "ball"]
            if balls:
                trail.append(balls[0].center)
    return trail


class DetectionWorker(threading.Thread):
    """Owns the YOLO model; the UI thread only reads results and posts requests."""

    def __init__(self, video_path, frame_count, weights, imgsz=1280, device=None,
                 player_conf=0.35, ball_conf=0.15):
        super().__init__(daemon=True)
        self.video_path = str(video_path)
        self.frame_count = frame_count
        self.weights, self.imgsz, self.device = str(weights), imgsz, device
        self.player_conf, self.ball_conf = player_conf, ball_conf

        self.lock = threading.Lock()
        self.wake = threading.Event()
        self._results = {}
        self._request = None  # (idx, frame): only the latest request is kept
        self.scanning = False
        self.scan_next = 0
        self.status = "loading model..."
        self.error = None
        self._stop = False

    # ---- called from the UI thread -----------------------------------------

    def get(self, idx):
        with self.lock:
            return self._results.get(idx)

    def snapshot(self):
        """Shallow copy of all results (for ball trails)."""
        with self.lock:
            return dict(self._results)

    def analysed(self):
        with self.lock:
            return len(self._results)

    def want(self, idx, frame):
        with self.lock:
            if idx in self._results:
                return
            self._request = (idx, frame)
        self.wake.set()

    def scan(self, from_idx):
        self.scan_next, self.scanning = from_idx, True
        self.wake.set()

    def stop(self):
        self._stop = True
        self.wake.set()

    # ---- worker thread -----------------------------------------------------

    def _detect(self, idx, frame):
        r = self.model.predict(
            frame, imgsz=self.imgsz, conf=min(self.player_conf, self.ball_conf),
            classes=[PERSON, SPORTS_BALL], device=self.device, verbose=False,
        )[0]
        b = r.boxes
        dets = parse_detections(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy(),
                                self.player_conf, self.ball_conf)
        with self.lock:
            self._results[idx] = dets

    def _next_scan_frame(self, cap, pos):
        """Next unanalysed frame from scan_next on, read with the worker's own capture."""
        with self.lock:
            while self.scan_next < self.frame_count and self.scan_next in self._results:
                self.scan_next += 1
        i = self.scan_next
        if i >= self.frame_count:
            return None, pos
        if i != pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            self.frame_count = i  # reported length was too long
            return None, pos
        return (i, frame), i + 1

    def run(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights)
            self._detect(-1, np.zeros((64, 64, 3), np.uint8))  # warm up
            with self.lock:
                self._results.pop(-1, None)
        except Exception as e:  # missing package, failed download, bad weights...
            self.error = f"{type(e).__name__}: {e}"
            self.status = "detection unavailable"
            return

        self.status = "ready"
        cap, pos = None, -1
        while not self._stop:
            with self.lock:
                req, self._request = self._request, None
            if req is not None:
                if self.get(req[0]) is None:
                    self._detect(*req)
                continue
            if self.scanning:
                if cap is None:
                    cap = cv2.VideoCapture(self.video_path)
                item, pos = self._next_scan_frame(cap, pos)
                if item is None:
                    self.scanning = False
                else:
                    self._detect(*item)
                    self.scan_next = item[0] + 1
                continue
            self.wake.wait(0.1)
            self.wake.clear()
        if cap is not None:
            cap.release()
