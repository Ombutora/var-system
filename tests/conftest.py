"""Shared helpers: a synthetic broadcast scene and the app with a fake detector instead of YOLO."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import var_review as vr  # noqa: E402
from detection import Detection  # noqa: E402

# Broadcast-style perspective for a 1920x1080 frame (u = metres from goal line, v = across)
H = np.diag([0.75, 0.75, 1.0]) @ np.array([[38.0, -6.0, 900.0], [4.0, 9.0, 260.0], [0.012, 0.004, 1.0]])
RED, BLUE, REF = (110.0, 180.0, 165.0), (70.0, 140.0, 85.0), (210.0, 120.0, 200.0)  # Lab shirts


def project(u, v):
    p = H @ np.array([u, v, 1.0])
    return (p[0] / p[2], p[1] / p[2])


def player(u, v, lab):
    x, y = project(u, v)
    return Detection("player", (x - 15, y - 70, x + 15, y), 0.9, lab=lab, bgr=(0, 0, 0))


def ball_at_feet(p):
    """A ball just in front of a player's feet."""
    x, y = p.foot
    return Detection("ball", (x + 4, y - 8, x + 12, y), 0.5)


class FakeDetector:
    status, error, scanning = "ready", None, False

    def __init__(self, results):
        self.results = results

    def get(self, idx):
        return self.results.get(idx)

    def snapshot(self):
        return dict(self.results)

    def analysed(self):
        return len(self.results)

    def want(self, idx, frame):
        pass


def make_app(tmp_path, dets):
    """App on a blank 1920x1080 clip, calibrated on the goal line and 18-yard line, in offside mode."""
    video = tmp_path / "clip.mp4"
    wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (1920, 1080))
    for _ in range(3):
        wr.write(np.zeros((1080, 1920, 3), np.uint8))
    wr.release()
    app = vr.VarApp(str(video), 1280, 720)
    app.detector = FakeDetector({0: dets})
    app.show_detect = app.show_teams = True
    app.update_teams()
    assert app.teams.fitted
    m = app.current_markup(create=True)
    m.add_calib((project(0, -25), project(0, 25)))
    m.add_calib((project(16.5, -25), project(16.5, 25)))
    app.handle_key(ord("5"))
    return app


def team_label(app, d):
    return app.team_of(d)[0]


def click(app, d, button="l", flags=0):
    x, y = d.center
    sx, sy = int(x * app.base_scale), int(y * app.base_scale)
    app.on_mouse(cv2.EVENT_MOUSEMOVE, sx, sy, 0, None)
    app.on_mouse(cv2.EVENT_LBUTTONDOWN if button == "l" else cv2.EVENT_RBUTTONDOWN, sx, sy, flags, None)


def near(p, q):
    return p is not None and np.hypot(p[0] - q[0], p[1] - q[1]) < 1e-6
