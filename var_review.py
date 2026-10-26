"""VAR review tool: frame-accurate replay, markup and perspective offside lines.

Usage:  python var_review.py path/to/clip.mp4
Press h in the window for the controls.
"""

import argparse
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import detection
import geometry

WINDOW = "VAR Review"
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "models" / "yolo11n.pt"
MODES = ["view", "line", "freehand", "calibrate", "offside"]
SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0]
MAX_ZOOM = 10.0
STRIP_H = 52  # status strip under the video

# BGR colours
MARKUP = (255, 0, 255)
CALIB = (0, 230, 255)
ATTACKER = (40, 40, 235)
DEFENDER = (235, 140, 30)
PLAYER_BOX = (90, 220, 90)
BALL = (0, 165, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# cv2.waitKeyEx codes for the arrow keys (Windows, then Linux)
KEY_LEFT = {2424832, 65361}
KEY_RIGHT = {2555904, 65363}

HELP = [
    "space      play / pause",
    "a d  <- -> step 1 frame     A D  step 10 frames",
    "[ ]        slower / faster",
    "wheel + -  zoom at cursor   z  reset zoom",
    "1 view     drag to pan (right-drag pans in any mode but offside)",
    "2 line     drag a straight line",
    "3 freehand drag to draw",
    "4 calibrate  2 clicks on the goal line, then 2 on the 6/18-yard line",
    "5 offside  left-click attacker, right-click 2nd-last defender",
    "           (click the ground point under the body part that counts)",
    "v          player / ball detection on/off   p analyse whole clip",
    "           (with detection on, offside clicks inside a player snap",
    "            to their feet; hold Ctrl to place the point freely)",
    "u undo     x clear this frame    s save snapshot",
    "h help     q / esc quit",
]


@dataclass
class Markup:
    """Everything drawn on one frame. Calibration is per frame because broadcast cameras pan."""

    shapes: list = field(default_factory=list)  # ("line" | "free", [points])
    calib: list = field(default_factory=list)  # up to two segments
    attacker: tuple = None
    defender: tuple = None
    history: list = field(default_factory=list)  # for undo

    def is_empty(self):
        return not (self.shapes or self.calib or self.attacker or self.defender)

    def add_shape(self, shape):
        self.shapes.append(shape)
        self.history.append(("shape", None))

    def add_calib(self, seg):
        if len(self.calib) == 2:  # a third segment starts calibration over
            self.history.append(("calib_reset", list(self.calib)))
            self.calib = []
        self.calib.append(seg)
        self.history.append(("calib", None))

    def set_player(self, role, point):
        self.history.append((role, getattr(self, role)))
        setattr(self, role, point)

    def undo(self):
        if not self.history:
            return False
        kind, prev = self.history.pop()
        if kind == "shape":
            self.shapes.pop()
        elif kind == "calib":
            self.calib.pop()
            if self.history and self.history[-1][0] == "calib_reset":
                self.calib = self.history.pop()[1]
        elif kind in ("attacker", "defender"):
            setattr(self, kind, prev)
        return True

    def verdict(self):
        if len(self.calib) < 2 or self.attacker is None or self.defender is None:
            return None
        try:
            return geometry.offside_verdict(
                self.calib[0], self.calib[1], self.attacker, self.defender, tolerance=0.5
            )
        except ValueError:
            return None


class FrameSource:
    """Random access to video frames with a cache so stepping backwards is instant."""

    def __init__(self, path, cache_size=240):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open video: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 and fps == fps else 25.0
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.next_index = 0
        if self.count <= 0:  # some containers don't report a length
            self.count = self._count_frames()

    def _count_frames(self):
        n = 0
        while self.cap.grab():
            n += 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return max(n, 1)

    def get(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        if i != self.next_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.next_index = i + 1
        self.cache[i] = frame
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return frame

    def release(self):
        self.cap.release()


def put_text(img, text, org, scale=0.55, color=WHITE, thickness=1):
    """Text on a dark box so it stays readable over any footage."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = max(3, h // 3)
    x, y = org
    cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + base + pad // 2), BLACK, -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_markup(img, m, frame_w, frame_h, to_img, th, preview=None):
    """Draw a frame's markup onto img. to_img maps frame coords to img pixel coords."""
    for kind, pts in m.shapes:
        if kind == "line":
            cv2.line(img, to_img(pts[0]), to_img(pts[1]), MARKUP, th, cv2.LINE_AA)
        else:
            poly = np.array([to_img(p) for p in pts], np.int32)
            cv2.polylines(img, [poly], False, MARKUP, th, cv2.LINE_AA)

    for seg in m.calib:
        cv2.line(img, to_img(seg[0]), to_img(seg[1]), CALIB, max(1, th // 2), cv2.LINE_AA)
        for p in seg:
            cv2.circle(img, to_img(p), th + 2, CALIB, -1, cv2.LINE_AA)

    vp = None
    if len(m.calib) == 2:
        try:
            vp = geometry.vanishing_point(m.calib[0], m.calib[1])
        except ValueError:
            vp = None

    for point, color, label in ((m.attacker, ATTACKER, "ATT"), (m.defender, DEFENDER, "DEF")):
        if point is None:
            continue
        if vp is not None:
            ends = geometry.clip_line(geometry.offside_line(point, vp), frame_w, frame_h)
            if ends:
                cv2.line(img, to_img(ends[0]), to_img(ends[1]), color, th, cv2.LINE_AA)
        c = to_img(point)
        cv2.circle(img, c, th * 3, color, th, cv2.LINE_AA)
        put_text(img, label, (c[0] + th * 4, c[1] - th * 4), 0.5 + th * 0.1, color, max(1, th // 2))

    if preview:
        cv2.line(img, to_img(preview[0]), to_img(preview[1]), preview[2], th, cv2.LINE_AA)


def draw_detections(img, dets, trail, to_img, th):
    for d in dets:
        if d.kind == "player":
            x1, y1, x2, y2 = d.box
            cv2.rectangle(img, to_img((x1, y1)), to_img((x2, y2)), PLAYER_BOX, max(1, th // 2), cv2.LINE_AA)
            cv2.circle(img, to_img(d.foot), th + 1, PLAYER_BOX, -1, cv2.LINE_AA)
    if len(trail) > 1:
        cv2.polylines(img, [np.array([to_img(p) for p in trail], np.int32)], False, BALL,
                      max(1, th // 2), cv2.LINE_AA)
    for d in dets:
        if d.kind == "ball":
            x1, y1, x2, y2 = d.box
            c, r = to_img(d.center), to_img((x2, y2))
            radius = max(th * 3, int(np.hypot(r[0] - c[0], r[1] - c[1])) + th)
            cv2.circle(img, c, radius, BALL, th, cv2.LINE_AA)


def verdict_style(verdict):
    name, margin = verdict
    color = {"OFFSIDE": ATTACKER, "ONSIDE": (60, 200, 60), "LEVEL": CALIB}[name]
    return f"{name}  ({margin:+.1f} px)", color


class VarApp:
    def __init__(self, path, max_w, max_h, detect_opts=None):
        self.path = Path(path)
        self.detect_opts = detect_opts or {}
        self.detector = None
        self.show_detect = False
        self.src = FrameSource(path)
        fw, fh = self.src.width, self.src.height
        self.base_scale = min(max_w / fw, max_h / fh)
        self.disp_w = int(round(fw * self.base_scale))
        self.disp_h = int(round(fh * self.base_scale))

        self.idx = 0
        self.frame = self.src.get(0)
        if self.frame is None:
            raise SystemExit(f"Could not read any frames from: {path}")
        self.playing = False
        self.speed_i = SPEEDS.index(1.0)
        self.mode = "view"
        self.markup = {}  # frame index -> Markup
        self.show_help = False
        self.message, self.message_until = "", 0.0

        self.zoom = 1.0
        self.center = np.array([fw / 2, fh / 2])
        self.mouse = (0.0, 0.0)  # frame coords
        self.drag = None  # current drag: dict with kind and data
        self.pending_calib = None
        self.seek = None
        self.window_open = False

    # ---- coordinate mapping -------------------------------------------------

    @property
    def scale(self):
        return self.base_scale * self.zoom

    def view_origin(self):
        fw, fh = self.src.width, self.src.height
        vw, vh = fw / self.zoom, fh / self.zoom
        x0 = min(max(self.center[0] - vw / 2, 0), fw - vw)
        y0 = min(max(self.center[1] - vh / 2, 0), fh - vh)
        self.center = np.array([x0 + vw / 2, y0 + vh / 2])
        return x0, y0

    def to_frame(self, dx, dy):
        x0, y0 = self.view_origin()
        x = min(max(x0 + dx / self.scale, 0), self.src.width - 1)
        y = min(max(y0 + dy / self.scale, 0), self.src.height - 1)
        return (x, y)

    def to_display(self, p):
        x0, y0 = self.view_origin()
        return (int(round((p[0] - x0) * self.scale)), int(round((p[1] - y0) * self.scale)))

    def zoom_at(self, dx, dy, factor):
        fx, fy = self.to_frame(dx, dy)
        self.zoom = min(max(self.zoom * factor, 1.0), MAX_ZOOM)
        # keep the frame point under the cursor fixed
        vw, vh = self.src.width / self.zoom, self.src.height / self.zoom
        x0, y0 = fx - dx / self.scale, fy - dy / self.scale
        self.center = np.array([x0 + vw / 2, y0 + vh / 2])

    # ---- state helpers ------------------------------------------------------

    def current_markup(self, create=False):
        m = self.markup.get(self.idx)
        if m is None and create:
            m = self.markup[self.idx] = Markup()
        return m

    def flash(self, text, seconds=2.5):
        self.message, self.message_until = text, time.perf_counter() + seconds

    def go_to(self, i):
        i = min(max(i, 0), self.src.count - 1)
        frame = self.src.get(i)
        while frame is None and i > 0:  # reported length was too long
            self.src.count = i
            i -= 1
            frame = self.src.get(i)
        if frame is None:
            return False
        self.idx, self.frame = i, frame
        self.pending_calib = None
        if self.window_open:
            cv2.setTrackbarPos("frame", WINDOW, self.idx)
        return True

    def pause(self):
        self.playing = False

    def ensure_detector(self):
        if self.detector is None:
            self.detector = detection.DetectionWorker(self.path, self.src.count, **self.detect_opts)
            self.detector.start()
        return self.detector

    def detections(self):
        """Detections for the frame on screen, or None while they're being computed."""
        if not self.show_detect or self.detector is None:
            return None
        dets = self.detector.get(self.idx)
        if dets is None:
            self.detector.want(self.idx, self.frame)
        return dets

    def detect_status(self):
        if not self.show_detect or self.detector is None:
            return ""
        d = self.detector
        if d.status != "ready":
            return f"  detect: {d.status}"
        scan = f" analysing {d.analysed()}/{self.src.count}" if d.scanning else ""
        return f"  detect: on{scan}"

    # ---- input --------------------------------------------------------------

    def on_mouse(self, event, x, y, flags, _param):
        if y >= self.disp_h and self.drag is None:
            return  # status strip
        y = min(y, self.disp_h - 1)
        self.mouse = self.to_frame(x, y)

        if event == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(flags) if hasattr(cv2, "getMouseWheelDelta") else flags
            self.zoom_at(x, y, 1.25 if delta > 0 else 0.8)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_down(x, y, flags)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.mode == "offside":
                self.place_player("defender", flags)
            else:
                self.drag = {"kind": "pan", "start": (x, y), "center": self.center.copy()}
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            self.drag_move(x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP) and self.drag:
            self.drag_end()

    def left_down(self, x, y, flags=0):
        if self.mode in ("line", "freehand", "calibrate", "offside"):
            self.pause()
        if self.mode == "view":
            self.drag = {"kind": "pan", "start": (x, y), "center": self.center.copy()}
        elif self.mode == "line":
            self.drag = {"kind": "line", "pts": [self.mouse, self.mouse]}
        elif self.mode == "freehand":
            self.drag = {"kind": "free", "pts": [self.mouse]}
        elif self.mode == "calibrate":
            if self.pending_calib is None:
                self.pending_calib = self.mouse
            else:
                self.current_markup(create=True).add_calib((self.pending_calib, self.mouse))
                self.pending_calib = None
        elif self.mode == "offside":
            self.place_player("attacker", flags)

    def place_player(self, role, flags=0):
        self.pause()
        point, snapped = self.mouse, False
        dets = self.detections()
        if dets and not flags & cv2.EVENT_FLAG_CTRLKEY:
            foot = detection.snap_to_player(self.mouse, dets)
            if foot is not None:
                point, snapped = foot, True
        m = self.current_markup(create=True)
        m.set_player(role, point)
        if len(m.calib) < 2:
            self.flash("Calibrate this frame (4) to draw offside lines")
        elif snapped:
            self.flash(f"{role.capitalize()} snapped to detected player's feet (Ctrl+click to place freely)")

    def drag_move(self, x, y):
        d = self.drag
        if d["kind"] == "pan":
            sx, sy = d["start"]
            self.center = d["center"] - np.array([x - sx, y - sy]) / self.scale
        elif d["kind"] == "line":
            d["pts"][1] = self.mouse
        elif d["kind"] == "free":
            if np.hypot(*np.subtract(self.mouse, d["pts"][-1])) * self.scale >= 2:
                d["pts"].append(self.mouse)

    def drag_end(self):
        d, self.drag = self.drag, None
        if d["kind"] == "line" and d["pts"][0] != d["pts"][1]:
            self.current_markup(create=True).add_shape(("line", d["pts"]))
        elif d["kind"] == "free" and len(d["pts"]) > 1:
            self.current_markup(create=True).add_shape(("free", d["pts"]))

    def on_trackbar(self, pos):
        if pos != self.idx:
            self.seek = pos

    def handle_key(self, key):
        """Returns False to quit."""
        ch = chr(key) if 0 <= key < 256 else ""
        if ch in ("q", "\x1b"):
            return False
        if ch == " ":
            if not self.playing and self.idx >= self.src.count - 1:
                self.go_to(0)
            self.playing = not self.playing
        elif ch == "d" or key in KEY_RIGHT:
            self.pause()
            self.go_to(self.idx + 1)
        elif ch == "a" or key in KEY_LEFT:
            self.pause()
            self.go_to(self.idx - 1)
        elif ch == "D":
            self.pause()
            self.go_to(self.idx + 10)
        elif ch == "A":
            self.pause()
            self.go_to(self.idx - 10)
        elif ch == "[":
            self.speed_i = max(self.speed_i - 1, 0)
        elif ch == "]":
            self.speed_i = min(self.speed_i + 1, len(SPEEDS) - 1)
        elif ch in ("+", "="):
            self.zoom_at(self.disp_w / 2, self.disp_h / 2, 1.25)
        elif ch in ("-", "_"):
            self.zoom_at(self.disp_w / 2, self.disp_h / 2, 0.8)
        elif ch == "z":
            self.zoom = 1.0
        elif ch and ch in "12345":
            self.mode = MODES[int(ch) - 1]
            self.pending_calib = None
        elif ch == "u":
            if self.pending_calib is not None:
                self.pending_calib = None
            else:
                m = self.current_markup()
                if not (m and m.undo()):
                    self.flash("Nothing to undo on this frame")
        elif ch == "x":
            self.markup.pop(self.idx, None)
            self.pending_calib = None
            self.flash("Cleared this frame")
        elif ch == "v":
            self.show_detect = not self.show_detect
            if self.show_detect:
                self.ensure_detector()
        elif ch == "p":
            self.show_detect = True
            self.ensure_detector().scan(self.idx)
            self.flash("Analysing the clip from this frame in the background")
        elif ch == "s":
            self.save_snapshot()
        elif ch == "h":
            self.show_help = not self.show_help
        return True

    # ---- output -------------------------------------------------------------

    def time_label(self, i):
        t = i / self.src.fps
        return f"{int(t // 60):02d}:{t % 60:06.3f}"

    def mode_hint(self, m):
        if self.mode == "view":
            return "Drag to pan, mouse wheel to zoom"
        if self.mode == "line":
            return "Drag to draw a straight line"
        if self.mode == "freehand":
            return "Drag to draw"
        n = len(m.calib) if m else 0
        if self.mode == "calibrate":
            if n == 0:
                return "Click two points on the GOAL LINE"
            if n == 1:
                return "Click two points on the 6-yard or 18-yard line at the same end"
            return "Calibrated. Press 5 to place players (clicking again recalibrates)"
        if n < 2:
            return "Calibrate this frame first (press 4)"
        return "Left-click attacker, right-click second-last defender (ground point)"

    def preview(self):
        if self.drag and self.drag["kind"] == "line":
            return (self.drag["pts"][0], self.drag["pts"][1], MARKUP)
        if self.mode == "calibrate" and self.pending_calib is not None:
            return (self.pending_calib, self.mouse, CALIB)
        return None

    def render(self):
        x0, y0 = self.view_origin()
        s = self.scale
        M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]])
        video = cv2.warpAffine(self.frame, M, (self.disp_w, self.disp_h), flags=cv2.INTER_LINEAR)

        m = self.current_markup()
        draw = m or Markup()
        if self.drag and self.drag["kind"] == "free":
            draw = Markup(shapes=draw.shapes + [("free", self.drag["pts"])], calib=draw.calib,
                          attacker=draw.attacker, defender=draw.defender)
        dets = self.detections()
        if dets is not None:
            trail = detection.ball_trail(self.detector.snapshot(), self.idx)
            draw_detections(video, dets, trail, self.to_display, 2)
        draw_markup(video, draw, self.src.width, self.src.height, self.to_display, 2, self.preview())

        if m and m.verdict():
            text, color = verdict_style(m.verdict())
            put_text(video, text, (20, 40), 0.9, color, 2)

        if self.show_help:
            overlay = video.copy()
            cv2.rectangle(overlay, (30, 60), (min(self.disp_w - 30, 640), 75 + 22 * len(HELP)), BLACK, -1)
            video = cv2.addWeighted(overlay, 0.75, video, 0.25, 0)
            for i, line in enumerate(HELP):
                cv2.putText(video, line, (45, 88 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

        strip = np.full((STRIP_H, self.disp_w, 3), 30, np.uint8)
        state = "PLAY" if self.playing else "PAUSED"
        marked = "  [markup]" if m and not m.is_empty() else ""
        status = (f"{self.idx + 1}/{self.src.count}  {self.time_label(self.idx)}  "
                  f"{SPEEDS[self.speed_i]:g}x {state}  zoom {self.zoom:.1f}x  "
                  f"mode: {MODES.index(self.mode) + 1} {self.mode}{marked}{self.detect_status()}  (h help)")
        cv2.putText(strip, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        if self.detector is not None and self.detector.error and self.show_detect:
            hint = f"Detection failed: {self.detector.error}"
        elif time.perf_counter() < self.message_until:
            hint = self.message
        else:
            hint = self.mode_hint(m)
        cv2.putText(strip, hint, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CALIB, 1, cv2.LINE_AA)
        return np.vstack([video, strip])

    def render_snapshot(self):
        """Full-resolution frame with markup, for saving."""
        img = self.frame.copy()
        th = max(2, self.src.width // 600)
        m = self.current_markup() or Markup()
        identity = lambda p: (int(round(p[0])), int(round(p[1])))  # noqa: E731
        dets = self.detections()
        if dets is not None:
            draw_detections(img, dets, detection.ball_trail(self.detector.snapshot(), self.idx), identity, th)
        draw_markup(img, m, self.src.width, self.src.height, identity, th)
        label = f"{self.path.name}  frame {self.idx + 1}  {self.time_label(self.idx)}"
        put_text(img, label, (15, img.shape[0] - 15), 0.4 + th * 0.15, WHITE, max(1, th // 2))
        if m.verdict():
            text, color = verdict_style(m.verdict())
            put_text(img, text, (15, 15 + th * 15), 0.5 + th * 0.3, color, th)
        return img

    def save_snapshot(self):
        out = Path("snapshots")
        out.mkdir(exist_ok=True)
        path = out / f"{self.path.stem}_f{self.idx + 1:06d}.png"
        cv2.imwrite(str(path), self.render_snapshot())
        print(f"Saved {path.resolve()}")
        self.flash(f"Saved {path}")

    # ---- main loop ----------------------------------------------------------

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("frame", WINDOW, 0, max(self.src.count - 1, 1), self.on_trackbar)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self.window_open = True
        print(f"{self.path.name}: {self.src.width}x{self.src.height}, "
              f"{self.src.fps:g} fps, {self.src.count} frames")
        print("\n".join(HELP))

        next_t = time.perf_counter()
        while True:
            if self.seek is not None:
                self.go_to(self.seek)
                self.seek = None
            now = time.perf_counter()
            if self.playing and now >= next_t:
                if self.idx >= self.src.count - 1 or not self.go_to(self.idx + 1):
                    self.playing = False
                next_t = now + 1.0 / (self.src.fps * SPEEDS[self.speed_i])

            cv2.imshow(WINDOW, self.render())
            wait = max(1, int((next_t - time.perf_counter()) * 1000)) if self.playing else 20
            key = cv2.waitKeyEx(wait)
            if key != -1 and not self.handle_key(key):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        if self.detector is not None:
            self.detector.stop()
        self.src.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="VAR review: replay, markup and offside lines")
    ap.add_argument("video", help="path to any video file OpenCV can read (mp4, mkv, avi, mov...)")
    ap.add_argument("--max-width", type=int, default=1280, help="max display width (default 1280)")
    ap.add_argument("--max-height", type=int, default=720, help="max display height (default 720)")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                    help="YOLO weights; downloaded automatically on first use (default models/yolo11n.pt)")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="detection input size; larger finds the small ball better but is slower (default 1280)")
    ap.add_argument("--device", default=None, help="torch device, e.g. cpu or 0 for the first GPU (default: auto)")
    ap.add_argument("--player-conf", type=float, default=0.35, help="min player confidence (default 0.35)")
    ap.add_argument("--ball-conf", type=float, default=0.15, help="min ball confidence (default 0.15)")
    args = ap.parse_args()
    if not Path(args.video).is_file():
        raise SystemExit(f"No such file: {args.video}")
    Path(args.weights).parent.mkdir(parents=True, exist_ok=True)
    detect_opts = dict(weights=args.weights, imgsz=args.imgsz, device=args.device,
                       player_conf=args.player_conf, ball_conf=args.ball_conf)
    VarApp(args.video, args.max_width, args.max_height, detect_opts).run()


if __name__ == "__main__":
    main()
