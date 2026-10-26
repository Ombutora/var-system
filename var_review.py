"""VAR review tool: frame-accurate replay, markup and perspective offside lines.

Usage:  python var_review.py path/to/clip.mp4
Press h in the window for the controls.
"""

import argparse
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

import detection
import geometry
import teams

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
OTHER_PERSON = (150, 150, 150)  # referees, goalkeepers, fans when teams are shown
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
    "           Shift+click: the ball's ground point (behind-the-ball rule)",
    "v          player / ball detection on/off   p analyse whole clip",
    "t          teams by jersey colour on/off (grey = referee/keeper/other)",
    "o          full auto offside check: attacking team from the ball,",
    "           most advanced attacker, second-last defender (needs teams)",
    "n          next attacker    m switch attacking team",
    "f          find the second-last defender automatically (needs teams)",
    "g          keeper rule: keeper in team colours or not",
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
    attacker_team: str = None  # "A" / "B" when the attacker was clicked on a detected player
    defender_auto: bool = False  # defender chosen automatically rather than clicked
    attacker_auto: bool = False  # attacker chosen automatically (o)
    attacker_rank: int = 0  # which auto attacker: 0 = most advanced (n cycles)
    attack_team_override: str = None  # attacking team chosen with m instead of from the ball
    ball: tuple = None  # the ball's ground point, for the behind-the-ball exception
    ball_auto: bool = False  # ball point taken from detection rather than Shift+clicked
    history: list = field(default_factory=list)  # for undo

    def to_json(self):
        """Plain JSON data for the markup file. The undo history isn't saved."""
        return {k: to_plain(getattr(self, k)) for k in self.SAVED_FIELDS}

    @classmethod
    def from_json(cls, data):
        m = cls(**{k: data[k] for k in cls.SAVED_FIELDS if k in data})
        m.shapes = [(kind, [tuple(p) for p in pts]) for kind, pts in m.shapes]
        m.calib = [tuple(tuple(p) for p in seg) for seg in m.calib]
        for k in ("attacker", "defender", "ball"):
            if getattr(m, k) is not None:
                setattr(m, k, tuple(getattr(m, k)))
        return m

    def is_empty(self):
        return not (self.shapes or self.calib or self.attacker or self.defender or self.ball)

    def add_shape(self, shape):
        self.shapes.append(shape)
        self.history.append(("shape", None))

    def add_calib(self, seg):
        if len(self.calib) == 2:  # a third segment starts calibration over
            self.history.append(("calib_reset", list(self.calib)))
            self.calib = []
        self.calib.append(seg)
        self.history.append(("calib", None))

    PLAYER_FIELDS = ("attacker", "defender", "attacker_team", "defender_auto",
                     "attacker_auto", "attacker_rank", "attack_team_override", "ball", "ball_auto")
    SAVED_FIELDS = ("shapes", "calib") + PLAYER_FIELDS  # what goes in the markup file

    def set_players(self, **changes):
        if all(getattr(self, k) == v for k, v in changes.items()):
            return  # nothing changes, so nothing to undo
        self.history.append(("players", {k: getattr(self, k) for k in self.PLAYER_FIELDS}))
        for k, v in changes.items():
            setattr(self, k, v)

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
        elif kind == "players":
            for k, v in prev.items():
                setattr(self, k, v)
        return True

    def verdict(self):
        if len(self.calib) < 2 or self.attacker is None or self.defender is None:
            return None
        try:
            return geometry.offside_decision(
                self.calib[0], self.calib[1], self.attacker, self.defender, self.ball, tolerance=0.5
            )
        except ValueError:
            return None


def to_plain(v):
    """Tuples to lists and numpy numbers to Python ones, so json can write them."""
    if isinstance(v, (list, tuple)):
        return [to_plain(x) for x in v]
    return v.item() if isinstance(v, np.generic) else v


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

    def_label = "DEF (auto)" if m.defender_auto else "DEF"
    att_label = "ATT (auto)" if m.attacker_auto else "ATT"
    ball_label = "BALL (auto)" if m.ball_auto else "BALL"
    for point, color, label in ((m.ball, BALL, ball_label), (m.attacker, ATTACKER, att_label),
                                (m.defender, DEFENDER, def_label)):
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


def contrast(color):
    """Black or white, whichever reads better against a BGR colour."""
    return BLACK if 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2] > 140 else WHITE


def draw_tag(img, text, corner, color, th):
    """Small filled label in a team's colour, with readable text on top."""
    scale = 0.35 + th * 0.1
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = corner
    cv2.rectangle(img, (x - 1, y - h - 7), (x + w + 7, y + 1), contrast(color), -1)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, contrast(color), 1, cv2.LINE_AA)


def draw_detections(img, dets, trail, to_img, th, team_of=None):
    """team_of(detection) -> (label, colour), or None to draw plain player boxes."""
    for d in dets:
        if d.kind == "player":
            label, color = team_of(d) if team_of else (None, PLAYER_BOX)
            x1, y1, x2, y2 = d.box
            tl, br = to_img((x1, y1)), to_img((x2, y2))
            if label:  # contrasting outline so dark or pale kits still show up
                cv2.rectangle(img, tl, br, contrast(color), max(1, th // 2) + 2, cv2.LINE_AA)
            cv2.rectangle(img, tl, br, color, max(1, th // 2), cv2.LINE_AA)
            cv2.circle(img, to_img(d.foot), th + 1, color, -1, cv2.LINE_AA)
            if label and label != teams.OTHER:
                draw_tag(img, label, tl, color, th)
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
    name, margin, line = verdict
    color = {"OFFSIDE": ATTACKER, "ONSIDE": (60, 200, 60), "LEVEL": CALIB}[name]
    if line == "ball":  # the ball was nearer the goal line than the defender
        name = {"OFFSIDE": "OFFSIDE (beyond the ball)", "ONSIDE": "ONSIDE - behind the ball",
                "LEVEL": "LEVEL with the ball"}[name]
    return f"{name}  ({margin:+.1f} px)", color


class VarApp:
    def __init__(self, path, max_w, max_h, detect_opts=None, markup_path=None):
        self.path = Path(path)
        self.detect_opts = detect_opts or {}
        self.detector = None
        self.show_detect = False
        self.teams = teams.TeamClassifier()
        self.show_teams = False
        self.teams_fit_at = 0  # analysed-frame count at the last fit
        self.keeper_in_team = False  # keeper's kit matches the outfield players'
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
        self.show_help = False
        self.message, self.message_until = "", 0.0
        # Markup is kept in a file next to the clip, so it survives closing the window
        self.markup_path = Path(markup_path) if markup_path else self.path.with_name(self.path.name + ".var.json")
        self.markup = self.load_markup()  # frame index -> Markup
        self.saved_text = self.markup_text()

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

    def load_markup(self):
        """Read this clip's markup file, if there is one. Returns {frame index: Markup}."""
        try:
            data = json.loads(self.markup_path.read_text(encoding="utf-8"))
            markup = {int(i): Markup.from_json(d) for i, d in data["frames"].items()}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            # Don't overwrite a file we couldn't read: the markup in it may still be wanted
            msg = f"Could not read {self.markup_path.name} ({e}): markup won't be saved this session"
            print(msg)
            self.flash(msg, 10)
            self.markup_path = None
            return {}
        if markup:
            print(f"Loaded markup for {len(markup)} frame(s) from {self.markup_path}")
            self.flash(f"Loaded markup for {len(markup)} frame(s) from {self.markup_path.name}", 5)
        return markup

    def markup_text(self):
        frames = {str(i): m.to_json() for i, m in sorted(self.markup.items()) if not m.is_empty()}
        return json.dumps({"version": 1, "video": self.path.name, "frames": frames}, indent=1)

    def save_markup(self):
        """Write the markup file if anything changed since the last save. Returns True if it wrote."""
        if self.markup_path is None:
            return False
        text = self.markup_text()
        if text == self.saved_text:
            return False
        self.saved_text = text  # on failure, try again only after the next change
        tmp = self.markup_path.with_name(self.markup_path.name + ".tmp")
        try:  # write then rename, so a crash mid-write can't leave a half-written file
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.markup_path)
        except OSError as e:
            self.flash(f"Could not save markup to {self.markup_path}: {e}", 5)
            return False
        return True

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

    def update_teams(self):
        """Fit the jersey clusters, and refit as more frames are analysed (each time the count doubles)."""
        if not (self.show_teams and self.show_detect and self.detector):
            return
        n = self.detector.analysed()
        if n == 0 or (self.teams.fitted and n < 2 * self.teams_fit_at) or n == self.teams_fit_at:
            return
        players = [d for dets in self.detector.snapshot().values() for d in dets
                   if d.kind == "player" and d.lab is not None]
        on_pitch = [d for d in players if d.on_pitch]
        if len(on_pitch) >= teams.MIN_PLAYERS:  # else the pitch may not be green: use everyone
            players = on_pitch
        if len(players) > 3000:  # plenty for two clusters; keep refits instant
            players = [players[i] for i in np.random.default_rng(0).choice(len(players), 3000, replace=False)]
        self.teams.fit([p.lab for p in players], [p.bgr for p in players])
        self.teams_fit_at = n  # also on failure, so we only retry once more frames arrive

    def team_of(self, d):
        label = self.teams.classify(d.lab) if d.on_pitch else teams.OTHER
        return label, self.teams.color(label) or OTHER_PERSON

    def detect_status(self):
        if not self.show_detect or self.detector is None:
            return ""
        d = self.detector
        if d.status != "ready":
            return f"  detect: {d.status}"
        scan = f" analysing {d.analysed()}/{self.src.count}" if d.scanning else ""
        team = ""
        if self.show_teams:
            team = "  teams: on" if self.teams.fitted else "  teams: need more players"
        return f"  detect: on{scan}{team}"

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
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.place_ball()
            else:
                self.place_player("attacker", flags)

    def teams_ready(self):
        return self.show_teams and self.teams.fitted

    def place_player(self, role, flags=0):
        self.pause()
        dets = self.detections()
        player = detection.player_at(self.mouse, dets) if dets else None
        snap = player is not None and not flags & cv2.EVENT_FLAG_CTRLKEY
        point = player.foot if snap else self.mouse
        m = self.current_markup(create=True)

        if role == "defender":
            m.set_players(defender=point, defender_auto=False)
            if len(m.calib) < 2:
                self.flash("Calibrate this frame (4) to draw offside lines")
            elif snap:
                self.flash("Defender snapped to player's feet (Ctrl+click to place freely)")
            return

        # The team comes from the player clicked on, even with Ctrl (exact placement)
        team = self.team_of(player)[0] if player is not None and self.teams_ready() else None
        m.set_players(attacker=point, attacker_team=team, attacker_auto=False)
        ball_msg = self.auto_ball(m, dets) if dets is not None else ""
        if len(m.calib) < 2:
            self.flash("Calibrate this frame (4) to draw offside lines")
        elif m.defender is None or m.defender_auto:
            self.flash(f"{self.auto_defender(m)} {ball_msg}".strip(), 5)
        elif snap:
            self.flash(f"Attacker snapped to player's feet (Ctrl+click to place freely). {ball_msg}".strip())

    def place_ball(self):
        self.pause()
        m = self.current_markup(create=True)
        m.set_players(ball=self.mouse, ball_auto=False)
        self.flash("Ball's ground point set: attackers level with or behind it are onside")

    def auto_ball(self, m, dets):
        """Take the ball's ground point from detection if it is at a player's feet. Returns a short message.

        A ball in the air can't be placed on the pitch from its box alone (its
        ground point is somewhere below it), so it is left for a Shift+click.
        """
        if m.ball is not None and not m.ball_auto:
            return "Ball: your point."
        ball = next((d for d in dets if d.kind == "ball"), None)
        carrier = detection.ball_carrier(ball, [d for d in dets if d.kind == "player"])
        if carrier is not None:
            x1, _, x2, y2 = ball.box
            m.set_players(ball=((x1 + x2) / 2, y2), ball_auto=True)
            return "Ball: auto."
        if m.ball_auto:  # the detected ball is gone or no longer at anyone's feet
            m.set_players(ball=None, ball_auto=False)
        if ball is None:
            return "Ball: not found (Shift+click under it)."
        return "Ball: in the air? Shift+click under it."

    def auto_attacker(self, m, rank=0, team_override=None):
        """Pick the attacking team and the attacker to check, then the defender. Returns a message.

        The attacking team is the team of the player nearest the ball (the
        passer, at the moment of the pass) unless chosen with m. Attackers are
        ranked nearest the goal line first, leaving out the passer, who can't be
        offside from their own pass. Rank 0, the most advanced, is the one to
        check: if they are onside, every attacker is.
        """
        if len(m.calib) < 2:
            return "Calibrate this frame (4) first"
        if not self.teams_ready():
            return "Turn on teams (t) for the automatic offside check"
        dets = self.detections()
        if dets is None:
            return "Detections not ready yet: press o again in a moment"

        in_team = [(d, self.team_of(d)[0]) for d in dets if d.kind == "player"]
        in_team = [(d, t) for d, t in in_team if t in teams.TEAM_LABELS]
        ball = next((d for d in dets if d.kind == "ball"), None)
        carrier = detection.ball_carrier(ball, [d for d, _ in in_team])

        override = team_override or m.attack_team_override
        if override:
            team, why = override, "chosen with m"
        elif carrier is not None:
            team, why = self.team_of(carrier)[0], "has the ball"
        else:
            reason = "ball not detected" if ball is None else "no player next to the ball"
            return f"Can't tell which team is attacking ({reason}): press m to choose"

        attackers = [d for d, t in in_team if t == team and d is not carrier]
        if not attackers:
            return f"No team {team} attackers visible apart from the ball carrier"
        try:
            order = geometry.rank_by_depth(m.calib[0], m.calib[1], [d.foot for d in attackers])
        except ValueError:
            return "Calibration lines are invalid: recalibrate (4)"
        rank %= len(order)
        m.set_players(attacker=attackers[order[rank]].foot, attacker_team=team, attacker_auto=True,
                      attacker_rank=rank, attack_team_override=override)

        which = "most advanced" if rank == 0 else f"{rank + 1} of {len(order)}"
        msg = f"Team {team} attacking ({why}). Attacker: {which} (n next, m switch team)."
        if m.defender is None or m.defender_auto:
            d_msg = self.auto_defender(m)
            msg += " Defender: auto." if m.defender_auto else f" {d_msg}"
        return f"{msg} {self.auto_ball(m, dets)}"

    def auto_defender(self, m):
        """Place the second-last defender automatically. Returns a message saying what happened."""
        if len(m.calib) < 2:
            return "Calibrate this frame (4) first"
        if not self.teams_ready():
            return "Turn on teams (t) to find the second-last defender automatically"
        dets = self.detections()
        if dets is None:
            return "Detections not ready yet: press f to try again"
        if m.attacker is None:
            return "Place the attacker first (left-click)"
        if m.attacker_team not in teams.TEAM_LABELS:
            return "Attacker's team unknown: left-click inside their box"
        defending = teams.TEAM_LABELS[1 - teams.TEAM_LABELS.index(m.attacker_team)]
        candidates = [d for d in dets if d.kind == "player" and self.team_of(d)[0] == defending]
        try:
            i = geometry.second_last_defender(m.calib[0], m.calib[1], [d.foot for d in candidates],
                                              keeper_in_list=self.keeper_in_team)
        except ValueError:
            return "Calibration lines are invalid: recalibrate (4)"
        if i is None:
            return f"Not enough team {defending} players visible: right-click the defender"
        m.set_players(defender=candidates[i].foot, defender_auto=True)
        rule = "2nd deepest, keeper in team colours" if self.keeper_in_team else "deepest outfield player"
        return f"Auto defender: team {defending} {rule} (g: keeper rule, right-click: override)"

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
        elif ch == "t":
            self.show_teams = not self.show_teams
            if self.show_teams and not self.show_detect:
                self.show_detect = True
                self.ensure_detector()
        elif ch == "o":
            self.pause()
            self.mode = "offside"
            self.flash(self.auto_attacker(self.current_markup(create=True)), 5)
        elif ch == "n":
            m = self.current_markup()
            if m and m.attacker_auto:
                self.flash(self.auto_attacker(m, rank=m.attacker_rank + 1), 5)
            else:
                self.flash("Press o first to pick the attacker automatically")
        elif ch == "m":
            self.pause()
            self.mode = "offside"
            m = self.current_markup(create=True)
            current = m.attacker_team if m.attacker_team in teams.TEAM_LABELS else teams.TEAM_LABELS[1]
            other = teams.TEAM_LABELS[1 - teams.TEAM_LABELS.index(current)]
            self.flash(self.auto_attacker(m, team_override=other), 5)
        elif ch == "f":
            self.pause()
            self.flash(self.auto_defender(self.current_markup(create=True)))
        elif ch == "g":
            self.keeper_in_team = not self.keeper_in_team
            m = self.current_markup()
            if m and m.defender_auto:
                self.flash(self.auto_defender(m))
            else:
                self.flash("Keeper counted in team colours" if self.keeper_in_team
                           else "Keeper not in team colours: deepest outfield player is second-last")
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
        if self.teams_ready():
            return "Press o for a full auto check, or left-click the attacker (defender found automatically)"
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
            draw = replace(draw, shapes=draw.shapes + [("free", self.drag["pts"])])
        dets = self.detections()
        self.update_teams()
        team_of = self.team_of if self.show_teams and self.teams.fitted else None
        if dets is not None:
            trail = detection.ball_trail(self.detector.snapshot(), self.idx)
            draw_detections(video, dets, trail, self.to_display, 2, team_of)
        if team_of:
            x = self.disp_w - 10
            for label in reversed(teams.TEAM_LABELS):
                text = f"Team {label}"
                w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 6
                x -= w
                draw_tag(video, text, (x, 30), self.teams.color(label), 2)
                x -= 8
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
            team_of = self.team_of if self.show_teams and self.teams.fitted else None
            draw_detections(img, dets, detection.ball_trail(self.detector.snapshot(), self.idx),
                            identity, th, team_of)
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

        if self.markup_path is not None:
            print(f"Markup is saved to {self.markup_path}")

        next_t = next_save = time.perf_counter()
        try:
            while True:
                if self.seek is not None:
                    self.go_to(self.seek)
                    self.seek = None
                now = time.perf_counter()
                if self.playing and now >= next_t:
                    if self.idx >= self.src.count - 1 or not self.go_to(self.idx + 1):
                        self.playing = False
                    next_t = now + 1.0 / (self.src.fps * SPEEDS[self.speed_i])
                if now >= next_save:  # autosave, so a crash loses at most a second of work
                    self.save_markup()
                    next_save = now + 1.0

                cv2.imshow(WINDOW, self.render())
                wait = max(1, int((next_t - time.perf_counter()) * 1000)) if self.playing else 20
                key = cv2.waitKeyEx(wait)
                if key != -1 and not self.handle_key(key):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.save_markup()
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
    ap.add_argument("--markup", default=None,
                    help="markup file to load and save (default: <video>.var.json next to the video)")
    args = ap.parse_args()
    if not Path(args.video).is_file():
        raise SystemExit(f"No such file: {args.video}")
    Path(args.weights).parent.mkdir(parents=True, exist_ok=True)
    detect_opts = dict(weights=args.weights, imgsz=args.imgsz, device=args.device,
                       player_conf=args.player_conf, ball_conf=args.ball_conf)
    VarApp(args.video, args.max_width, args.max_height, detect_opts, args.markup).run()


if __name__ == "__main__":
    main()
