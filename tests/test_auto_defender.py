"""The second-last defender flow in the app, with a fake detector instead of YOLO."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

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


ATTACKERS = [player(9.0, 20, RED), player(25, -10, RED), player(30, 15, RED)]
# Blue defenders; deepest outfield at u=9.5 (wide), next at u=10.5 (central)
DEFENDERS = [player(9.5, -25, BLUE), player(10.5, 0, BLUE), player(18, 10, BLUE), player(24, -20, BLUE)]
KEEPER_LIKE_REF = [player(2.0, 3, REF)]  # grey, far from both teams


@pytest.fixture
def app(tmp_path):
    video = tmp_path / "clip.mp4"
    wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (1920, 1080))
    for _ in range(3):
        wr.write(np.zeros((1080, 1920, 3), np.uint8))
    wr.release()
    a = vr.VarApp(str(video), 1280, 720)
    a.detector = FakeDetector({0: ATTACKERS + DEFENDERS + KEEPER_LIKE_REF})
    a.show_detect = a.show_teams = True
    a.update_teams()
    assert a.teams.fitted
    m = a.current_markup(create=True)
    m.add_calib((project(0, -25), project(0, 25)))
    m.add_calib((project(16.5, -25), project(16.5, 25)))
    a.handle_key(ord("5"))
    return a


def click(app, d, button="l", flags=0):
    x, y = d.center
    sx, sy = int(x * app.base_scale), int(y * app.base_scale)
    app.on_mouse(cv2.EVENT_MOUSEMOVE, sx, sy, 0, None)
    app.on_mouse(cv2.EVENT_LBUTTONDOWN if button == "l" else cv2.EVENT_RBUTTONDOWN, sx, sy, flags, None)


def near(p, q):
    return np.hypot(p[0] - q[0], p[1] - q[1]) < 1e-6


def test_clicking_attacker_finds_deepest_outfield_defender(app):
    click(app, ATTACKERS[0])
    m = app.current_markup()
    assert m.defender_auto and near(m.defender, DEFENDERS[0].foot)
    assert m.verdict()[0] == "OFFSIDE"  # attacker u=9.0 beyond defender u=9.5
    assert app.message.startswith("Auto defender: team")


def test_keeper_rule_takes_second_deepest(app):
    click(app, ATTACKERS[0])
    app.handle_key(ord("g"))
    m = app.current_markup()
    assert m.defender_auto and near(m.defender, DEFENDERS[1].foot)


def test_manual_defender_is_kept_when_attacker_moves(app):
    click(app, ATTACKERS[0])
    click(app, DEFENDERS[2], button="r")
    m = app.current_markup()
    assert not m.defender_auto and near(m.defender, DEFENDERS[2].foot)
    click(app, ATTACKERS[1])
    assert near(m.defender, DEFENDERS[2].foot)
    app.handle_key(ord("f"))  # but f re-runs it on demand
    assert m.defender_auto and near(m.defender, DEFENDERS[0].foot)


def test_ctrl_click_still_reads_attacker_team(app):
    click(app, ATTACKERS[0], flags=cv2.EVENT_FLAG_CTRLKEY)
    m = app.current_markup()
    assert not near(m.attacker, ATTACKERS[0].foot)  # placed exactly, not snapped
    assert m.defender_auto


def test_undo_removes_auto_defender_then_attacker(app):
    click(app, ATTACKERS[0])
    m = app.current_markup()
    app.handle_key(ord("u"))
    assert m.defender is None and m.attacker is not None
    app.handle_key(ord("u"))
    assert m.attacker is None


def test_needs_teams(app):
    app.show_teams = False
    click(app, ATTACKERS[0])
    m = app.current_markup()
    assert m.defender is None
    assert "Turn on teams" in app.message


def test_attacker_off_any_player_has_no_team(app):
    app.on_mouse(cv2.EVENT_MOUSEMOVE, 5, 5, 0, None)
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, 5, 5, 0, None)
    assert app.current_markup().defender is None
    assert "team unknown" in app.message
