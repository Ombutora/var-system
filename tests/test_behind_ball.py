"""The behind-the-ball exception in the app, with a fake detector instead of YOLO."""

import cv2
import var_review as vr
from conftest import BLUE, RED, ball_at_feet, make_app, near, player
from detection import Detection

# Attacker at u=9.0 is beyond the deepest outfield defender at u=9.5
FORWARDS = [player(25, -10, RED), player(9.0, 20, RED), player(12, -5, RED)]
DEFENDERS = [player(9.5, -25, BLUE), player(10.5, 0, BLUE), player(18, 10, BLUE), player(24, -20, BLUE)]
CUTBACK_PASSER = player(5.0, 0, RED)  # passing back from near the byline
DEEP_PASSER = player(20, 5, RED)


def shift_click(app, point):
    sx, sy = int(point[0] * app.base_scale), int(point[1] * app.base_scale)
    app.on_mouse(cv2.EVENT_MOUSEMOVE, sx, sy, 0, None)
    app.on_mouse(cv2.EVENT_LBUTTONDOWN, sx, sy, cv2.EVENT_FLAG_SHIFTKEY, None)


def test_cutback_beyond_defender_but_behind_ball_is_onside(tmp_path):
    app = make_app(tmp_path, [CUTBACK_PASSER, ball_at_feet(CUTBACK_PASSER)] + FORWARDS + DEFENDERS)
    app.handle_key(ord("o"))
    m = app.current_markup()
    assert near(m.attacker, FORWARDS[1].foot) and near(m.defender, DEFENDERS[0].foot)
    assert m.ball_auto and m.ball is not None
    assert m.verdict()[0] == "ONSIDE" and m.verdict()[2] == "ball"
    assert vr.verdict_style(m.verdict())[0].startswith("ONSIDE - behind the ball")
    assert "Ball: auto." in app.message


def test_ball_further_back_than_defender_changes_nothing(tmp_path):
    app = make_app(tmp_path, [DEEP_PASSER, ball_at_feet(DEEP_PASSER)] + FORWARDS + DEFENDERS)
    app.handle_key(ord("o"))
    verdict, _, line = app.current_markup().verdict()
    assert (verdict, line) == ("OFFSIDE", "defender")


def test_ball_in_the_air_is_left_for_shift_click(tmp_path):
    flying = Detection("ball", (60, 40, 70, 50), 0.5)  # nowhere near anyone's feet
    app = make_app(tmp_path, [DEEP_PASSER, flying] + FORWARDS + DEFENDERS)
    app.handle_key(ord("o"))
    assert "no player next to the ball" in app.message  # nobody carries it: choose the team
    app.handle_key(ord("m"))
    m = app.current_markup()
    if m.attacker_team != app.team_of(DEEP_PASSER)[0]:
        app.handle_key(ord("m"))  # make red the attacking team
    assert m.ball is None and "in the air" in app.message
    assert near(m.attacker, FORWARDS[1].foot)  # DEEP_PASSER has no ball, so is an attacker too
    assert m.verdict()[0] == "OFFSIDE"  # judged against the defender only

    shift_click(app, CUTBACK_PASSER.foot)  # mark the ground under the ball near the byline
    assert m.ball is not None and not m.ball_auto
    assert m.verdict()[:3:2] == ("ONSIDE", "ball")

    app.handle_key(ord("o"))  # re-running keeps the hand-placed ball
    assert not m.ball_auto and "Ball: your point." in app.message

    app.handle_key(ord("u"))  # undo the Shift+click (the re-run changed nothing)
    assert m.ball is None


def test_no_ball_detected_says_so(tmp_path):
    app = make_app(tmp_path, [DEEP_PASSER] + FORWARDS + DEFENDERS)
    app.handle_key(ord("m"))  # no ball, so choose the attacking team by hand
    assert app.current_markup().ball is None and "Ball: not found" in app.message


def test_manual_attacker_click_also_picks_up_ball(tmp_path):
    from conftest import click
    app = make_app(tmp_path, [CUTBACK_PASSER, ball_at_feet(CUTBACK_PASSER)] + FORWARDS + DEFENDERS)
    click(app, FORWARDS[1])
    m = app.current_markup()
    assert m.ball_auto and m.verdict()[0] == "ONSIDE"


def test_ball_line_renders(tmp_path):
    app = make_app(tmp_path, [CUTBACK_PASSER, ball_at_feet(CUTBACK_PASSER)] + FORWARDS + DEFENDERS)
    app.handle_key(ord("o"))
    assert app.render().shape[0] == app.disp_h + vr.STRIP_H
    assert app.render_snapshot().shape[:2] == (1080, 1920)
