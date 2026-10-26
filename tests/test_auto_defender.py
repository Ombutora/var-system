"""The second-last defender flow in the app, with a fake detector instead of YOLO."""

import cv2
import pytest
from conftest import BLUE, RED, REF, click, make_app, near, player

ATTACKERS = [player(9.0, 20, RED), player(25, -10, RED), player(30, 15, RED)]
# Blue defenders; deepest outfield at u=9.5 (wide), next at u=10.5 (central)
DEFENDERS = [player(9.5, -25, BLUE), player(10.5, 0, BLUE), player(18, 10, BLUE), player(24, -20, BLUE)]
KEEPER_LIKE_REF = [player(2.0, 3, REF)]  # grey, far from both teams


@pytest.fixture
def app(tmp_path):
    return make_app(tmp_path, ATTACKERS + DEFENDERS + KEEPER_LIKE_REF)


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
