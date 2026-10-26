"""Saving markup next to the clip and loading it again, with a fake detector instead of YOLO."""

import json

import numpy as np
import var_review as vr
from conftest import BLUE, RED, ball_at_feet, make_app, near, player

PASSER = player(5.0, 0, RED)
FORWARD = player(9.0, 20, RED)
DEFENDERS = [player(9.5, -25, BLUE), player(10.5, 0, BLUE), player(18, 10, BLUE)]
SCENE = [PASSER, ball_at_feet(PASSER), FORWARD, player(25, -10, RED)] + DEFENDERS


def reopen(app):
    return vr.VarApp(str(app.path), 1280, 720)


def test_markup_survives_reopening_the_clip(tmp_path):
    app = make_app(tmp_path, SCENE)
    app.handle_key(ord("o"))
    m = app.current_markup()
    m.add_shape(("line", [(10.0, 20.0), (300.0, 400.0)]))
    assert app.save_markup() and app.markup_path == tmp_path / "clip.mp4.var.json"

    loaded = reopen(app).markup[0]
    for k in vr.Markup.SAVED_FIELDS:
        assert getattr(loaded, k) == getattr(m, k), k
    assert isinstance(loaded.attacker, tuple) and isinstance(loaded.calib[0][0], tuple)
    assert loaded.verdict() == m.verdict() and loaded.verdict()[2] == "ball"
    assert loaded.history == []  # undo history isn't saved
    assert "Loaded markup for 1 frame(s)" in reopen(app).message


def test_numpy_coordinates_are_saved_as_plain_numbers(tmp_path):
    app = make_app(tmp_path, SCENE)
    app.current_markup().set_players(attacker=(np.float32(100.5), np.float64(200.25)))
    app.save_markup()
    data = json.loads(app.markup_path.read_text(encoding="utf-8"))
    assert data["frames"]["0"]["attacker"] == [100.5, 200.25]
    assert near(reopen(app).markup[0].attacker, (100.5, 200.25))


def test_saves_only_when_something_changed(tmp_path):
    app = make_app(tmp_path, SCENE)
    assert app.save_markup()
    assert not app.save_markup()
    app.handle_key(ord("o"))
    assert app.save_markup()
    assert not reopen(app).save_markup()  # freshly loaded: nothing new to write


def test_no_file_for_a_clip_without_markup(tmp_path):
    app = make_app(tmp_path, SCENE)
    app.handle_key(ord("x"))  # clear the calibration make_app added
    assert not app.save_markup() and not app.markup_path.exists()


def test_cleared_frame_is_removed_from_the_file(tmp_path):
    app = make_app(tmp_path, SCENE)
    app.save_markup()
    app.handle_key(ord("x"))
    assert app.save_markup()
    assert reopen(app).markup == {}


def test_unreadable_file_is_left_alone(tmp_path):
    app = make_app(tmp_path, SCENE)
    app.markup_path.write_text("{not json", encoding="utf-8")
    again = reopen(app)
    assert again.markup == {} and "Could not read clip.mp4.var.json" in again.message
    again.markup = app.markup
    assert not again.save_markup()
    assert app.markup_path.read_text(encoding="utf-8") == "{not json"


def test_markup_path_can_be_chosen(tmp_path):
    app = make_app(tmp_path, SCENE)
    other = tmp_path / "review" / "notes.json"
    other.parent.mkdir()
    app.markup_path = other
    app.save_markup()
    loaded = vr.VarApp(str(app.path), 1280, 720, markup_path=other).markup
    assert len(loaded[0].calib) == 2 and not (tmp_path / "clip.mp4.var.json").exists()
