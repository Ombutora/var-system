import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import teams  # noqa: E402

GRASS = (40, 140, 50)
RED, BLUE, YELLOW = (40, 30, 200), (180, 60, 20), (30, 220, 240)


def draw_player(frame, x, y, shirt, shade=1.0, w=40, h=100):
    """A crude player: grass shows through either side of a narrow body, as in a real box."""
    s = tuple(int(min(255, c * shade)) for c in shirt)
    frame[y:y + 15, x + 14:x + 26] = (120, 160, 210)  # head
    frame[y + 15:y + 50, x + 10:x + 30] = s  # shirt, narrower than the box
    frame[y + 50:y + 70, x + 12:x + 28] = (240, 240, 240)  # shorts
    frame[y + 70:y + h, x + 13:x + 27] = (30, 30, 30)  # socks and boots
    return (x, y, x + w, y + h)


def make_scene(rng, n_per_team=6, referee=True):
    frame = np.zeros((400, 1200, 3), np.uint8)
    frame[:] = GRASS
    boxes, truth = [], []
    x = 10
    for shirt, label in [(RED, "A")] * n_per_team + [(BLUE, "B")] * n_per_team + (
            [(YELLOW, teams.OTHER)] if referee else []):
        shade = rng.uniform(0.7, 1.15)  # shadows and floodlights
        boxes.append(draw_player(frame, x, int(rng.integers(10, 280)), shirt, shade))
        truth.append(label)
        x += 80
    return frame, boxes, truth


def colors(frame, boxes):
    c = [teams.jersey_color(frame, b) for b in boxes]
    return [x[0] for x in c], [x[1] for x in c]


def test_jersey_color_ignores_grass_and_finds_shirt():
    frame = np.zeros((200, 200, 3), np.uint8)
    frame[:] = GRASS
    box = draw_player(frame, 50, 50, RED)
    _, bgr = teams.jersey_color(frame, box)
    assert np.allclose(bgr, RED, atol=2)


def test_tiny_box_has_no_colour():
    assert teams.jersey_color(np.zeros((50, 50, 3), np.uint8), (10, 10, 11, 11)) is None


def test_two_teams_and_referee_separated():
    frame, boxes, truth = make_scene(np.random.default_rng(1))
    labs, bgrs = colors(frame, boxes)
    clf = teams.TeamClassifier()
    assert clf.fit(labs, bgrs)
    got = [clf.classify(lab) for lab in labs]
    # Team names are arbitrary; the red players must share one and the blue the other.
    red, blue = {got[i] for i in range(6)}, {got[i] for i in range(6, 12)}
    assert len(red) == 1 and len(blue) == 1 and red != blue
    assert red | blue == {"A", "B"}
    assert got[12] == teams.OTHER  # referee
    # Display colours come from the real shirts
    assert np.allclose(clf.color(red.pop()), RED, atol=40)


def test_labels_stay_put_when_refitting():
    rng = np.random.default_rng(2)
    clf = teams.TeamClassifier()
    frame, boxes, _ = make_scene(rng)
    labs, bgrs = colors(frame, boxes)
    clf.fit(labs, bgrs)
    red_team = clf.classify(labs[0])
    for _ in range(5):  # more frames arrive, refit each time
        frame, boxes, _ = make_scene(rng)
        more_labs, more_bgrs = colors(frame, boxes)
        labs, bgrs = labs + more_labs, bgrs + more_bgrs
        clf.fit(labs, bgrs)
        assert clf.classify(more_labs[0]) == red_team


def test_on_grass_tells_players_from_fans():
    frame = np.zeros((300, 300, 3), np.uint8)
    frame[150:] = GRASS  # pitch below, stands (grey seats) above
    frame[:150] = (90, 90, 90)
    player = draw_player(frame, 120, 180, RED)
    fan = draw_player(frame, 20, 20, BLUE)
    assert teams.on_grass(frame, player)
    assert not teams.on_grass(frame, fan)


def test_needs_enough_players():
    frame, boxes, _ = make_scene(np.random.default_rng(3), n_per_team=1, referee=False)
    clf = teams.TeamClassifier()
    assert not clf.fit(*colors(frame, boxes))
    assert clf.classify((100, 150, 150)) == teams.OTHER
