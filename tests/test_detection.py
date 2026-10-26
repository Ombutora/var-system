import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection import (PERSON, SPORTS_BALL, Detection, ball_trail,  # noqa: E402
                       parse_detections, snap_to_player)


def test_parse_keeps_confident_players_and_best_ball():
    xyxy = [[0, 0, 10, 30], [50, 0, 60, 30], [5, 5, 7, 7], [20, 20, 22, 22], [80, 0, 90, 30]]
    cls = [PERSON, PERSON, SPORTS_BALL, SPORTS_BALL, 2]  # last one is a car
    conf = [0.9, 0.2, 0.3, 0.6, 0.99]
    dets = parse_detections(xyxy, cls, conf, player_conf=0.35, ball_conf=0.15)
    players = [d for d in dets if d.kind == "player"]
    balls = [d for d in dets if d.kind == "ball"]
    assert len(players) == 1 and players[0].box == (0, 0, 10, 30)
    assert len(balls) == 1 and balls[0].box == (20, 20, 22, 22)


def test_parse_handles_empty_output():
    assert parse_detections([], [], []) == []


def test_foot_is_bottom_centre():
    assert Detection("player", (10, 20, 30, 100), 0.9).foot == (20, 100)


def test_snap_picks_smallest_overlapping_player():
    near = Detection("player", (100, 100, 120, 160), 0.9)  # small, in front
    far = Detection("player", (80, 50, 200, 300), 0.9)  # big box overlapping it
    ball = Detection("ball", (105, 105, 115, 115), 0.5)
    assert snap_to_player((110, 110), [far, near, ball]) == near.foot
    assert snap_to_player((190, 60), [far, near]) == far.foot
    assert snap_to_player((5, 5), [far, near]) is None


def test_ball_trail_skips_frames_without_ball():
    ball = lambda x: Detection("ball", (x, 0, x + 2, 2), 0.5)  # noqa: E731
    results = {0: [ball(0)], 1: [], 3: [ball(30)], 4: [ball(40)]}
    assert ball_trail(results, 4, length=10) == [(1, 1), (31, 1), (41, 1)]
    assert ball_trail(results, 4, length=1) == [(31, 1), (41, 1)]
