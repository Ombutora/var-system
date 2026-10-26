import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import geometry  # noqa: E402

# Pitch coordinates (metres): u = distance out from the goal line, v = across the pitch.
# A homography with strong perspective, like a high camera near the halfway line.
PERSPECTIVE = np.array([
    [38.0, -6.0, 900.0],
    [4.0, 9.0, 260.0],
    [0.012, 0.004, 1.0],
])
# Pure affine camera: pitch lines stay parallel on screen (vanishing point at infinity).
AFFINE = np.array([
    [30.0, -5.0, 300.0],
    [3.0, 12.0, 150.0],
    [0.0, 0.0, 1.0],
])


def project(H, u, v):
    p = H @ np.array([u, v, 1.0])
    return (p[0] / p[2], p[1] / p[2])


def segment(H, u, v1=-15, v2=15):
    """Image segment along the pitch line u = const (parallel to the goal line)."""
    return (project(H, u, v1), project(H, u, v2))


@pytest.mark.parametrize("H", [PERSPECTIVE, AFFINE], ids=["perspective", "affine"])
@pytest.mark.parametrize("inner_u", [5.5, 16.5], ids=["six-yard", "eighteen-yard"])
def test_attacker_nearer_goal_is_offside(H, inner_u):
    goal, inner = segment(H, 0), segment(H, inner_u)
    # Players at very different widths, so a naive horizontal comparison would be wrong.
    attacker = project(H, 9.6, 25)
    defender = project(H, 10.0, -20)
    verdict, margin = geometry.offside_verdict(goal, inner, attacker, defender)
    assert verdict == "OFFSIDE"
    assert margin > 0


@pytest.mark.parametrize("H", [PERSPECTIVE, AFFINE], ids=["perspective", "affine"])
def test_attacker_behind_defender_is_onside(H):
    goal, inner = segment(H, 0), segment(H, 16.5)
    attacker = project(H, 10.4, -25)
    defender = project(H, 10.0, 20)
    verdict, margin = geometry.offside_verdict(goal, inner, attacker, defender)
    assert verdict == "ONSIDE"
    assert margin < 0


def test_same_depth_is_level():
    goal, inner = segment(PERSPECTIVE, 0), segment(PERSPECTIVE, 16.5)
    attacker = project(PERSPECTIVE, 11.0, 28)
    defender = project(PERSPECTIVE, 11.0, -28)
    verdict, _ = geometry.offside_verdict(goal, inner, attacker, defender, tolerance=0.5)
    assert verdict == "LEVEL"


def test_calibration_segment_direction_does_not_matter():
    goal, inner = segment(PERSPECTIVE, 0), segment(PERSPECTIVE, 16.5)
    goal_rev, inner_rev = goal[::-1], inner[::-1]
    attacker, defender = project(PERSPECTIVE, 9.6, 25), project(PERSPECTIVE, 10.0, -20)
    assert geometry.offside_verdict(goal_rev, inner_rev, attacker, defender)[0] == "OFFSIDE"


def test_offside_line_follows_pitch_line():
    """The drawn line through a player must pass through other points at the same depth."""
    vp = geometry.vanishing_point(segment(PERSPECTIVE, 0), segment(PERSPECTIVE, 16.5))
    line = geometry.offside_line(project(PERSPECTIVE, 12.0, -10), vp)
    for v in (-30, 0, 30):
        x, y = project(PERSPECTIVE, 12.0, v)
        a, b, c = line
        assert abs(a * x + b * y + c) / np.hypot(a, b) < 1e-6


def test_second_last_defender_uses_pitch_depth_not_screen_position():
    goal, inner = segment(PERSPECTIVE, 0), segment(PERSPECTIVE, 16.5)
    # (u = metres from goal line, v = across). Wide players look "further up" on screen.
    defenders = [project(PERSPECTIVE, u, v) for u, v in [(20, -30), (8.0, 30), (12, 0), (8.5, -30)]]
    # Keeper not in the list: the deepest outfield player (u = 8.0) is second-last
    assert geometry.second_last_defender(goal, inner, defenders) == 1
    # Keeper in the list (u = 1): skip them and take the next deepest
    with_keeper = defenders + [project(PERSPECTIVE, 1.0, 5)]
    assert geometry.second_last_defender(goal, inner, with_keeper, keeper_in_list=True) == 1


def test_second_last_defender_needs_enough_players():
    goal, inner = segment(PERSPECTIVE, 0), segment(PERSPECTIVE, 16.5)
    one = [project(PERSPECTIVE, 10, 0)]
    assert geometry.second_last_defender(goal, inner, []) is None
    assert geometry.second_last_defender(goal, inner, one, keeper_in_list=True) is None
    assert geometry.second_last_defender(goal, inner, one) == 0


def test_identical_calibration_lines_rejected():
    seg = ((0, 0), (10, 10))
    with pytest.raises(ValueError):
        geometry.vanishing_point(seg, seg)


def test_clip_line_spans_image():
    # Vertical line x = 50 in a 200x100 image
    ends = geometry.clip_line(np.array([1.0, 0.0, -50.0]), 200, 100)
    assert sorted(ends) == [(50, 0), (50, 99)]
    # Diagonal through a corner is not duplicated
    ends = geometry.clip_line(geometry.line_through(geometry.to_h((0, 0)), geometry.to_h((1, 1))), 100, 100)
    assert sorted(ends) == [(0, 0), (99, 99)]


def test_clip_line_missing_image():
    assert geometry.clip_line(np.array([1.0, 0.0, 500.0]), 200, 100) is None
