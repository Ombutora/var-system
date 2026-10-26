"""The full automatic offside check (o / n / m) with a fake detector instead of YOLO."""

from conftest import BLUE, RED, REF, ball_at_feet, make_app, near, player, team_label

PASSER = player(20, 5, RED)
# Red attackers by distance from the goal line: u=9.0 (wide), 12, 25
FORWARDS = [player(25, -10, RED), player(9.0, 20, RED), player(12, -5, RED)]
DEFENDERS = [player(9.5, -25, BLUE), player(10.5, 0, BLUE), player(18, 10, BLUE), player(24, -20, BLUE)]
REFEREE = player(15, 28, REF)


def test_most_advanced_attacker_and_defender(tmp_path):
    app = make_app(tmp_path, [PASSER, ball_at_feet(PASSER)] + FORWARDS + DEFENDERS + [REFEREE])
    app.handle_key(ord("o"))
    m = app.current_markup()
    assert m.attacker_auto and near(m.attacker, FORWARDS[1].foot)  # u=9.0
    assert m.attacker_team == team_label(app, PASSER)
    assert m.defender_auto and near(m.defender, DEFENDERS[0].foot)  # u=9.5
    assert m.verdict()[0] == "OFFSIDE"
    assert "has the ball" in app.message and "most advanced" in app.message


def test_n_cycles_attackers_by_depth_skipping_the_passer(tmp_path):
    app = make_app(tmp_path, [PASSER, ball_at_feet(PASSER)] + FORWARDS + DEFENDERS)
    app.handle_key(ord("o"))
    m = app.current_markup()
    seen = []
    for _ in range(4):
        seen.append(m.attacker)
        app.handle_key(ord("n"))
    expected = [FORWARDS[1].foot, FORWARDS[2].foot, FORWARDS[0].foot, FORWARDS[1].foot]  # wraps
    assert all(near(a, e) for a, e in zip(seen, expected))
    assert near(m.defender, DEFENDERS[0].foot)  # defender unchanged throughout


def test_passer_furthest_forward_is_not_the_attacker_checked(tmp_path):
    passer = player(5.0, 0, RED)  # e.g. a cut-back from the byline
    app = make_app(tmp_path, [passer, ball_at_feet(passer)] + FORWARDS + DEFENDERS)
    app.handle_key(ord("o"))
    assert near(app.current_markup().attacker, FORWARDS[1].foot)


def test_no_ball_asks_for_team_then_m_picks_one(tmp_path):
    app = make_app(tmp_path, [PASSER] + FORWARDS + DEFENDERS)
    app.handle_key(ord("o"))
    m = app.current_markup()
    assert m.attacker is None and "ball not detected" in app.message
    app.handle_key(ord("m"))
    assert m.attacker_auto and m.attack_team_override is not None
    attacking = m.attacker_team
    red = team_label(app, PASSER)
    # The passer has no ball now, so he's an attacker too; the most advanced red is still u=9.0
    if attacking == red:
        assert near(m.attacker, FORWARDS[1].foot)
    else:
        assert near(m.attacker, DEFENDERS[0].foot)  # blue attacking: deepest blue player
    app.handle_key(ord("m"))  # switch to the other team
    assert m.attacker_team != attacking


def test_ball_with_a_defender_makes_their_team_attack_until_m(tmp_path):
    holder = DEFENDERS[2]
    app = make_app(tmp_path, [PASSER, ball_at_feet(holder)] + FORWARDS + DEFENDERS)
    app.handle_key(ord("o"))
    m = app.current_markup()
    blue, red = team_label(app, holder), team_label(app, PASSER)
    assert m.attacker_team == blue
    app.handle_key(ord("m"))
    assert m.attacker_team == red and "chosen with m" in app.message
    app.handle_key(ord("o"))  # the choice sticks for this frame
    assert m.attacker_team == red


def test_manual_click_replaces_auto_attacker(tmp_path):
    from conftest import click
    app = make_app(tmp_path, [PASSER, ball_at_feet(PASSER)] + FORWARDS + DEFENDERS)
    app.handle_key(ord("o"))
    click(app, FORWARDS[2])
    m = app.current_markup()
    assert not m.attacker_auto and near(m.attacker, FORWARDS[2].foot)
    app.handle_key(ord("n"))
    assert "Press o first" in app.message


def test_undo_restores_previous_attacker(tmp_path):
    app = make_app(tmp_path, [PASSER, ball_at_feet(PASSER)] + FORWARDS + DEFENDERS)
    app.handle_key(ord("o"))
    m = app.current_markup()
    app.handle_key(ord("n"))
    app.handle_key(ord("u"))
    assert near(m.attacker, FORWARDS[1].foot) and m.attacker_rank == 0


def test_needs_calibration(tmp_path):
    app = make_app(tmp_path, [PASSER, ball_at_feet(PASSER)] + FORWARDS + DEFENDERS)
    app.handle_key(ord("x"))  # clears calibration too
    app.handle_key(ord("o"))
    assert "Calibrate" in app.message
