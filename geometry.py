"""Perspective geometry for drawing offside lines.

Lines on the pitch that are parallel to the goal line are not parallel in a
broadcast image: perspective makes them converge at a vanishing point. An
offside line through a player is therefore the image line joining the player's
ground position to that vanishing point.

Everything here uses homogeneous coordinates, so a vanishing point "at
infinity" (when the pitch lines look parallel on screen) needs no special case.
"""

import numpy as np

EPS = 1e-9


def to_h(p):
    """Image point (x, y) -> homogeneous point."""
    return np.array([float(p[0]), float(p[1]), 1.0])


def line_through(p, q):
    """Homogeneous line through two homogeneous points."""
    return np.cross(p, q)


def vanishing_point(seg_a, seg_b):
    """Intersection of two image segments, each given as ((x1, y1), (x2, y2)).

    Returns a homogeneous point. Its last component is ~0 when the segments
    are parallel in the image (the vanishing point is at infinity).
    """
    la = line_through(to_h(seg_a[0]), to_h(seg_a[1]))
    lb = line_through(to_h(seg_b[0]), to_h(seg_b[1]))
    vp = np.cross(la, lb)
    if np.allclose(vp, 0):
        raise ValueError("calibration lines are the same line")
    return vp / np.linalg.norm(vp)


def offside_line(point, vp):
    """Homogeneous image line through a player's ground point and the vanishing point."""
    return line_through(to_h(point), vp)


def clip_line(line, width, height):
    """Clip an infinite homogeneous line to the image rectangle.

    Returns ((x1, y1), (x2, y2)) as ints, or None if the line misses the image.
    """
    a, b, c = line
    pts = []
    if abs(b) > EPS:  # crossings with the left and right edges
        for x in (0.0, width - 1.0):
            y = -(a * x + c) / b
            if 0 <= y <= height - 1:
                pts.append((x, y))
    if abs(a) > EPS:  # crossings with the top and bottom edges
        for y in (0.0, height - 1.0):
            x = -(b * y + c) / a
            if 0 <= x <= width - 1:
                pts.append((x, y))
    # Corners can be found twice; keep the two points furthest apart.
    if len(pts) < 2:
        return None
    best = max(
        ((p, q) for i, p in enumerate(pts) for q in pts[i + 1:]),
        key=lambda pq: (pq[0][0] - pq[1][0]) ** 2 + (pq[0][1] - pq[1][1]) ** 2,
    )
    (x1, y1), (x2, y2) = best
    return (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))


def depth_along(line, transversal, origin, direction):
    """Where `line` crosses `transversal`, as a scalar position along `direction`."""
    p = np.cross(line, transversal)
    if abs(p[2]) < EPS:
        raise ValueError("line is parallel to the transversal")
    xy = p[:2] / p[2]
    return float(np.dot(xy - origin, direction))


def goal_depths(goal_seg, inner_seg, points):
    """How far toward the goal line each ground point is (larger = nearer the goal line).

    goal_seg  -- segment on the goal line
    inner_seg -- segment on another line parallel to it (6-yard or 18-yard line)

    Every offside line passes through the vanishing point, so the lines are
    ordered by where they cut any transversal that avoids that point. The
    transversal used joins the midpoints of the two calibration segments, and
    depths are pixels along it.
    """
    vp = vanishing_point(goal_seg, inner_seg)
    m_goal = (np.asarray(goal_seg[0], float) + np.asarray(goal_seg[1], float)) / 2
    m_inner = (np.asarray(inner_seg[0], float) + np.asarray(inner_seg[1], float)) / 2
    span = m_goal - m_inner
    if np.linalg.norm(span) < EPS:
        raise ValueError("calibration lines are too close together")
    toward_goal = span / np.linalg.norm(span)
    transversal = line_through(to_h(m_inner), to_h(m_goal))
    return [depth_along(offside_line(p, vp), transversal, m_inner, toward_goal) for p in points]


def rank_by_depth(goal_seg, inner_seg, points):
    """Indices of `points`, nearest the goal line first."""
    if not points:
        return []
    depths = goal_depths(goal_seg, inner_seg, points)
    return [int(i) for i in np.argsort(depths, kind="stable")[::-1]]


def second_last_defender(goal_seg, inner_seg, defenders, keeper_in_list=False):
    """Index into `defenders` (ground points) of the second-last defender, or None.

    The goalkeeper is normally the last defender. If the keeper is not in the
    list (their kit differs from the outfield players'), the deepest player
    listed is the second-last defender; if the keeper is in it, the second deepest.
    """
    rank = 1 if keeper_in_list else 0
    if len(defenders) <= rank:
        return None
    return rank_by_depth(goal_seg, inner_seg, defenders)[rank]


def offside_verdict(goal_seg, inner_seg, attacker, defender, tolerance=0.0):
    """Compare an attacker and the second-last defender against the goal line.

    attacker, defender -- image ground points (x, y)

    Returns ("OFFSIDE" | "ONSIDE" | "LEVEL", margin) where margin is in pixels
    (positive = attacker nearer the goal line). See goal_depths.
    """
    d_att, d_def = goal_depths(goal_seg, inner_seg, [attacker, defender])
    margin = d_att - d_def
    if abs(margin) <= tolerance:
        return "LEVEL", margin
    return ("OFFSIDE" if margin > 0 else "ONSIDE"), margin
