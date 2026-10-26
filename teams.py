"""Split detected players into two teams by jersey colour.

Each player's colour is the median of their upper body (shirt) with grass
pixels masked out, in Lab colour space so that brightness changes from
shadows and floodlights matter less than hue. Two clusters are found with
k-means. Anyone far from both clusters (referees, goalkeepers, fans) is
"other" rather than being forced into a team.
"""

import cv2
import numpy as np

TEAM_LABELS = ("A", "B")
OTHER = "other"
MIN_PLAYERS = 4
L_WEIGHT = 0.5  # brightness counts half as much as hue/saturation (a, b)


def grass_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (hue >= 35) & (hue <= 85) & (sat > 60) & (val > 40)


def on_grass(frame, box, min_fraction=0.2):
    """True if the player stands on grass: green around their feet. Fans in the stands fail this."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = box
    h = y2 - y1
    # A band at boot height, plus a little below the box
    by1, by2 = int(max(0, y2 - 0.08 * h)), int(min(h_img, y2 + 0.04 * h + 1))
    bx1, bx2 = int(max(0, x1)), int(min(w_img, x2))
    if by2 - by1 < 1 or bx2 - bx1 < 1:
        return False
    return grass_mask(frame[by1:by2, bx1:bx2]).mean() >= min_fraction


def jersey_color(frame, box):
    """Median shirt colour of a player box as (lab, bgr), or None if the box is too small."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    # Shirt: 15-50% down the box, middle 60% across (skips head, arms, shorts and background)
    cx1, cx2 = int(max(0, x1 + 0.2 * w)), int(min(w_img, x2 - 0.2 * w))
    cy1, cy2 = int(max(0, y1 + 0.15 * h)), int(min(h_img, y1 + 0.5 * h))
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    crop = frame[cy1:cy2, cx1:cx2]

    keep = ~grass_mask(crop)
    if keep.sum() < 0.1 * keep.size:  # all green: maybe a green kit, so use everything
        keep[:] = True

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[keep].astype(float)
    bgr = crop[keep].astype(float)
    return tuple(np.median(lab, axis=0)), tuple(np.median(bgr, axis=0))


def _feature(lab):
    lab = np.asarray(lab, float)
    return lab * np.array([L_WEIGHT, 1.0, 1.0])


class TeamClassifier:
    def __init__(self):
        self.centers = None  # (2, 3) in feature space
        self.colors = None  # BGR display colour per team
        self.threshold = None

    @property
    def fitted(self):
        return self.centers is not None

    def fit(self, labs, bgrs):
        """Fit two jersey clusters. Returns False if there are too few players."""
        if len(labs) < MIN_PLAYERS:
            return False
        X = _feature(labs).astype(np.float32)
        bgrs = np.asarray(bgrs, float)

        centers = self._kmeans(X)
        # Refit without outliers so referees and fans don't drag the team colours
        dist, assign = self._nearest(X, centers)
        threshold = max(20.0, 3.0 * float(np.median(dist)))
        inliers = dist <= threshold
        if inliers.sum() >= MIN_PLAYERS and len(np.unique(assign[inliers])) == 2:
            centers = self._kmeans(X[inliers])
            dist, assign = self._nearest(X, centers)
            threshold = max(20.0, 3.0 * float(np.median(dist[dist <= threshold])))

        # Keep team A as team A across refits
        if self.centers is not None:
            same = np.linalg.norm(centers - self.centers, axis=1).sum()
            swapped = np.linalg.norm(centers[::-1] - self.centers, axis=1).sum()
            if swapped < same:
                centers = centers[::-1]
                dist, assign = self._nearest(X, centers)

        colors = []
        for t in range(2):
            members = (assign == t) & (dist <= threshold)
            colors.append(tuple(int(v) for v in np.median(bgrs[members], axis=0)) if members.any()
                          else (128, 128, 128))
        self.centers, self.colors, self.threshold = centers, colors, threshold
        return True

    def classify(self, lab):
        """"A", "B" or "other" for one jersey colour."""
        if not self.fitted or lab is None:
            return OTHER
        dist, assign = self._nearest(_feature([lab]), self.centers)
        return TEAM_LABELS[assign[0]] if dist[0] <= self.threshold else OTHER

    def color(self, team):
        return self.colors[TEAM_LABELS.index(team)] if self.fitted and team in TEAM_LABELS else None

    @staticmethod
    def _kmeans(X):
        cv2.setRNGSeed(0)  # same clusters every run
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
        _, _, centers = cv2.kmeans(np.ascontiguousarray(X, np.float32), 2, None, criteria, 5,
                                   cv2.KMEANS_PP_CENTERS)
        return centers.astype(float)

    @staticmethod
    def _nearest(X, centers):
        d = np.linalg.norm(np.asarray(X, float)[:, None, :] - centers[None, :, :], axis=2)
        return d.min(axis=1), d.argmin(axis=1)
