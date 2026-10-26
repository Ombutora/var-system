# VAR System

A Video Assistant Referee review tool for football footage. Open any video file and you can:

- **Replay it frame by frame:** play, pause, step one or ten frames, slow motion down to 0.1x, a seek bar, and zoom up to 10x at the cursor.
- **Mark it up:** draw straight lines and freehand on any frame. Markup is saved per frame.
- **Draw offside lines:** calibrate the frame's perspective from the pitch markings, then click the attacker and the second-last defender. You get true offside lines that follow the pitch's perspective, plus a verdict.
- **Save snapshots:** full-resolution PNGs of the marked-up frame, written to `snapshots/`.

## Setup

Requires Python 3.9+. On this machine plain `python` is Python 2.7, so use `py -3`:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.venv\Scripts\python.exe var_review.py path\to\clip.mp4
```

It works with any format OpenCV can decode (mp4, mkv, avi, mov...). Use `--max-width` and `--max-height` to change the window size (default 1280x720). Press **h** in the window to show the controls.

## Controls

| Key | Action |
|---|---|
| `space` | Play / pause |
| `a` `d` or `←` `→` | Step 1 frame back / forward |
| `A` `D` | Step 10 frames |
| `[` `]` | Slower / faster (0.1x – 2x) |
| mouse wheel, `+` `-` | Zoom at the cursor; `z` resets |
| `1` view | Drag to pan. Right-drag also pans in every mode except offside |
| `2` line | Drag to draw a straight line |
| `3` freehand | Drag to draw |
| `4` calibrate | Click 2 points on the goal line, then 2 points on the 6-yard or 18-yard line |
| `5` offside | Left-click the attacker, right-click the second-last defender |
| `u` | Undo the last action on this frame |
| `x` | Clear this frame's markup |
| `s` | Save a snapshot |
| `q` / `Esc` | Quit |

## Offside review workflow

1. Step to the exact frame where the ball is played.
2. Press `4`. Click two points on the **goal line**, then two points on the **6-yard or 18-yard line at the same end**. Click points far apart along each line, and zoom in for accuracy.
3. Press `5`. Left-click the **ground point directly under** the attacker's furthest-forward body part that can score (not hands or arms), then right-click the same for the second-last defender. Clicking again moves a player.
4. Read the verdict at the top left and press `s` to save the evidence.

### How it works

Lines that are parallel to the goal line on the real pitch are not parallel on screen: perspective makes them converge at a **vanishing point**. The two calibration lines locate that point. Each offside line is drawn through the player's clicked ground point and the vanishing point, so it matches the pitch markings. The verdict compares where the two offside lines cross a reference line between the calibration segments (see `geometry.py`).

### Limits

- Calibration is **per frame**, because broadcast cameras pan and zoom. Recalibrate on each frame you review.
- The verdict is only as good as your clicks. The margin is in screen pixels, not centimetres.
- Click on the ground. A point in mid-air, such as a shoulder, has to be projected straight down to the grass first, or the line will be wrong.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

The tests simulate broadcast cameras with known perspective, then check that the offside lines and verdicts match the true pitch positions.
