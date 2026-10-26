# VAR System

A Video Assistant Referee review tool for football footage. Open any video file and you can:

- **Replay it frame by frame:** play, pause, step one or ten frames, slow motion down to 0.1x, a seek bar, and zoom up to 10x at the cursor.
- **Mark it up:** draw straight lines and freehand on any frame. Markup is saved per frame.
- **Draw offside lines:** calibrate the frame's perspective from the pitch markings, then click the attacker and the second-last defender. You get true offside lines that follow the pitch's perspective, plus a verdict.
- **Detect players and the ball automatically:** a pretrained YOLO model boxes every player and the ball and draws the ball's recent path. In offside mode, clicking a detected player snaps the point to their feet.
- **Split players into teams by jersey colour:** each team's boxes are drawn in its shirt colour. Referees, goalkeepers and fans in the stands are shown in grey.
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

Detection options: `--imgsz` (default 1280), `--device` (`cpu`, or `0` for the first GPU), `--player-conf`, `--ball-conf` and `--weights`.

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
| `v` | Turn player and ball detection on or off |
| `p` | Analyse the whole clip in the background, starting from this frame |
| `t` | Turn team colours on or off (also turns on detection) |
| `Ctrl`+click | In offside mode, place the point exactly instead of snapping to a detected player |
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

## Player and ball detection

Press `v` to turn on detection. The first time, it downloads the YOLO11n weights (5 MB) into `models/`. Starting the model takes about 20–30 seconds on CPU.

- **Players** get green boxes, with a dot at the bottom centre of each box. That dot is where the player meets the ground.
- **The ball** gets an orange circle. Once neighbouring frames have been analysed, its recent path is drawn as a trail.
- **Snapping:** in offside mode, clicking inside a player's box places the point at their feet. This is quick, but it's the centre of the box, not the furthest-forward foot. For a close call, Ctrl+click the exact spot.

Detection runs on a background thread, so the window never freezes. The frame on screen is always analysed first. On a CPU each frame takes about 0.3 s at the default `--imgsz 1280`, so playback with detection on lags behind. Press `p` to analyse the whole clip ahead of time, and then playback and ball trails are instant. A lower `--imgsz` (e.g. 640) is about 3x faster but misses more balls. A CUDA GPU (`--device 0`) is much faster.

**Limits:** the model is trained on general photos (the COCO "person" and "sports ball" classes), not football broadcasts.

- It also boxes referees, coaches and fans near the pitch.
- In wide shots the ball is only a few pixels across, so it is often missed. Lowering `--ball-conf` finds more balls but adds false hits.
- Ultralytics YOLO is licensed under AGPL-3.0.

## Team detection

Press `t` to show teams. Each player's box is drawn in their team's shirt colour with an **A** or **B** tag, and a legend appears at the top right. In offside mode, the message after a snap says which team the player is on.

How it works (see `teams.py`):

1. **Shirt colour:** for each player, the tool takes the median colour of the upper middle of the box (the shirt), ignoring grass pixels. It uses Lab colour space, so shadows and floodlights change the result less than the actual kit colour does.
2. **On the pitch:** a player counts only if there's grass around their feet. This leaves out fans and staff in the stands.
3. **Two clusters:** k-means groups the shirt colours into two teams. It then refits without outliers, so a referee doesn't pull a team's colour off. Anyone far from both team colours, such as referees and usually goalkeepers, is shown in grey.
4. **More data over time:** the teams are learned from every analysed frame and refitted each time that number doubles. Team A keeps its name when this happens. Pressing `p` to analyse the clip makes the teams more stable.

**Limits:**

- The two teams need clearly different kits. Similar colours (e.g. dark navy vs black) can get mixed up.
- A green kit is hard to tell apart from the grass.
- Players hidden behind others may have the wrong colour, or no team if their feet are hidden.
- A goalkeeper whose kit looks like one team's kit is put in that team.
- "Team A" and "Team B" are only names. The tool doesn't know which side is attacking.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

The tests simulate broadcast cameras with known perspective, then check that the offside lines and verdicts match the true pitch positions. They also cover how detections are filtered, how clicks snap to players, and how the ball trail is built. Team tests draw synthetic players in different lighting, with a referee and fans, and check the split. The tests don't need the YOLO model.
