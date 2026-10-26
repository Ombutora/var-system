# VAR System

A Video Assistant Referee review tool for football footage. Open any video file and you can:

- **Replay it frame by frame:** play, pause, step one or ten frames, slow motion down to 0.1x, a seek bar, and zoom up to 10x at the cursor.
- **Mark it up:** draw straight lines and freehand on any frame. Markup is kept per frame while the window is open.
- **Draw offside lines:** calibrate the frame's perspective from the pitch markings, then click the attacker and the second-last defender. You get true offside lines that follow the pitch's perspective, plus a verdict. The "behind the ball" exception is applied when the ball's position is known.
- **Detect players and the ball automatically:** a pretrained YOLO model boxes every player and the ball and draws the ball's recent path. In offside mode, clicking a detected player snaps the point to their feet.
- **Split players into teams by jersey colour:** each team's boxes are drawn in its shirt colour. Referees, goalkeepers and fans in the stands are shown in grey.
- **Find the second-last defender automatically:** click the attacker, and the tool works out the defending team and draws the defender's offside line.
- **Run a fully automatic offside check:** one key works out the attacking team from who has the ball, picks the most advanced attacker and the second-last defender, and gives the verdict.
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

## Step-by-step guide

This walks through a full offside review, from opening a clip to saving the evidence. Each feature has its own section further down with more detail.

### 1. The window

- **Video:** the current frame, with your markup and any detections drawn on it. The verdict appears at the top left once there's enough to decide.
- **Seek bar** (`frame`, at the top): drag it to jump anywhere in the clip.
- **Status strip** (under the video): the first line shows the frame number, time, playback speed, play/pause, zoom, current mode and detection progress. The second line, in yellow, shows the last message or a hint for the current mode. Read it after every key press: it tells you what the tool did and what it needs next.
- **Help:** press `h` to show or hide the list of controls over the video.

### 2. Find the frame where the ball is played

Offside is judged at the moment the ball is passed, so the whole review happens on that one frame.

1. Press `space` to play and `space` again to pause near the pass.
2. Use `[` to slow down (down to 0.1x) if the moment is hard to catch.
3. Step with `a` / `d` (or the arrow keys) one frame at a time, or `A` / `D` for 10 frames, until the ball has just left the passer's foot.
4. Zoom with the mouse wheel (or `+` / `-`) on the area you care about. In view mode (`1`), drag to pan. `z` resets the zoom.

### 3. Turn on detection and teams

Press `t`. This turns on player and ball detection and colours each player by team. The first time, the YOLO model downloads (5 MB) and takes 20–30 seconds to start on CPU. The status strip shows its progress.

For the best team split, press `p` once to analyse the whole clip in the background: the more frames the tool sees, the steadier the team colours. Check that the two teams have different colours and that referees and goalkeepers are grey. If the keeper is shown in their team's colour, press `g` (see [the goalkeeper rule](#automatic-second-last-defender)).

You can skip this step and do everything by hand (step 5b).

### 4. Calibrate the frame

Press `4`, then click:

1. Two points on the **goal line**, as far apart as you can.
2. Two points on the **6-yard or 18-yard line at the same end**, also far apart.

Zoom in for each click: accuracy here sets the accuracy of every line. The two segments are drawn in yellow. Calibration belongs to this frame only, because the camera moves between frames. To redo it, press `u` and click again.

### 5a. Automatic check

Press `o`. The tool:

1. Finds the attacking team from the player who has the ball.
2. Picks the most advanced attacker (**ATT (auto)**, red line).
3. Picks the second-last defender (**DEF (auto)**, blue line).
4. Places the ball's ground point if the ball is at someone's feet (**BALL (auto)**, orange line).

Then **check every pick against the video**:

- Wrong attacking team (or "Can't tell which team is attacking")? Press `m`.
- The attacker who receives the ball isn't the most advanced one? Press `n` until it is, or left-click them.
- Wrong defender? Right-click the correct one.
- The ball is in the air, or wasn't found? Shift+click the spot on the ground directly under it.

### 5b. Manual check

Press `5`, then:

1. **Left-click the attacker.** Click the ground directly under the furthest-forward part of their body that can legally score (head, body or feet, not hands or arms). With detection on, a click inside a player's box snaps to their feet. Hold `Ctrl` to place the point exactly where you click, which is better for a close call.
2. **Right-click the second-last defender** in the same way. With teams on, this is done for you when you place the attacker. Press `f` to redo the automatic pick.
3. **Shift+click under the ball** if it's nearer the goal line than the defender, so the behind-the-ball exception applies.

Clicking again moves a point. `u` undoes one change at a time, and `x` clears the frame.

### 6. Read the verdict

The verdict appears at the top left:

| Verdict | Meaning |
|---|---|
| **OFFSIDE** | The attacker is nearer the goal line than the offside line |
| **ONSIDE** | The attacker is level with or behind the offside line |
| **LEVEL** | Within half a pixel of the line: level counts as onside |
| **ONSIDE - behind the ball** | Beyond the defender, but not beyond the ball |

The number in brackets is the margin in screen pixels: positive means beyond the line. Each line is drawn through the vanishing point, so it follows the pitch's perspective. If the lines don't look parallel to the pitch markings, the calibration is off: redo step 4.

### 7. Save the evidence

Press `s`. A full-resolution PNG of the frame, with the lines, labels, verdict, file name and timestamp, is saved to `snapshots\<clip>_f<frame>.png` in the folder you ran the tool from.

Markup is kept for each frame while the window is open, so you can step away and come back to it. It isn't saved to disk, so save a snapshot of each frame you want to keep before quitting with `q` or `Esc`.

### Tips

- On a slow computer, press `p` first and let detection run ahead before reviewing, so stepping through frames is instant.
- For wide shots where the ball is tiny, try `--ball-conf 0.1`, and be ready to choose the team with `m` and Shift+click the ball.
- Always zoom in before placing points for a close call. A few pixels can change the verdict.
- Treat the automatic picks as suggestions. The tool can only use players it detected and gave a team.

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
| `o` | Fully automatic offside check: attacking team, attacker and defender (needs teams) |
| `n` | Next attacker: check the next most advanced attacker |
| `m` | Switch the attacking team |
| `f` | Find the second-last defender automatically (needs teams) |
| `g` | Switch the keeper rule: keeper wears team colours, or not (default) |
| `Ctrl`+click | In offside mode, place the point exactly instead of snapping to a detected player |
| `Shift`+click | In offside mode, mark the ball's ground point (for the behind-the-ball exception) |
| `u` | Undo the last action on this frame |
| `x` | Clear this frame's markup |
| `s` | Save a snapshot |
| `q` / `Esc` | Quit |

## Offside review workflow

1. Step to the exact frame where the ball is played.
2. Press `4`. Click two points on the **goal line**, then two points on the **6-yard or 18-yard line at the same end**. Click points far apart along each line, and zoom in for accuracy.
3. Press `5`. Left-click the **ground point directly under** the attacker's furthest-forward body part that can score (not hands or arms), then right-click the same for the second-last defender. Clicking again moves a player.
4. Read the verdict at the top left and press `s` to save the evidence.

With teams on (`t`), step 3 is a single click. Left-click the attacker, and the defender line is placed for you. Or press `o`, and both players are chosen for you (see below).

### Behind the ball

Under Law 11 an attacker is only in an offside position if they are nearer the goal line than **both** the second-last defender and the ball. So on a cut-back, an attacker beyond the defender but level with or behind the ball is onside.

- **Automatic:** when you place the attacker (by click or `o`) and the detected ball is at a player's feet, its ground point is taken from the bottom of the ball's box and labelled **BALL (auto)**.
- **By hand:** a ball in the air can't be placed on the pitch from its box, because its ground point is somewhere below it. Shift+click the spot on the ground directly under the ball. Your point is kept when the attacker or `o` is re-run, and `u` undoes it.

The offside line is whichever of the defender and the ball is nearer the goal line. When the ball sets it, the verdict says so: **ONSIDE - behind the ball**, **LEVEL with the ball** or **OFFSIDE (beyond the ball)**, with the margin measured from the ball. With no ball point, the verdict is judged against the defender only.

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

## Automatic second-last defender

With the frame calibrated and teams on, left-click inside the attacker's box. Then:

1. **Defending team:** the attacker's team (A or B) is read from the box you clicked, so the defending team is the other one. Ctrl+click still reads the team but places the point exactly.
2. **Ranking:** every visible defending player is ranked by how close their feet are to the goal line, measured along the pitch's perspective, not by height on screen.
3. **Placing the line:** the second-last defender is chosen, and their line is drawn labelled **DEF (auto)**.

**The goalkeeper rule.** The offside line is set by the second-last opponent, and the last one is usually the goalkeeper.

- **Default:** keepers wear a different kit, so they normally show up grey and aren't counted in the team. The deepest outfield defender is then the second-last defender. This also works when the keeper is out of shot.
- **Press `g`** if the keeper is grouped with their team. The tool then skips the deepest player and takes the second deepest.
- **If the keeper has come off their line,** so that an outfield player is the last defender, right-click the correct defender.

**Overriding it:**

- A right-click always sets the defender by hand, and moving the attacker afterwards keeps your choice.
- Press `f` to re-run the automatic pick on this frame.
- `u` undoes the auto defender first, then the attacker.

**It can only rank players who were detected and assigned to a team.** If the real second-last defender is hidden, merged with another player's box, or shown grey, the tool picks someone else. Always check the pick before trusting the verdict.

## Fully automatic offside check

On the frame where the ball is played, calibrate (`4`), turn on teams (`t`) and press **`o`**:

1. **Attacking team:** the team of the player whose feet are nearest the ball. At the moment of the pass, that's the passer. "Nearest" has a limit based on that player's size on screen, so a ball in the air or out of reach doesn't count.
2. **Attacker:** every attacking player is ranked by distance to the goal line, measured in the pitch's perspective. The passer is left out, because a player can't be offside from their own pass. The **most advanced** attacker is checked first: if they're onside, every attacker is. Their line is labelled **ATT (auto)**.
3. **Defender:** the second-last defender is found as described above, unless you've already set one by hand.
4. **Ball:** if the ball is at a player's feet, its ground point is placed for the behind-the-ball exception (see above). Otherwise the message asks you to Shift+click under it.

**When to step in:**

- **`n`** moves to the next most advanced attacker, wrapping round at the end. Use it when the player who actually receives the ball isn't the one furthest forward. Only the attacker who becomes involved in play can be penalised.
- **`m`** switches the attacking team, and the choice stays for this frame. Use it when the ball isn't detected (common in wide shots), when no player is close enough to it, or when the nearest player is a defender. The tool always attacks the goal you calibrated on, so if the defending team has the ball, there's no offside to check at that goal.
- A left-click on any player replaces the auto attacker. Undo steps back one change at a time.

**Limits:**

- The pick is only as good as the ball and player detection and the team colours.
- An attacker who wasn't detected or got no team is never considered.
- The behind-the-ball exception needs the ball's ground point. If the ball is in the air or not detected, Shift+click under it, or the verdict ignores the ball.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

The tests simulate broadcast cameras with known perspective, then check that the offside lines and verdicts match the true pitch positions. They also cover how detections are filtered, how clicks snap to players, and how the ball trail is built. Team tests draw synthetic players in different lighting, with a referee and fans, and check the split. The auto-defender and auto-attacker tests run the app with a fake detector (see `tests/conftest.py`). They cover picking by pitch depth, the keeper rule, finding the ball carrier, leaving out the passer, cycling attackers, switching teams, manual overrides, Ctrl+click, undo and the behind-the-ball exception (auto ball point, Shift+click, a ball in the air). The tests don't need the YOLO model.
