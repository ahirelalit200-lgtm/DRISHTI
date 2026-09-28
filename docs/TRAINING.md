# Training Guide

How to go from Tier 0 (heuristic) to Tier 1 (a trained model) in about an hour.

---

## Before you record anything

Two things must be right first, or you will record a dataset that cannot produce
a good model.

### 1. Finalise the protocol

Edit `configs/protocol.yaml` so the `action` labels are exactly the classes you
want the model to learn. Changing them later invalidates your dataset.

Rules of thumb:

* **One label per visually distinct motion.** If two steps look identical to a
  camera, give them the same `action` and let the `zone` distinguish them.
* **Reuse labels deliberately.** Two `open_box` steps at different zones is
  fine and is fewer classes to learn.
* **Keep it under ~10 classes** for a first model. More classes need more clips.

### 2. Calibrate zones

```
MAKE_ARUCO_MARKERS.bat     print, cut, tape to the rack corners
CALIBRATE_ZONES.bat        click the rack quad, then draw each zone
```

**This must happen before recording.** Zone occupancy is part of the feature
vector, so the feature width depends on how many zones exist. Recalibrating with
a different number of zones changes the vector width and invalidates every clip
you recorded — the trainer will refuse to mix them, which is the safe behaviour
but still costs you the recording session.

Verify with `DIAGNOSTICS.bat`: you want **PASS** on zones and on the landmark
backend before you spend half an hour recording.

---

## Recording

```
RECORD_DATASET.bat
```

It asks for an operator ID. **Use a different ID per person.** This is not
bookkeeping — it is what makes the accuracy number meaningful.

| Key | Action |
|---|---|
| `1`–`9`, `0` | select the label to record |
| `TAB` | cycle labels |
| `i` | jump to `idle` |
| `SPACE` | start / stop a clip |
| `d` | delete the clip you just recorded |
| `q` | finish |

### The recipe

| Target | Why |
|---|---|
| **12–20 clips per action** | Below ~8 the class is unlearnable; the recorder flags it |
| **At least 3 operators** | One is held out; two leaves nothing to generalise across |
| **Trim tight** | The clip must contain *only* the action. Leading/trailing idle frames teach the model that idle looks like the action |
| **Vary everything harmless** | Distance, lighting, sleeve colour, hand used, speed |
| **Record plenty of `idle`** | 20+ clips. This is the class that stops the system firing constantly. Include hands resting, reaching past, adjusting, scratching your nose |

### What makes a bad dataset

* All clips from one person, one session, one lighting condition → reports 99%, fails live
* Clips that start before the action begins → the model learns the approach, not the action
* No `idle` class → constant false positives
* Fewer than 5 clips for a class → the trainer warns you; believe it

### What is stored

Not video — the **feature sequence**. Each clip is a `(frames, ~90)` float array,
roughly 30 kB. Two hundred clips is 6 MB, trains in minutes, and contains no
identifiable imagery, which also means you are not shipping a dataset of your
teammates' faces.

Add `--keep-video` if you want the raw mp4 alongside, so you can re-extract later
if the feature layout changes.

---

## Training

```
TRAIN_MODEL.bat
```

Defaults are sensible. Useful overrides:

```
TRAIN_MODEL.bat --epochs 60 --hidden 128 --holdout op3
```

| Flag | Meaning | Default |
|---|---|---|
| `--holdout` | Operator held out for validation | last one alphabetically |
| `--epochs` | Training epochs | 40 |
| `--hidden` | GRU hidden size | 96 |
| `--window` | Frames per window (must match `app.yaml`) | 32 |
| `--augment` | Augmentation factor, 1 = off | 3 |
| `--force-numpy` | Ignore PyTorch even if installed | off |

### Backends

**PyTorch** if installed — 2-layer GRU, faster, slightly better.
**NumPy** otherwise — the same architecture with hand-written forward and
backward passes, exported by emitting the ONNX graph directly.

The NumPy path exists so the deliverable does not require a 2.5 GB dependency.
It trains in minutes rather than seconds; for datasets this size that is fine.

Both produce an identical artefact: `action_model.onnx` + `.meta.json` +
`MODEL_CARD.md`.

---

## Reading the result

```
CLASS                      SUPPORT   RECALL  PRECISION     F1
-------------------------------------------------------------
idle                           141    0.94       0.91    0.92
open_outer_box                 162    0.88       0.90    0.89
retrieve_red_box               165    0.71       0.68    0.69
transfer_sample                147    0.86       0.89    0.87

Held-out accuracy: 84.7%  (operator: op3)
```

**Held-out accuracy is the only number worth quoting.** Validation holds out an
entire operator, never random windows — consecutive windows from one clip are
near-duplicates, so a random split reports ~99% and then collapses on a new
person. The number in the model card is the honest one, and it is the one to put
on a slide.

### Diagnosing a weak class

Look at per-class recall, not just the headline number.

| Symptom | Cause | Fix |
|---|---|---|
| One class much worse than the rest | Too few clips, or it looks like a neighbour | Record 10 more; check the two are actually distinguishable |
| Two classes confused with each other | Same motion, same zone | Give them distinct zones, or merge them into one label |
| Everything ~60% | Feature signal is weak | Check the landmark backend is not degraded — `DIAGNOSTICS.bat` |
| High train, low held-out | Overfitting to one operator | Add operators. This is the most common cause |
| `idle` recall low | Not enough idle variety | Record idle during realistic downtime, not staged stillness |

### If accuracy is below 75%

The trainer tells you this and ranks the fixes. In order of effect:

1. More clips per class (aim 15–20)
2. More operators
3. Re-run `CALIBRATE_ZONES.bat` — vague zones blur similar actions
4. Trim clips tighter

**The app keeps working meanwhile.** A weak model does not degrade the system: if
the learned tier is not confident above `learned_min_confidence`, Tier 0 stays in
charge. You can ship with Tier 0 and improve the model afterwards.

---

## Deployment

The runtime picks up the model automatically on next launch. Verify:

```
DIAGNOSTICS.bat
```

You want:

```
--- Action model ---
  [ OK ]  8 classes, window 32
          features v2, dim 62
  [ OK ]  held-out accuracy 84.7% (operator op3)
```

The GUI tier indicator flips to **Tier 1 - learned + heuristic**.

### The compatibility interlock

The runtime **refuses** to load a model whose feature version or dimension does
not match the current configuration. This is deliberate. Without it, the classic
failure is: recalibrate zones → feature width silently changes → the model keeps
loading and predicts confident nonsense. Instead you get a clear message in the
tier detail line telling you to retrain.

If you see `model expects 62 features, current zone set yields 68 - retrain`,
that is the interlock doing its job.

---

## Tuning the runtime

In `configs/app.yaml`:

| Setting | Raise it when | Lower it when |
|---|---|---|
| `learned_min_confidence` | The model fires wrongly | The model is right but never trusted |
| `stability_frames` | Flickering detections | Steps feel sluggish to register |
| `prefer_learned: false` | You want to A/B against Tier 0 | — |

Per-step in `protocol.yaml`:

| Setting | Effect |
|---|---|
| `min_confidence` | Evidence bar for that step. Raise for safety-critical |
| `dwell_s` | How long the action must persist. Raise for slow, deliberate steps |
| `timeout_s` | When to nudge the operator |

---

## Optional: Tier 2 object detection

Only worth it if two actions differ **solely by which object is held** — identical
motion, identical location, different item. That is the one case landmarks cannot
resolve.

1. Record video with `--keep-video`
2. Label object boxes (Roboflow, CVAT, or Label Studio)
3. Train YOLO: `yolo detect train data=data.yaml model=yolo11n.pt epochs=100`
4. Put the weights at `models/objects/best.pt`
5. Set `objects_enabled: true` in `app.yaml`

The system does not depend on this. It is additive.
