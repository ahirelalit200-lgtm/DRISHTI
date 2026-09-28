# Five-Minute Demo Script

A rehearsed run that shows every requirement without dead air.

---

## Before the room

```
DIAGNOSTICS.bat
```

Everything you plan to show must be **PASS**. Fix warnings now, not on stage.

Physical setup:
- Rack markers taped and visible
- Zones calibrated (`CALIBRATE_ZONES.bat`)
- Props laid out where the zones are
- **Volume up** — the voice alert is a graded requirement
- A phone on the same Wi-Fi, browser open, ready for the stream URL

Have `logs/` open in a second window.

---

## 0:00 — The problem (20s)

> "An astronaut running an experiment on the Bharatiya Antariksh Station has no
> real-time ground support — the round trip is too slow, and the bandwidth to
> stream raw video down does not exist. If they skip a step, nobody on Earth
> finds out until the science has already failed."

Press **START SESSION**.

> "This runs entirely on the payload computer. No internet, no GPU."

---

## 0:20 — Guidance (40s)

Point at the **NEXT STEP** card.

> "It tells the operator what to do next — on screen, burned into the video, and
> spoken aloud."

Perform **step 1 correctly.** Let the voice confirm it. The rail dot turns green,
the card advances.

> "Verified. It advances on its own and announces the next step."

Perform **step 2 correctly** to establish a rhythm.

---

## 1:00 — Skip detection (45s)

**Now skip step 3 and perform step 4.**

> "Watch what happens when I skip ahead."

Voice alert fires, step 3 goes amber, the log line appears.

> "It named the step I skipped and kept the session going, because that step
> was not safety-critical. It is recorded as a deviation."

---

## 1:45 — The safety block (60s) — *the moment that lands*

> "Step 5 is marked safety-critical in the protocol. Watch what changes."

**Skip step 5. Perform step 6.**

- Voice **alarm**
- Video border turns red
- Card reads **HALTED — RECOVER STEP 5**
- Guidance stops advancing

> "It has halted. It will not guide me forward until the missed step is done."

Now do something else deliberately — reach for step 7.

> "It refuses. Anything other than the missed step is ignored and logged."

**Now perform step 5.**

> "And it recovers automatically. Both the bypass and the recovery are in the
> log with timestamps."

---

## 2:45 — Out of sequence (20s)

**Repeat a step you already completed.**

> "Third violation class: a step that was already done. Different alert, and the
> cursor does not move."

---

## 3:05 — Streaming and recording (40s)

Press **STREAM**. Read the URL aloud, open it on the phone, hold it up.

> "MJPEG over HTTP to the ground station. No client software. It is streaming
> and recording locally at the same time — the recording has the overlay burned
> in, so the archive shows what the system believed and when."

---

## 3:45 — The log (45s)

Press **STOP**, then open the report.

> "Three files per session. Fixed-width for humans, JSONL for downlink, and this
> summary."

Point at the verdict.

```
Verdict         : FAIL
Steps failed    : 1  (incl. 1 bypassed safety-critical)
```

> "A bypassed safety-critical step forces a FAIL verdict. Every step has its
> outcome, its confidence, and which evidence path validated it."

---

## 4:30 — What is underneath (30s)

Point at the tier indicator.

> "Two recognition tiers. The heuristic works with zero training. The trained
> model — a GRU over rack-relative features, ONNX, CPU, offline — overrides it
> when confident. The GUI shows which one made each call."

Point at the rack quad on the video, then **rotate a marker board or tilt the rig**.

> "And this is the orientation-agnostic part. Every measurement is relative to
> the rack, not the floor. There is no 'up' in the maths. We test that at five
> rotations in the automated suite."

---

## 5:00 — Close

> "Forty-five automated tests, none of which need a camera or a GPU. The safety
> logic is a pure state machine, so it is unit-tested rather than just
> demonstrated. `RUN_TESTS.bat` if you want to see it."

---

## If something breaks

| Problem | Say this, then do it |
|---|---|
| Camera dead | "Let me switch to the recorded feed" — set `video_source` to a demo video |
| Recognition not firing | "I'll confirm manually" — press **CONFIRM STEP**. It logs as `manual` and the demo continues |
| Voice silent | "Alerts are also visual" — point at the banner and the red border |
| App will not start | `DIAGNOSTICS.bat` on screen. Reading a clean diagnostic report out loud is a better look than a blank stare |

**Never apologise for a fallback.** Say: *"That is the degraded path, and it is
visible in the status lamps — the system tells you when it is less capable
instead of pretending."* That is a design strength.

---

## Questions you will get

**"Does this need internet?"**
No. Nothing in the inference path touches the network. Streaming is outbound and
optional.

**"What is your accuracy?"**
Quote the held-out number from the model card, and say the method: an entire
operator is held out, never random windows, because consecutive windows from one
clip are near-duplicates and a random split would report a meaninglessly high
number.

**"Why not full 3D human mesh recovery?"**
HMR2.0 needs ~4 GB VRAM and runs at 5–10 fps. It recovers a full body mesh — far
more than sequence validation needs. The actual requirement is a rack-relative
reference frame, which a planar homography gives exactly, at 200+ fps, on CPU.
The architecture leaves a clean seam to add HMR as a Tier-3 module.

**"What if the markers are hidden?"**
It holds the last good frame for a grace period, then falls back to the
calibrated static quad, then to whole-image normalisation. All three are tested,
and the active mode is shown in the rack lamp.

**"How do we adapt this to another experiment?"**
Edit `configs/protocol.yaml`, recalibrate zones, record clips, retrain. No Python
changes.

**"What does not work yet?"**
Actions that differ only by which object is held. That is what the optional
Tier-2 object detector addresses. Say this before they find it.
