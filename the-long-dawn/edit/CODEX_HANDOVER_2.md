# Codex handover 2: from the director (Claude), 27 Sep 2026 ~23:50Z

Thank you: the handback was exemplary.

## What happened to your PRs

- **PR #1 is merged** into `claude/long-dawn-v2` as a fast-forward to `b614e3c`. All 36 tests were re-run on the production Mac and pass.
- **PR #2 was reviewed separately and is not merged.** Keep it open as the record.
  - Council material v1: the fragmented char works; the circuit-trace look on the logs is gone. The dark fuel heads suit the *spent* torches at 5566. But at 5160 the torches must be alight, and they read as smouldering. The slab, the figures and the miniature look are unchanged, and the council is being rebuilt separately.
  - The three A15 cairn alternatives are rejected (the user's call). Do not reuse them.

## Ground rules

- Work only on `codex/*` branches, and open PRs against `claude/long-dawn-v2`. Never push `claude/long-dawn-v2`.
- **Don't touch files that have unpublished work on the production Mac:**
  - `cloud/farm.py`
  - `edit/NOTES_v3.md`
  - `edit/edl/edl_A.json`, `edit/edl/edl_B.json`
  - `music/sound/events_B.json`, `music/src/sound_*.py`
  - `shots/run/reveal_a.py`
- No farm or cloud operations. Local test renders on the M4 only, written to your own directories; 960×402 is fine.
- No creative redesigns. Keep the accepted constraints:
  - A's uncoded giants, with the coded-towers ALT kept separate;
  - B's single-night Vigil;
  - C's book and ink language, and the invented Ring script;
  - silhouettes, gloves and the red shawl only;
  - each film's own landforms;
  - the white lantern core and the A17→A18 match cut.
- The M4 is also a render node, and its owner runs heavy jobs on it. Use at most 4 threads, and pause heavy work if its load is high.

## Tasks, in priority order

### 1. Terrain-ceiling audit (urgent: these renders have already landed)

For each shot, compare the ceiling it uses with a 3000 m diagnostic, using your depth-strip method. Report which frames change, where, and the maximum pixel difference.

| Shot | Frames on disk | Frames to test |
|---|---|---|
| A2 FALSE DAWN (`falsedawn.py`) | A 80–559, rendered with the OLD 700 m bound before PR #1 | 80, 300, 380, 500, 559. Does the new ceiling change the far wall? If so, we re-render A2 on the farm. |
| A11 STARS (`stars_a.py`) | 3120–3359 | 3120 and 3359; the ranges are a faint strip at the bottom |
| A18 CROSSING (`crossing.py`) | 960 frames, A 4880–5839 | 4880, 5100, 5300, 5480, 5700, 5839 |
| A13 REVEAL (`reveal_a.py`, pushed version) | 3680–3799 | 3680, 3740, 3799. Report only; don't patch this file. |
| bluehour (A19, not yet rendered), `hills/beacon`, `bh_probe` | none | report whether they clip |

Deliver a table (shot, frame, clipped y/n, where, max diff, action). Open a PR with ceiling fixes for any clipping caller that is not on the WIP list above.

### 2. The legacy `cloud/run_job.py`

A nonzero renderer exit must fail the job even if every expected image was pushed; you flagged this yourself. Deliver a small PR with a test.

### 3. A frame-QC tool

We run it on the production Mac, where the renders are. Write `edit/tools/frame_qc.py` with tests on synthetic frames. Given a render dir (`f_%05d.jpg`/`.png`) and a frame range, it reports:

- missing frames, and frames that fail to decode or aren't 1920×804;
- black, blank or near-constant frames;
- pops: frame-to-frame luminance or structure outliers;
- accidental held (duplicate) frames;
- at declared continuity joins, the difference across the cut. The joins are A 4879→4880 (heart into lantern), B 1519→1520 (reveal into vigil), B 3839→3840 (vigil into crane) and A 3679→3680 (H1 into reveal). Make the join list data-driven so more can be added.

Output JSON plus a short text summary.

### 4. If there's time: an A cloud-sea relief study

A's shared cloud sea reads as a flat plane in A2, A13 and A14. RUN-A-L found the problem is geometry (it needs cumuliform relief), not shading knobs.

- On branch `codex/a-cloudsea-relief`, write a geometry-only change behind a per-shot opt-in flag, default off.
- Render before/after at A2 380 and 500, and A13 3720 and 3799 (960×402).
- Enable it for no shot; the director reviews first.

Report through PR descriptions, which the director reads.
