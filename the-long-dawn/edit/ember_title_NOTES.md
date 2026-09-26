# TITLE dept (ember title) — NOTES

## >>> HANDOFF (2026-09-26): M6 in progress, resume here <<<

### 1. M6 status
Asked (critic_tone.md §M6 + director): cut the out-of-focus gathering embers by ~70%, make the
sparks rise visibly FROM the answering fires (HILLS v2 fires in `renders/hills_v2/`), keep the
letters clear of the beacon flame (it cut "G D" at ~2846-2866), keep the 2880 chord sync and the
crumble; re-render; report with a BEFORE/AFTER sheet.

**Done (code, all in `edit/ember_title.py` + `edit/ember_title_fires.py`):**
- Fire extractor re-run on HILLS v2: YES. `ember_title_fires.py` now reads `coda.Coda().wave/.old`
  (+ the cairn), 22 fires = 14 answering + 7 festival + cairn; world positions match
  `renders/hills_v2/coda_fires.json` (verified by projecting them onto hills_v2 frames: exact).
  `HILLS_HASH = 15f3bdb2f0c29398`. The title CLI re-extracts automatically if shots/hills code
  changes (`sync_fires()`, then re-execs).
- Title raised to **y = 312** (was 402); landing schedule `TA0, TA1 = 2853, 2870`. Measured on
  hills_v2: flame (luma > 0.6) top 2850:329, 2852:341, 2854:359, 2856:384, 2858:429, 2860:479. The
  band is 273-341. "G D" land from about 2856, and at 2850-2870 no letter overlaps the flame (checked in crops).
- Gathering embers: 750 defocused (+440 free + 14 bokeh) became 225 letter sparks (about 1/3 mildly
  soft, CoC <= 4) + 150 free sparks, with no bokeh. That is about 69% fewer. 184 of the letter sparks
  are thrown by a real fire (21 of them by the child's cairn); 41 find no free fire slot and are
  dropped (never shown). Each spark is integrated FORWARD from its fire's flame (launch burst, spray
  kick, curl noise, wind, crane coupling fading after launch), aimed (3 passes) to arrive 28-110 px
  below its target at TG, then glides home. The glyph's reveal field is now driven by the embers that
  land (`V_IGNITE` 2.6 px/frame): each ember ignites its stroke.
- `plate()` reads `renders/hills_v2` first. `--out DIR` CLI option.

**Rendered:** `renders/title_m6/t_02800..02966.exr`: complete, 167 frames, 69.5 s, from the
current code. **NOT swapped in.** `renders/title/` still holds the delivered, reviewed v3 (y=402,
old fires). Note: `layer()/composite()/review()` read `renders/title/`.

**Reviewed so far (M6):** gathering 2818-2866 (sparks visibly leave the ridgeline fires as thin
plumes; glitter gone), flame clearance 2850-2870, fire close-ups 2814-2838.
**NOT yet reviewed:** the hold at y=312 (2880-2925: moon, orbital ring, ridge), the release
2923-2961 at the new height, the whole-sequence sheet, the before/after sheet.

### 2. Next steps (commands)
    source ~/.venvs/longdawn/env.sh; cd ~/mishamisha/the-long-dawn; export NUMBA_NUM_THREADS=2
    mv renders/title renders/title_v3 && mv renders/title_m6 renders/title    # swap; keep v3 = BEFORE
    python edit/ember_title.py --review          # -> ~/mishamisha/_local_logs/review/title.jpg ; Read it
1. Review the hold (2880, 2895, 2910, 2925) and the crumble (2930, 2938, 2946, 2954) at 1x crops.
   Tune if needed (see §3), re-render with `python edit/ember_title.py` (~70 s, into renders/title).
2. BEFORE/AFTER sheet `~/mishamisha/_local_logs/review/title_before_after.jpg`: for frames
   2818 2834 2846 2856 2866 2880 2910 2944, composite `renders/title_v3/t_%05d.exr` (left) and
   `renders/title/t_%05d.exr` (right) over `ember_title.plate(f)` with the edit fade:
   `bg = look.linear_to_srgb(look.srgb_to_linear(plate(f)) * edit_fade(f))`, then
   `look.linear_to_srgb(soft_clip(look.srgb_to_linear(bg) + exr_rgb))` (EXR is BGR: flip).
3. Brief report to the director: title at y=312; sparks from 21 real fires (22 incl. festival/cairn);
   ~69% fewer gathering embers, no bokeh; letters clear of the flame; chord/crumble unchanged; sheet paths.

### 3. Files, functions, parameters
- `edit/ember_title.py`
  - constants: `TITLE_Y 312`, `TA0/TA1 2853/2870`, `A_SPREAD 16`, `REL0/REL1 2927/2938`, `R_SPREAD 8`,
    `END 2961`, `BIRTH_MIN 2807`, `FIRE_GAP 1.0` (title sparks; free sparks use 2.0), `V_IGNITE 2.6`,
    `SHUT 0.6`; curl flow `OCT_L/OCT_V/OCT_T`, `ADV`.
  - `Title(n_letter=225, n_release=2200, n_free=150, n_bokeh=0, seed=2880)`.
  - `_letter_targets` (farthest-point targets + landing frames), `_init_particles` (fire launches,
    aim loop, glide `glide_ease`, release, free sparks), `_fire_table`, `_launch_fire`,
    `_spark_origin`, `_release_params` (loosening kick/lift), `_free_births`, `particles(f)` (energy,
    colour, CoC per group), `glyph(f)` (mask reveal/dissolve, stipple, heat, flash, crumble), `halo`.
  - `render(f)`, `layer(f)`, `composite(img, f)`, `soft_clip`, `review()`, `plate(f)`, `edit_fade(f)`,
    `sync_fires()`, `render_range(a, b, workers, out_dir)`, `main()`.
- `edit/ember_title_fires.py`: `FIRES` rows (x, y, z, f_ign_v2, size, kind 0/1/2), `CREST_*`,
  `HILLS_HASH`, `hills_hash()`, `_extract()`, `write_in_place()`, CLI `--write`.
- Backup of the pre-M6 code: none in the repo (v3 behaviour = renders/title EXRs).

### 4. Constraints that must not move
- v2 frames 2800-2966; `layer(f)` is None outside; the layer is exactly 0 through ~2806 and from 2961.
- Fully formed by ~2880 (the final chord); crumble ~2923-2945; dark by 2961 (picture fades
  2915-2965).
- Letterforms: exactly `titles.render_line('THE LONG DAWN', titles.CINZEL, 92, 500, 0.28)`
  (read-only reuse), centred in x; the M6 position is centre y=312.
- Compositing (director's `edit/assemble.py`; not edited by TITLE): after `picture(f)`, before
  `titles.composite`/`finish`: `img = look.linear_to_srgb(soft_clip(look.srgb_to_linear(img) +
  ember_title.layer(f)))` (knee 0.8), i.e. `ember_title.composite(img, f)`; `_title()` dropped from
  `titles.story_lines`.
- Deterministic; at most 2 processes; don't edit assemble.py/titles.py or other departments' folders;
  no git writes; the CLI re-enters via `import ember_title` (numba cache is per module name).

### 5. Taste bar
- Fire is the protagonist: every visible spark comes out of a real fire; no free glitter, no bokeh
  band, no firework fountains, nothing over the beacon's flame.
- Restraint and physics: sparse sparks on curl-noise air, buoyant rise, eased settle. Letters must
  not read before their embers land; never the "particles form text" preset look.
- The formed title is crisp, exact Cinzel in warm gold (not pale/white), fine ember stipple, soft halo.
- The crumble reads as letters loosening into cooling embers (gold to ember red), dispersed by ~2952.
- Judge at 1x, in sequence (every 2-6 frames), over the real plate with the edit's fade.

---

## v3 delivery (2026-09-26, superseded by M6 once swapped)
167 frames `renders/title/t_02800..02966.exr` (half-float linear RGB, additive, v2 numbering),
title at y=402, sparks from the v1 fires. Review sheet `~/mishamisha/_local_logs/review/title.jpg`.

Re-render: `python edit/ember_title.py` (2 workers, ~70 s, < 400 MB each); `--range a b`,
`--workers n`, `--out DIR`, `--review`. Deterministic (seeded); each frame is a pure function of f.
`ember_title` sets `OPENCV_IO_ENABLE_OPENEXR=1` on import (set it in the environment if cv2 reads an
EXR before the import).
