# FINISH (lane FINISH): the films' photographic finish

## >>> STATE AT HANDOFF (FINISH, 27 Sep ~21:00Z): the brief is DONE; the finish is LIVE in the masters <<<

- **The look (director, ~20:35Z): `250D_2383_fire`, blend 0.75, grain 0.5, halation as on the sheets, C's ink grain
  only.** 250D over 500T because the rose is load-bearing across the trilogy (B's alpenglow and Belt of Venus, A's
  "east pales to rose", C's illumination wash) and 500T cooled it toward lavender; 250D's finer grain also holds up
  better in the dark night fields. `finish/stage.py` LOOK.
- **Live in the delivery chain (commit f855ea5):** `edit/assemble.py` `_init(..., finish)` + `_finishing()`;
  `edit/deliver.py` master profile `finish=True`, finished segment keys, the budgeted backlog, the QC's finish line.
  All applied by `finish/wire_edit.py` (idempotent, anchored; re-run it if EDIT ever rewrites those functions). The
  FINISH block in `edit/NOTES_v3.md` tells EDIT-2 (commit 74ca341).
- **Sanity check before wiring (director's condition): PASSED.** A finished B master through the wired chain: EDIT's
  QC all PASS but the known slate WARN (lengths exact, bt709 1920x804, no black, 0 flashes, -1.30 dBTP, -16.1 LUFS);
  `_local_logs/review/finish/qc/B_finished_master_QC.txt`, `B_master_before_after_finish.jpg`. Darkest night skies
  through the CRF 14 encode: no banding (`qc/banding_darkest_skies.{jpg,txt}`; `finish/banding.py`).
- **Cost now:** ~5,700 rendered frames (A 2,320, B 952, C 2,453), ~76 CPU-min (~25 min on the 3 pool workers) on the
  next master build, which is a cold rebuild anyway after EDIT's 16834cf; after that each new render is finished as
  its segment is encoded. `FINISH_BUDGET` only matters when a look/code change leaves unfinished twins cached.
- **Open / for whoever picks this up:** (1) the H9 kit's stills are unfinished (h9_kit.py builds its own Ctx; one
  line gives it the finish, EDIT-2's call); (2) a look change (any edit to filmfinish/filmfast/stage or the LUTs)
  re-keys every finished segment and, since no unfinished twins exist by then, re-encodes them all in one run: batch
  look tweaks; (3) C's ink finals (`runC_*`) are not on the Mac yet: their grain-only path was tested on RUN-C's farm
  tests only; (4) our halation is ~3x spektrafilm's Vision3 rem-jet preset (kept: approved by eye).
- **Venvs:** spektrafilm lives ONLY in `~/.venvs/finish` (py3.13) for the LUT bake; the edit stage runs in
  `~/.venvs/longdawn` (numpy/cv2/numba) and never imports spektrafilm. LUTs: `finish/luts/` (git-ignored; re-bake
  with `finish/bake_luts.sh`, ~35 s a stock; the farm is not needed for any of this).

## Look-dev results (27 Sep ~20:10Z)

- **Flicker (48 frames, A 1600-1647, embers_A3 moving, `500T_2383_fire` through the delivery path):** the finish adds
  at most 0.16% frame-to-frame mean-luminance change (mean 0.05%) on top of the source's own (up to 4.8%); out/src
  ratio std 0.003. No flicker. Grain renews every frame (high-pass frame-to-frame correlation 0.48 = the source's
  own texture; grain alone would be 0).
- **Fire hue** (`fire_hue.txt`, mean hue of warm bright pixels): the `_fire` looks hold every fire at or below the
  source's hue (0.3-3 deg MORE orange, saturation +0.05-0.10); plain `500T_2383` drifts up to +2.4 deg yellow (the
  per-pixel 2-deg cap with a soft edge), so it is not recommended.
- **Tone:** mid-grey unchanged; the print's toe deepens 0.08-0.2 (e.g. 0.10 -> 0.084); the renders' film-base black
  (display 0.052) is held (-> 0.054); anything an edit grade or fade took below it is left untouched.
- **Grain** at 0.6 of spektrafilm's physical default (13 um a pixel): 2-3 codes RMS in 8-bit mid-tones; 85% of the
  field shared by the three layers (the first try, 70% shared at 1.0, read as coloured video noise on the skies).
- **Known colour cost:** the film cools B's dusk mauve toward lavender (sky R-G 0.072 -> 0.042 with 500T, 0.056
  with 250D): the reason the director picked 250D.
- **Halation strength:** ours is spektrafilm's schema default (red 0.05, green 0.015 at 65 um), about 3x its
  Vision3 'cine, strong anti-halation' preset (0.015 / 0.005 at 50 um: rem-jet all but kills it). Kept, because it
  is what the director approved on the sheets; drop to the preset if a red rim ever reads as a glow fringe.
- **Bitrate:** at the master's CRF 14 the finished embers shot came out SMALLER (16.2 vs 17.7 Mb/s): the print toe
  swallows the renders' dither noise; the grain at 0.5 costs less than it saves.
- **C:** MAP-L's book and map (`book_C`, `map_C`) and C's H1 are Hill-curved: full finish. RUN-C's ink (`runC_*`)
  is plain sRGB: grain only (`ink_grain`); running the film on it darkens and oranges the parchment (C5/C6 show it).

## Findings that change the brief

- **v3 masters have NO grain today.** The looping 24-frame grain bank lives only in `edit/assemble_v1.py` and
  `edit/assemble_v2.py` (the old films); v3's `edit/assemble.py` + `deliver.py` add none. So FINISH adds grain rather
  than replacing a bank.
- **spektrafilm's grain is FROZEN by default, not unseeded.** In v0.3.3 `apply_grain_to_density_layers(...)` is called
  without `fixed_seed`, which sets `seed=[0,1,2]` and `np.random.seed(0/1/2 + 10*sublayer)` inside every call: the
  same grain pattern on every frame (a "dirty window"). `np.random.seed(frame_idx)` before a frame does not help.
  Our runtime never calls spektrafilm's grain: it draws its own from spektrafilm's particle statistics with a numpy
  PCG64 generator seeded per (cut, frame) (deterministic, distinct per frame, identical across workers and re-runs).
- **auto_exposure** is irrelevant to the baked path: the LUTs are exposure-fixed; our EV is one fixed calibration per
  look (scene 0.18 prints at the same display value the Hill curve gave it: -0.49 EV for 500T/2383). Nothing is
  measured per frame, so nothing can pump.
- **The LUT CLI has no linear `acescg` input in v0.3.3**: we feed ACEScct (the log encoding of the same AP1 linear).
- **Dependencies on this Mac (macOS 12):** pyfftw, lensfunpy, rawpy, exiv2 have no wheels; the runtime needs none of
  them (RAW loading only; pyfftw is shimmed onto scipy.fft). opencv 5 for py3.13 would build from source: pinned to
  the 4.10.0.84 abi3 wheel.

## Design: the insertion point (step 1)

- **Chosen: (a) a finishing pass on the frames as they are**, inserted in the edit as the last per-frame picture stage
  (after the take/crop/per-cut grade/book matte, before titles and burn-ins), so every shot of every master gets the
  same finish and no approved shot is re-rendered.
  1. **Exact inverse of look.finish's tone path** (`display_to_709`): sRGB decode -> remove the 0.004 lift -> inverse
     output matrix -> the Hill curve's own inverse (a closed-form quadratic root per channel) -> inverse input matrix.
     Round trip on 200k random colours: median error 0.02 stop where no channel clipped. Bloom and vignette stay
     baked in (they were applied before the curve, in linear light, so they invert correctly). Clipped codes (a
     channel at 255) invert to "at least 25.7x" (capped at 32): the print's shoulder is flat there, so the cap does
     not show; their hue cannot be recovered and is handled by the fire rule below.
  2. **Film:** ACEScg -> ACEScct -> L1 (film log exposure) -> **halation** in linear exposure (spektrafilm's model:
     red 0.05 / green 0.015 / blue 0, 65 um first bounce at 13 um per pixel, 3 bounces decaying 0.5, energy
     conserving) -> L2 (negative density) -> **grain** in density (spektrafilm's particle statistics, Gaussian limit,
     dye-cloud blur 0.65 px, 85% of the field shared by the three layers, 0.6 of the physical amount) -> L3
     (print + scan to sRGB).
  3. **Print grey balance:** per-channel 1-D curves that make the neutral scale neutral (2383's base prints warm and
     its toe cool) and let the print's white (0.88 as baked) reach display white above mid-grey. Mid-grey unchanged.
  4. **Fire rule (orange-gold), on the FINAL colour:** where the source is warm, the result's hue may never move more
     than 2 deg toward yellow of the source; `_fire` looks also turn it 40% back toward the scene's own emission hue (the linear ratio
     before the Hill curve), never redder than 22 deg.
  5. **Restraint:** the source's film-base black (the 0.004 lift) is kept, the film result is blended with the
     source in linear light (mix 0.75), and anything below the renders' floor (an edit grade, a fade) is untouched.
- **Rejected for today: (b) linear EXR out of look.py for future renders.** Almost every final is already rendered or
  rendering to 8-bit display frames; switching only future renders would give two looks in one film, and EXR at
  1920x804 is ~9 MB a frame (disk and farm transport both say no). A film curve inside look.py for new renders has the
  same two-looks problem. If a re-render wave ever happens, `look.finish(..., curve='linear')` + EXR would let FINISH
  skip the inversion; nothing else would change.
- **Cost:** reference path (numpy) 2-4 s a frame; delivery path (`filmfast.py`, two numba kernels) ~0.85 CPU-s a
  frame at 1920x804 on the loaded Mac, so a cold rebuild of all four masters is ~4.5 CPU-hours (~1.5 h on 3 slots),
  once; after that the chain stays incremental. The farm cannot help: the frames live on this Mac, not in git.

## Look-dev (step 2)

- Heroes (`finish/lookdev.py` HEROES), taken from the edit itself (`assemble.Ctx.picture`, clean, full res):
  A: embers_A3 f1160, f1700, f1400 (towers); H1 re-key (h1_v3h5 src 1480, A f3604); KARST v3 (A f3830); DESERT v3
  (A f3890). B: dusk_B f120, f330, f600; B's H1 crop (strike 3, catch) and roar crop (B f1178, f1316, f1364).
  C: MAP-L's v3 book_C farm tests f175, f728 (the letters kindle), f6600 (the Havens page); H1-C's catch in the
  parchment grade (C f3316); RUN-C's ink farm tests reveal_h f114, illum_d f2480 (C's finals are not on the Mac).
- Looks: `500T_2383` (fire guard only), `500T_2383_fire`, `250D_2383_fire`, `500T_2393_fire`; C's ink pages:
  `ink_grain` (grain only) vs `500T_2383`.
- Outputs: `_local_logs/review/finish/sheet_{A,B,C}.jpg`, `pairs/<hero>_<look>.jpg` (full-res crops, BEFORE | AFTER),
  `stills/`, `flicker_<cut>_<f0>.txt/png`.
