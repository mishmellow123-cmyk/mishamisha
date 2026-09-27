# FINISH (lane FINISH): the films' photographic finish

## >>> STATE AT HANDOFF (FINISH, 27 Sep ~19:50Z; new lane, first agent) <<<

- **Brief (director, ~19:10Z):** the finish = tone curve, film print emulation, grain, halation, with spektrafilm
  (github.com/andreavolpato/spektrafilm, v0.3.3, GPLv3 code, CC BY-SA 4.0 LUTs). (1) design the insertion point;
  (2) before/after pairs + contact sheets of six hero frames per film in `_local_logs/review/finish/`, 2-3 stock/print
  pairs at restrained strength; fires ORANGE-GOLD, never yellow; a 48-frame flicker check; (3) after the director
  approves ONE look: integrate it into EDIT's delivery chain (edit/deliver.sh, assemble.py) as the final per-frame
  stage for all masters, coordinated with EDIT-2 via a "FINISH" block in edit/NOTES_v3.md; (4) the credit line in
  edit/CREDITS.md.
- **Done so far:** spektrafilm v0.3.3 in its own venv `~/.venvs/finish` (Python 3.13; NOT the shared longdawn venv);
  3-LUT bundles for Vision3 500T / 250D / 50D x 2383 / 2393 baked at 65^3 (`finish/bake_luts.sh`, ~35 s each; cubes
  git-ignored); `finish/filmfinish.py` (the runtime: numpy + cv2 + numba, runs in the edit's venv, never imports
  spektrafilm); `finish/lookdev.py` (hero before/after pairs, contact sheets, flicker check).
- **Waiting on:** the director's pick of ONE look from the sheets; then the integration (step 3).

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
     dye-cloud blur 0.65 px, 70% of the field shared by the three layers) -> L3 (print + scan to sRGB).
  3. **Print grey balance:** per-channel 1-D curves that make the neutral scale neutral (2383's base prints warm and
     its toe cool) and let the print's white (0.88 as baked) reach display white above mid-grey. Mid-grey unchanged.
  4. **Fire rule (orange-gold):** where the source is warm, the result's hue may never move more than 2 deg toward
     yellow of the source; `_fire` looks also turn it 40% back toward the scene's own emission hue (the linear ratio
     before the Hill curve), never redder than 22 deg.
  5. **Restraint:** the source's film-base black (the 0.004 lift) is kept, and the film result is blended with the
     source in linear light (mix 0.75 in the first sheets).
- **Rejected for today: (b) linear EXR out of look.py for future renders.** Almost every final is already rendered or
  rendering to 8-bit display frames; switching only future renders would give two looks in one film, and EXR at
  1920x804 is ~9 MB a frame (disk and farm transport both say no). A film curve inside look.py for new renders has the
  same two-looks problem. If a re-render wave ever happens, `look.finish(..., curve='linear')` + EXR would let FINISH
  skip the inversion; nothing else would change.
- **Cost:** ~2-4 s a frame on the loaded Mac in the look-dev path (numpy); the delivery version will fuse the per-pixel
  steps into numba kernels (target well under 0.5 s a frame) before it goes into deliver.py.

## Look-dev (step 2)

- Heroes (`finish/lookdev.py` HEROES), taken from the edit itself (`assemble.Ctx.picture`, clean, full res):
  A: embers_A3 f1160, f1700, f1400 (towers); H1 re-key (h1_v3h5 src 1480, A f3604); KARST v3 (A f3830); DESERT v3
  (A f3890). B: dusk_B f120, f330, f600; B's H1 crop (strike 3, catch) and roar crop (B f1178, f1316, f1364).
  C: X1 letters test (f700); map_C f1960, f2060; dawn_C f2450, f2600; H1 in C's parchment grade (f3044).
- Looks: `500T_2383` (fire guard only), `500T_2383_fire`, `250D_2383_fire`, `500T_2393_fire`; C's ink pages:
  `ink_grain` (grain only) vs `500T_2383`.
- Outputs: `_local_logs/review/finish/sheet_{A,B,C}.jpg`, `pairs/<hero>_<look>.jpg` (full-res crops, BEFORE | AFTER),
  `stills/`, `flicker_<cut>_<f0>.txt/png`.
