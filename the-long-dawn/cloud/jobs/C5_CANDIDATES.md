# C5 candidate farm jobs

Prepared for owner review. Nothing was launched, no nodes were created, and no accepted delivery or EDIT selection was changed. These jobs produce review candidates at 1920×804, with absolute cut-frame names. The eight C jobs cover their complete requested ranges. The A crossing job contains conservative full-shot coverage and remains blocked because the requested minimal changed interval and unchanged join controls have not been established.

**All nine jobs have a launch blocker:** their requested `branch` is `claude/owner-night-20260929`, but the current farm ignores that field. `farm_node.py:48` sets `BRANCH = 'claude/long-dawn-v2'`; `git_update()` at `farm_node.py:95–103` fetches that constant. The bootstrap in `farm.py` also names `claude/long-dawn-v2`. The owner must fix and verify branch handling, or integrate the identical reviewed code into the branch the farm actually fetches, before launching. These JSON files do not repair shared farm infrastructure. A successful dry-run cannot establish which source a node will fetch.

## Jobs, frames and units

Every JSON has one source render lane. At `--nodes 3` on CPU nodes with `cpu_limit=8`, the current planner splits that lane into three units and then eight single-thread render processes per unit. In the table, `1 → 8/unit` describes this rewrite; it does not mean the job author supplied eight commands. Unit sizes are ordered by their original unit index, before the planner sorts larger units first. Counts below are distinct cut frames; RGB and matte are two output files for each page frame.

`P + E N` means the two setup commands described below: prepare the kind, then prove native equality at frame N. Output paths start with `renders/`; page jobs also declare the sibling `_matte` output explicitly. The estimate column is seconds per original study row multiplied by four, with denominator details in the cost section.

| Job JSON | Inclusive frames; count | `out_dir` after `renders/` | Source → planned lanes | Three units: frame counts | Setup | Blockers / unresolved checks | 960 row ×4 estimate |
|---|---|---|---|---|---|---|---|
| [cand_reveal_night-fire.json](cand_reveal_night-fire.json) | C2880–3119; 240 | `cand_reveal_night-fire` | 1 → 8/unit | 80 / 80 / 80 | P + E 3040 | Branch; native proof and delivery review pending | 46.324084 s / three-variant batch |
| [cand_watch_night-fire.json](cand_watch_night-fire.json) | C4480–4719; 240 | `cand_watch_night-fire` | 1 → 8/unit | 80 / 80 / 80 | P + E 4599 | Branch; native proof and delivery review pending | 43.627864 s / three-variant batch |
| [cand_trap_front_smoke_near.json](cand_trap_front_smoke_near.json) | C2320–2639; 320 | `cand_trap_front_smoke_near` | 1 → 8/unit | 107 / 106 / 107 | P + E 2479 | Branch; native proof and delivery review pending | 1.904140 s / candidate frame |
| [cand_map_beacon-falloff.json](cand_map_beacon-falloff.json) | C3440–3839; 400 | `cand_map_beacon-falloff` | 1 → 8/unit | 133 / 134 / 133 | P + E 3724 | Branch; rebuilt cache identity with frozen arrays unverified; native proof pending | 1.486720 s / three-variant batch |
| [cand_deep_leaned_ladders.json](cand_deep_leaned_ladders.json) | C4240–4479; 240 | `cand_deep_leaned_ladders` + `_matte` | 1 → 8/unit | 80 / 80 / 80 | P + E 4360 | Branch; native proof and RGB/matte review pending | 4.059628 s / candidate frame |
| [cand_cold_lead24.json](cand_cold_lead24.json) | C3816–3999; 184 | `cand_cold_lead24` | 1 → 8/unit | 61 / 62 / 61 | P + E 3840 | Branch; native proof and Map/Cold join review pending | 1.973748 s / candidate frame |
| [cand_pen_soft_spine_metal.json](cand_pen_soft_spine_metal.json) | C5440–5679; 240 | `cand_pen_soft_spine_metal` + `_matte` | 1 → 8/unit | 80 / 80 / 80 | P + E 5560 | Branch; native proof and RGB/matte review pending | 5.599652 s / candidate frame |
| [cand_t1_current-words.json](cand_t1_current-words.json) | C320–559; 240 | `cand_t1_current-words` + `_matte` | 1 → 8/unit | 80 / 80 / 80 | P + E 430 | Branch; current CRC32 revision unmeasured; native proof, memory and text review pending | 14.526872 s / legacy-seed frame only |
| [cand_crossing_both.json](cand_crossing_both.json) | A4880–5839; 960 | `cand_crossing_both` | 1 → 8/unit | 320 / 320 / 320 | P + E 5584 | **Branch and unresolved minimal range/unchanged controls**; no completed study image or equality result | null; no completed timing |

The unit calculation follows `farm.py:483–485` (`split_even`, rounded boundaries), `farm.py:550–585` (unit and process splitting), and `farm.py:1680–1689` (request allocation). It applies to **each job submitted individually** with `--nodes 3`. Passing all nine jobs to one invocation shares the request's unit budget and gives a different plan. The source prefix declares demand of one CPU; at CPU limit eight the planner therefore creates eight items per unit, with concurrency eight. No multiplication of the old RSS measurements establishes safe native concurrency: the whole-process measurements came from another stack and sometimes included multiple variants. The owner must provide adequate memory for the actual planned native processes.

## Wrapper and setup contract

The wrapper is `cloud/cand_render.py`. Every setup and render command starts with this exact environment prefix:

```sh
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
```

Each job has exactly these two sequential setup commands, with its kind and probe substituted:

```sh
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 cloud/cand_render.py KIND --prepare
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 cloud/cand_render.py KIND --equal FRAME --scale 1
```

`--prepare` checks the required interpreter/package metadata and assets. For Map it also invokes the source-pinned world/atlas generator on the farm. The equality command renders through the direct original and omitted-default APIs; Reveal, Watch and Map also compare the shared accepted route. Page equality covers HDR, alpha and finished RGB. An inequality or failed setup exits nonzero before rendering. The node caches successful setup by the setup-command hash and fetched commit (`farm_node.py:228–243`); a reused success is scoped to that pair. This compares implementations at a sampled frame on the node; it does not compare against previously delivered JPEGs or establish that the candidate is visually approved.

The required stack is Python 3.12, NumPy 2.5.3, Numba 0.67.0, llvmlite 0.49.0, SciPy 1.18.1, opencv-python-headless 4.10.0.84, Pillow 12.3.0 and fonttools 4.66.0. These match the designated `~/ldfarm/venv` environment and the farm bootstrap package pins. Setup performs no `pip install`, upgrade or fallback. A mismatch must be resolved in the farm environment before the equality check can proceed. The old study stack, recorded in [ADOPTION.md](../../ADOPTION.md) and [Map runtime receipt](../../review/c-v5-last-beacon/runtime.json), used Python 3.13.9, NumPy 2.3.5 and Numba 0.62.1; its timing and pixel evidence do not verify the new stack.

The render command is the same prefix followed by:

```sh
python3 cloud/cand_render.py KIND OPTION --range FIRST-LAST --out renders/cand_KIND_OPTION
```

`--range A-B` is inclusive and matches `farm.py`'s rewritable frame syntax. Each process emits atomic `f_XXXXX.png` files. The farm verifies dimensions and decodability, then ships JPEG quality 95 with 4:4:4 sampling (`farm_node.py:156–210`). Page RGB and full-resolution matte are emitted by one renderer process; the JSON declares both outputs, so the collector tracks both. Mattes also land as JPEG quality 95 with 4:4:4 sampling. T1 uses `with_fire=False` and `no_burn=True`. Page finish is applied once with the same arguments as `ld_candidate`. EDIT adoption, caption changes and delivery selection remain separate decisions.

## Input audit and reconstruction

This audit inspected source and file metadata within the worktree, plus the read-only renders link. It did not access the frozen Map worktree or modify any render cache. The files below were checked against this checkout; a node must possess the corresponding tracked content before its setup runs.

| Kind | Inputs and call path | Reconstruction / availability |
|---|---|---|
| Reveal / Watch | `shots/run/summits.npy`, read at `reveal_pair_c_v5.py:43` and `render_ink.py:134`; procedural terrain and fire | Catalogue is tracked and present. No untracked fire file was found on these selected paths. Their frames use `render_aov` directly rather than a pre-rendered AOV cache. |
| Trap / Cold | `assets/ring/inscription_outer.png` and `inscription_inner.png`, loaded by `ringsolid.py:77`; tower geometry via `scene_b.py:922–940` | Both inscription images are tracked and present. The missing tower pickle is generated from tracked `towers2.py` when setup equality constructs the scene. The source-derived cache path is `renders/embers_A3/cache/towers_0b1d0ada982294a7.pkl`. |
| Deep / Pen | Procedural page strokes, paper textures and geometry through `book_c_v5.py`, `book_c.py`, `redbook.py`, `pages.py` and `pen.py` | No untracked input file is used by the selected shot methods. These paths do not open T1's font, filmed burn footage or the later book plate. |
| T1 | `assets/fonts/EBGaramond-Italic.ttf`, from `inkline.py:18,39` and `book_c_t1_candidates.py:57` | Tracked and present. The supplied flags exclude filmed burns; no footage restoration is needed for C320–559. |
| Map | Source-pinned `terra_5.npz`, six atlas levels, relief and manifest | Missing through the local renders link. Farm setup rebuilds them from the committed recipe, as detailed below. |
| Crossing | `shots/run/crossing_fires.npy` and the pinned `shots/run/crossing.py` baseline | Both tracked and present; measured hashes below. `crossing_candidates.load_renderer():150–165` rejects a missing fire cache or changed baseline. No regeneration or replacement is allowed by this job. |

The tower cache key hashes `towers2.py` and the switches `(giants, alt)` (`scene_b.py:927`). C3 selects `giants=(), alt=False` (`variant.py:26–31`), yielding the key above. The cache was absent at audit time. The generator has seeded geometry constructors; equality setup creates it on a writable farm checkout. Locally, calling the real constructor without isolating this cache would write through the owner's renders symlink, so that is prohibited. Numba's compilation caches are generated runtime artifacts rather than missing artistic inputs.

`LONGDAWN_SYSFONTS` is declared at `shots/embers/glyphs.py:18` and supports the multilingual glyph generator's Noto files. Trap/Cold import that module transitively (`c3.py:802 → timeline.py:8 → scene_a.py:9`), which requires the `fontTools` package at import time, but their `c_v5.Scene` builds its own Forges/smoke (`c_v5.py:172–180`) and never invokes the lazy Timeline glyph path (`timeline.py:29`). Consequently these nine jobs do not require a Noto installation or a `LONGDAWN_SYSFONTS` override. The sparse-excluded, tracked `assets/data` and `assets/glyphs` files are also outside these selected render paths. An unrelated glyph or globe job would need its own input audit.

### Map source identity

`last_beacon.source_manifest():102–108` hashes `bake.py`, `geo.py`, `terra.py`, `features.py`, `sheet.py`, `ink.py`, `pen.py`, `noise.py` and `../../lib/look.py`. Every current hash matched the tracked `atlas` record in [runtime.json](../../review/c-v5-last-beacon/runtime.json). Standard-library digest calculations gave these required paths:

```text
renders/map_last_beacon_C/world/c7ef80d8796f25a1/terra_5.npz
renders/map_last_beacon_C/atlas/eeef0dd857771b36/manifest.json
renders/map_last_beacon_C/atlas/eeef0dd857771b36/L0.npy … L5.npy
renders/map_last_beacon_C/atlas/eeef0dd857771b36/relief.npy
```

The atlas specification in that matched manifest is `x0=-46.0`, `y1=42.0`, width 4800, height 2688, 48 pixels per map degree. `validate_atlas():131–147` checks manifest equality, level dimensions/dtypes and relief dimensions/dtype. It does not check generated pixel hashes. The world guard requires the expected file before render construction; it does not attest to equality with the archived world's arrays.

The supported reconstruction command is `python3 shots/map/last_beacon.py bake` (`last_beacon.py:352–363`); `--prepare` calls the same builder. `build_atlas():157–213` creates the source-pinned world directory, builds world/features/glyph geometry, then writes the atlas serially in tiles. Supporting world caches are generated under the source-pinned world directory and are not separate downloads. `geo.py:3–6` describes the invented geography; this path reads no real-world geographic asset.

Code inspection found fixed RNG seeds (`terra.py:312`; `features.py:334,338,557,593,628`; `sheet.py:124–133`), deterministic integer noise hashing (`noise.py:10–14`), and atlas grain seeded by tile coordinates (`last_beacon.py:189`). The set at `terra.py:544` is used for membership; the river traversal follows ordered arrays and lists. No unseeded generator, source of external entropy, or output-affecting iteration over an unordered set was found. This supports reconstruction from the tracked recipe. **Exact array identity with the frozen Mac cache has not been measured**, especially on the different farm library stack. Equality on a rebuilt cache establishes consistency of original, default and shared accepted entrypoints using that cache; historical cache parity remains unverified. To attest to historical parity, obtain the frozen arrays through an owner-approved transfer and compare decoded array hashes in the farm environment; do not relabel a source digest as a content digest.

### Measured tracked-file hashes

These SHA-256 values were calculated from this worktree without importing renderers:

| Path relative to `the-long-dawn/` | SHA-256 |
|---|---|
| `shots/run/summits.npy` | `833f158f4207a6124242d0647bbf7f4585a797705a5ad2ae3524f9c5fed419a5` |
| `assets/fonts/EBGaramond-Italic.ttf` | `bba2c4499c93c9612b90b9825d32b07da52fce2fe57562a1eb6b833553f93c4e` |
| `assets/ring/inscription_outer.png` | `6ea4ecfd207aa485fed0fdde56300b4022bc53d8129efc4bb393d8c6e7cd9a71` |
| `assets/ring/inscription_inner.png` | `c6868a1ed33984463c14177433ad84ef37d3af79dddc74f3d87a30f6f7094ab5` |
| `shots/run/crossing.py` | `67c2ee812fd4f6e30436c7ec53dbcc1b21806ee70a0b18125d7045a348238ac3` |
| `shots/run/crossing_fires.npy` | `173d886fbe8484da992cc19616115ed5bb041b3c9965bd6b9f049381c47f8479` |

## Crossing interval remains unresolved

`crossing_candidates.render_cut():194–203` installs the rope replacement and rock scene modifier for every requested frame; there is no restriction to the reported problem windows. The accepted rock is added whenever an event exists (`crossing.py:1192–1208`). The accepted rope's rasterization includes per-pixel depth and coverage checks (`crossing.py:1245–1290`), and the render invokes the rope at every frame (`crossing.py:1459`). Camera visibility, occlusion and the rope's terrain-clearance predicate therefore determine whether the changes affect a pixel. Neither the reported A5040–5063 rope window nor the A5584 rock probe supplies mathematical boundaries for the changed pixels.

The current job's A4880–5839 range conservatively contains the entire accepted crossing. **It is not a proven minimal interval and contains no frames proven unchanged as join controls.** Resolving the brief requires a resource-authorized comparison of accepted and `both` HDR/finished RGB over the shot, with the same camera, supersampling and grade, to identify changed frames; then include verified unchanged controls around each changed interval. If differences extend to a shot boundary, controls must come from the adjacent shot and need an explicit review plan. Actual boot support, contact shadows and motion also remain unproved. This session did not run the heavy comparison and does not authorize the conservative job's launch.

## Cost basis and limits

The requested native estimates apply a factor of four to the old 960×402 median seconds per recorded row in [ADOPTION.md](../../ADOPTION.md). This is a pixel-count heuristic, **not a measured native benchmark, farm throughput estimate, memory estimate or price quote**. Initialization, equality setup, Map baking, transfers, process concurrency and the changed package stack prevent interpreting these figures as job elapsed time. No native time or RSS was extrapolated beyond the explicitly labelled ×4 time column.

| Study | Inherited 960 median | Original denominator | ×4 estimate, same denominator |
|---|---:|---|---:|
| Reveal night | 11.581021 s | 25 batches; accepted + night-fire + night-wisp together | 46.324084 s / batch |
| Watch night | 10.906966 s | 27 batches; the same three variants together | 43.627864 s / batch |
| Trap | 0.476035 s | 170 candidate frames | 1.904140 s / candidate frame |
| Map | 0.371680 s | 50 batches; accepted + both territory variants together | 1.486720 s / batch |
| Deep | 1.014907 s | 26 candidate frames | 4.059628 s / candidate frame |
| Cold | 0.493437 s | 47 candidate frames | 1.973748 s / candidate frame |
| Pen | 1.399913 s | 26 candidate frames | 5.599652 s / candidate frame |
| T1 | 3.631718 s | One archived legacy-seed current-word frame; failed memory gate | 14.526872 s / legacy frame |
| Crossing | null | No completed image | null |

Night and Map share rendering work across variants within their recorded batches; dividing by three would invent single-candidate throughput. T1's figure does not cover the current CRC32 revision. These figures are inherited receipt summaries, not measurements repeated in this session. ADOPTION records T1 peak RSS 2,205,745,152 bytes and crossing peak RSS 2,960,359,424 bytes at the failed 960 probes; neither was retried locally. Map, T1 and crossing smoke renders are skipped locally under the lane's resource constraints.

## Local validation

Validation date: 2026-09-29. All execution was local and sequential. No network request, farm launch, node creation or push occurred. The accepted shot sources, `farm.py`, `farm_node.py` and `beaconrun_a_catches3.json` are unchanged. All twelve proposed output directories were absent when checked. Tests copied renderer source and tracked assets into temporary projects; their caches never used the owner's `renders` symlink.

- **Dry-runs:** all nine passed with `python3 cloud/farm.py cloud/jobs/<job>.json --nodes 3 --dry-run`, from `the-long-dawn/`. Here `python3` was Python 3.14.7. Each result explicitly printed three units and “dry run: nothing starts”; neither credentials nor network access was needed. The reported stream totals were 240 for Reveal/Watch, 320 for Trap, 400 for Map, 184 for Cold, 960 for Crossing, and 480 each for Deep/Pen/T1 (240 RGB plus 240 matte).
- **Tests:** 13 lightweight tests passed (8 adapter/planner tests and 5 existing cloud-runner tests). The final discovery run reported 17 tests with the four real-render cases skipped by default; all four were executed separately and passed as recorded below. The suite checks setup/native-proof arguments, exact planner frame coverage, page output pairing, PNG quantization and no-overwrite publication, linked-path rejection, and failing proofs for changed RGB/HDR/alpha, dtype and keys. Existing cloud runner tests are included.
- **Real array comparisons:** the table below records separate, serial test invocations of `CandidateRecipeTests.test_<kind>` with `LD_CAND_RENDER_SLOW=1`. Each compares the wrapper to `build`, `checked` and `book_output` extracted literally from ADOPTION's Python recipe. Scale is 0.05 (96×40 output); page texture resolution remains the original 110 pixels/cm. The frame arrays and dtypes must be exactly equal; these are not image-similarity scores.

| Candidate/frame | Arrays checked | Result | Measured process peak RSS, bytes |
|---|---|---|---:|
| Trap C2479 | RGB | exact | 647,970,816 |
| Cold C3816 | RGB | exact | 742,146,048 |
| Deep C4360 | HDR, alpha, RGB | exact | 981,925,888 |
| Pen C5560 | HDR, alpha, RGB | exact | 358,154,240 |

The lightweight suite and array tests use `~/ldfarm/venv/bin/python`: Python 3.12.14 and the exact package versions listed above. The historical validation venv was inspected for package metadata only; it was not used for these tests. The largest measured render-test peak was 981,925,888 bytes, below the lane's 1.5 GB cap. Each child has a 1,400,000,000-byte `ru_maxrss` watchdog and a final peak check. An initial Trap attempt stopped before rendering because the sandbox forbids `ps`; the supervisor now samples its own process with `resource.getrusage` and the rerun passed. A native call that holds Python's GIL can delay watchdog polling; the final high-water check still rejects an excessive peak. This is an observed test-process bound, not a farm memory guarantee.

**Skipped:** every native setup equality proof and all native/full-shot renders remain for the farm. No Map, T1 or Crossing smoke render was attempted; the Map bake was not run. Reveal/Watch had no local render comparison in this lane. The small frame tests establish wrapper/recipe equality only at their listed frames and scale. They do not establish historical-cache parity, native pixels, motion, contact or visual acceptance.

**Review:** an independent source audit checked wrapper routing/finish, native proof failure behavior, output publication and farm splitting. A second read checked the runbook's source claims and independently recalculated input hashes, cache keys, unit counts and the cost arithmetic. The audit corrections were incorporated; native evidence gaps and the two launch blockers remain explicit above.

**Commit note:** the lane's sandbox could not write the shared git index, so these files were committed by Claude after review (the branch-handling blocker above was fixed in the same change: farm units now fetch their job's `branch`).
