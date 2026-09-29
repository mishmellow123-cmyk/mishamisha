# Candidate adoption runbook

These are review candidates, not adopted deliveries. The commands below were prepared from the public APIs at `1d133ee`; **none of these native batches or equality commands was executed while writing this document**. All measured studies were 960×402. Native 1920×804 time and peak RSS are **not measured** for every candidate.

T1 and A crossing exceeded the authorized memory gate. Their commands are recipes for a later, explicitly authorized session with adequate resources; do not retry them during the stopped overnight session. A crossing has no completed image and real boot contact remains unproved. T1's current CRC32 grain has not been rendered. Keep accepted finals and their frozen sources intact.

## Before using the commands

Work from the repository's `the-long-dawn/` directory, in the prepared render environment. The study environment used Python 3.13.9, NumPy 2.3.5 and Numba 0.62.1, with OpenCV, SciPy and Pillow. Use an existing compatible environment; these instructions do not install or upgrade dependencies. Set `LD_PYTHON` to its interpreter if `python3` is not that environment.

Restore and validate the original render assets and caches first. The Map commands explicitly reject a missing source-pinned world or atlas. RUN needs its existing summit/fire catalogues. Book commands need the existing fonts and book dependencies. Do not substitute missing assets or launch cache/farm/cloud generation to make a command proceed.

A renderer switch generates an alternative image; **EDIT must explicitly select the new output stem before the film uses it**. The output names below are new review directories, not aliases for delivered finals. The helper refuses existing directories, writes absolute cut-frame filenames, preserves full-resolution mattes for book shots, and applies each original grade exactly once. It is a sequential recipe, not a farm supervisor or resumable production runner.

Retain the session resource gates: one owned heavy process at a time, pressure 1 or 2 at launch, at least 8 GiB free disk, a 4 GiB candidate budget, and no new render at battery ≤15%. A 960 probe must peak at ≤1.5 GiB; the owned supervisor stops its process at pressure 4 or RSS >2 GiB. Native/full jobs additionally need ≥5 GiB free+inactive memory and explicit authorization. Run the recipe under the approved supervisor; the function below does not replace it. Measure a single native frame before a full batch. A successful 960 equality check establishes neither native cost nor visual approval.

## Define the command once

This shell block **defines a function only**. Each invocation starts a fresh Python process, avoiding collisions among the renderers' generic module names. `equal` defaults to scale .5; an optional fourth argument of `1` requests native equality. `render` always uses scale 1, inclusive first/last frames, JPEG quality 95 and 4:4:4 sampling. Its output directory must be new and directly inside `renders/`.

```bash
ld_candidate() {
  NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  "${LD_PYTHON:-python3}" - "$@" <<'PY'
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time

root = Path.cwd()
if not (root / 'shots').is_dir() or not (root / 'lib').is_dir():
    raise SystemExit('Run from the-long-dawn/')
args = sys.argv[1:]
if not args or args[0] not in ('equal', 'render'):
    raise SystemExit('equal KIND FRAME [SCALE] | render KIND OPTION FIRST LAST OUT')
mode = args[0]
if (mode == 'equal' and len(args) not in (3, 4)) or (mode == 'render' and len(args) != 6):
    raise SystemExit('Wrong argument count')
kind = args[1]
ranges = {'reveal': (2880, 3119), 'watch': (4480, 4719),
          'trap': (2320, 2639), 'map': (3440, 3839),
          'deep': (4240, 4479), 'cold': (3816, 3999),
          'pen': (5440, 5679), 't1': (320, 559), 'crossing': (4880, 5839)}
options = {'reveal': 'night-fire', 'watch': 'night-fire',
           'trap': 'front_smoke_near', 'map': 'beacon-falloff',
           'deep': 'leaned_ladders', 'cold': 'lead24',
           'pen': 'soft_spine_metal', 't1': 'current-words'}
if kind not in ranges:
    raise SystemExit('Unknown kind')
scale = float(args[3]) if mode == 'equal' and len(args) == 4 else (.5 if mode == 'equal' else 1.)
if scale not in (.5, 1.):
    raise SystemExit('Scale must be .5 or 1')
if mode == 'equal':
    first = last = int(args[2]); chosen = None
    if kind == 'cold' and first < 3840:
        raise SystemExit('Cold equality must use a common accepted frame, at or after 3840')
else:
    chosen, first, last = args[2], int(args[3]), int(args[4])
    if chosen not in (('rope', 'rock', 'both') if kind == 'crossing' else (options[kind],)):
        raise SystemExit('Unsupported candidate')
if not ranges[kind][0] <= first <= last <= ranges[kind][1]:
    raise SystemExit('Frame range is outside this shot')
family = 'run' if kind in ('reveal', 'watch', 'crossing') else ('embers' if kind in ('trap', 'cold') else 'map')
sys.path.insert(0, str(root / 'shots' / family))
sys.path.append(str(root / 'lib'))
import numpy as np
import cv2
import numba
from PIL import Image

W, H = int(1920 * scale), int(804 * scale)
page_kind = kind in ('deep', 'pen', 't1')
def threads():
    cv2.setNumThreads(0)       # reset after constructors/imports that alter the pool
    numba.set_num_threads(1)
    assert cv2.getNumThreads() == 1 and numba.get_num_threads() == 1

def checked(render, f):
    threads()
    result = render(f)
    assert cv2.getNumThreads() == 1 and numba.get_num_threads() == 1
    assert result['rgb'].shape == (H, W, 3)
    assert all(np.isfinite(a).all() for a in result.values())
    return result

def book_output(module, result):
    hdr, alpha = result
    rgb = module.look.finish(hdr, exposure=1.15, bloom_strength=.06,
                             bloom_threshold=1.2, vignette_amount=.32)
    return dict(hdr=hdr, alpha=alpha, rgb=rgb)

def build(route, option=None):
    # route distinguishes direct original, omitted default, shared accepted and candidate.
    if kind in ('reveal', 'watch'):
        import beacon_night_candidates as X
        driver = X.reveal if kind == 'reveal' else X.watch
        shot = driver.make_shot()
        if route == 'original':
            call = lambda f: driver.render(f, shot, scale=scale, ss=2.)
        elif route == 'shared':
            call = lambda f: X.render_variants(f, shot, kind=kind, scale=scale,
                                               ss=2., variants=('accepted',))['accepted']
        elif route == 'default':
            call = lambda f: X.render(f, shot, kind=kind, scale=scale, ss=2.)
        else:
            call = lambda f: X.render(f, shot, kind=kind, variant=option, scale=scale, ss=2.)
        return lambda f: dict(rgb=call(f))
    if kind in ('trap', 'cold'):
        name = 'c_v5_trap_candidates' if kind == 'trap' else 'c_v5_cold_leadin'
        X = importlib.import_module(name)
        base = X.C if kind == 'trap' else X.base
        scene = base.Scene() if route == 'original' else (X.Scene() if route == 'default' else X.Scene(variant=option))
        return lambda f: dict(rgb=scene.frame(f, scale=scale))
    if kind == 'map':
        import last_beacon_candidates as X
        base = X.BASE
        manifest = base.source_manifest()
        identity = dict(world=manifest['world'], sources=manifest['sources'])
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        world = base.CACHE.parent / 'world' / key / f"terra_{manifest['world']}.npz"
        if not world.is_file():
            raise RuntimeError('Existing source-pinned Map world is required; no bake')
        base.validate_atlas(base.atlas_dir(), manifest)
        shot = base.LastBeacon() if route == 'original' else X.make_shot()
        if route == 'original':
            call = lambda f: shot.render_c(f, scale)
        elif route == 'shared':
            call = lambda f: X.render_variants(f, shot, scale=scale, variants=('accepted',))['accepted']
        elif route == 'default':
            call = lambda f: X.render(f, shot, scale=scale)
        else:
            call = lambda f: X.render(f, shot, candidate=option, scale=scale)
        return lambda f: dict(rgb=call(f))
    if page_kind:
        name = {'deep': 'book_c_v5_deep_candidates', 'pen': 'book_c_v5_pen_candidates',
                't1': 'book_c_t1_candidates'}[kind]
        X = importlib.import_module(name)
        if kind == 't1':
            BC = X.BC
            scene = (BC.Book3(W, H, with_fire=False) if route == 'original' else
                     X.make_renderer(scale=scale, with_fire=False) if route == 'default' else
                     X.make_renderer(option, scale=scale, with_fire=False))
            scene.no_burn = True
        else:
            BC = X.V.BC
            shot = 'deep_abandoned' if kind == 'deep' else 'pen'
            scene = (X.V.PagesV5(shot, W, H, 110) if route == 'original' else
                     X.make_renderer(scale=scale, ppc=110) if route == 'default' else
                     X.make_renderer(option, scale=scale, ppc=110))
        return lambda f: book_output(BC, scene.frame(f))
    import crossing_candidates as X
    import crossing_rock_candidates as rock
    cr = X.load_renderer()     # rejects a missing fire cache or changed baseline
    def call(f):
        if route == 'original':
            hdr = cr.render(f - 4880, scale=scale, ss=1.5, variant='main', trail=True)
        elif route == 'default':
            hdr = X.render_cut(f, renderer=cr, scale=scale, ss=1.5, variant='main', trail=True)
        else:
            modifier = (lambda scene, renderer, cfg: rock.apply_to_scene(
                scene, renderer, cfg, candidate='low_shoulders')) if option in ('rock', 'both') else None
            hdr = X.render_cut(f, renderer=cr, scale=scale, ss=1.5, variant='main', trail=True,
                               rope='snow_clearance' if option in ('rope', 'both') else 'accepted',
                               rock='low_shoulders' if modifier else 'accepted', rock_modifier=modifier)
        return dict(hdr=hdr, rgb=cr.PI.look.finish(hdr, **cr.FINISH))
    return call

def digest(a):
    return dict(dtype=a.dtype.str, shape=list(a.shape), sha256=hashlib.sha256(a.tobytes()).hexdigest())

if mode == 'equal':
    original = build('original'); expected = checked(original, first)
    del original; gc.collect()
    proofs = {}
    for route in ('default', 'shared') if kind in ('reveal', 'watch', 'map') else ('default',):
        render = build(route); actual = checked(render, first)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert actual[key].dtype == expected[key].dtype
            assert np.array_equal(actual[key], expected[key]), (route, key, first)
        proofs[route] = {key: digest(a) for key, a in actual.items()}
        del render, actual; gc.collect()
    print(json.dumps(dict(kind=kind, frame=first, scale=scale, exact_equal=True, proofs=proofs)))
else:
    out = Path(args[5])
    if out.parent.resolve() != (root / 'renders').resolve() or not out.name.startswith('cand_'):
        raise SystemExit('Use a new cand_* directory directly inside renders/')
    directories = [out] + ([out.with_name(out.name + '_matte')] if page_kind else [])
    if any(p.exists() or p.is_symlink() for p in directories):
        raise SystemExit('Output directory already exists; never overwrite or silently resume')
    for p in directories:
        p.mkdir(parents=True, exist_ok=False)
    render = build('candidate', chosen); rows = []
    for f in range(first, last + 1):
        started = time.perf_counter(); result = checked(render, f)
        arrays = [('rgb', out, result['rgb'])]
        if page_kind:
            assert result['alpha'].shape == (H, W)
            arrays.append(('matte', directories[1], np.repeat(np.clip(result['alpha'], 0, 1)[..., None], 3, axis=2)))
        files = {}
        for label, directory, image in arrays:
            target = directory / f'f_{f:05d}.jpg'
            pixels = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
            with target.open('xb') as stream:
                Image.fromarray(pixels).save(stream, format='JPEG', quality=95, subsampling=0)
            files[label] = dict(name=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        row = dict(frame=f, seconds=time.perf_counter()-started, files=files)
        rows.append(row); print(json.dumps(row), flush=True)
        del result; gc.collect()
    with (out / 'render_receipt.json').open('x') as stream:
        json.dump(dict(kind=kind, candidate=chosen, scale=scale, dimensions=[W, H],
                       first=first, last=last, complete=True, frames=rows), stream, indent=2)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps(dict(process_peak_rss_bytes=rss if sys.platform == 'darwin' else rss*1024)))
PY
}
```

## Switches, equality probes and native commands

Use the `equal` command first. It performs real renders through the original and omitted-default APIs; Night and Map additionally compare the shared accepted path. Pages compare HDR, alpha and finished RGB. Cold's probe is a frame common to both timings. These are new measurements when executed, not results asserted here. Append `1` to an equality command only when a native probe is authorized and resources permit it.

| Candidate | Rendering switch and public module | First equality command |
|---|---|---|
| Reveal night | `variant='night-fire', kind='reveal'` in `shots/run/beacon_night_candidates.py` | `ld_candidate equal reveal 3040` |
| Watch night | `variant='night-fire', kind='watch'` in the same module | `ld_candidate equal watch 4599` |
| Trap | `Scene(variant='front_smoke_near')` in `shots/embers/c_v5_trap_candidates.py` | `ld_candidate equal trap 2479` |
| Map | `candidate='beacon-falloff'` in `shots/map/last_beacon_candidates.py` | `ld_candidate equal map 3724` |
| Deep | `make_renderer('leaned_ladders')` in `shots/map/book_c_v5_deep_candidates.py` | `ld_candidate equal deep 4360` |
| Cold | `Scene(variant='lead24')` in `shots/embers/c_v5_cold_leadin.py` | `ld_candidate equal cold 3840` |
| Pen | `make_renderer('soft_spine_metal')` in `shots/map/book_c_v5_pen_candidates.py` | `ld_candidate equal pen 5560` |
| T1 | `make_renderer('current-words', with_fire=False)` in `shots/map/book_c_t1_candidates.py`; `no_burn=True` | `ld_candidate equal t1 430` |
| A crossing | `rope='snow_clearance'`; `rock='low_shoulders'` plus the explicit modifier in the helper, in `shots/run/crossing_candidates.py` | `ld_candidate equal crossing 5584` |

Each following command renders a native batch into its named new folder. Book shots also write the sibling `_matte` folder. For a one-frame native probe, replace both frame arguments with the probe frame above and use a different new folder ending `_probe`; do not use that probe folder for the full batch.

```bash
ld_candidate render reveal night-fire 2880 3119 renders/cand_runC_reveal_night_fire_native
ld_candidate render watch night-fire 4480 4719 renders/cand_runC_watch_night_fire_native
ld_candidate render trap front_smoke_near 2320 2639 renders/cand_embers_C5_trap_front_smoke_near_native
ld_candidate render map beacon-falloff 3440 3839 renders/cand_map_last_beacon_falloff_native
ld_candidate render deep leaned_ladders 4240 4479 renders/cand_book_C5_deep_lamp_ladders_native
ld_candidate render cold lead24 3816 3999 renders/cand_embers_C5_cold_lead24_native
ld_candidate render pen soft_spine_metal 5440 5679 renders/cand_book_C5_pen_metal_native
ld_candidate render t1 current-words 320 559 renders/cand_book_C_T1_current_words_native
```

A's two changes remain separate comparisons until their contact and motion are actually verified. Choose one of these batches; running all three is not implied authorization:

```bash
ld_candidate render crossing rope 4880 5839 renders/cand_crossing_rope_native
ld_candidate render crossing rock 4880 5839 renders/cand_crossing_rock_native
ld_candidate render crossing both 4880 5839 renders/cand_crossing_both_native
```

The first T1 batch covers the complete C320–559 mountain shot; its handwriting is active only at C400–539. It reconstructs the original engine without filmed burns. The accepted/default equality check does not establish that the opt-in current wording is legible: inspect the write-on and native text crops after any authorized render. Deep's `leaned_ladders` changes both the lamp and ladders; it is not an isolated ladder experiment.

## Measured cost and limits

The figures below come from the completed 960×402 receipt rows and `/usr/bin/time -l` process high-water values. RSS is the whole study/probe process, often containing several variants; it is not isolated candidate memory. Medians are seconds per recorded row, with the row's scope stated. No native cost is extrapolated.

| Study | Probe peak RSS, bytes | Study peak RSS, bytes | Median seconds / denominator |
|---|---:|---:|---|
| Reveal night | 1,211,088,896 | 1,544,716,288 | 11.581021 / 25 batches, accepted + night-fire + night-wisp together |
| Watch night | 1,357,004,800 | 1,586,069,504 | 10.906966 / 27 batches, the same three variants together |
| Trap front_smoke_near | 1,159,249,920 | 769,409,024 | 0.476035 / 170 candidate frames |
| Map beacon-falloff | 399,425,536 | 655,474,688 | 0.371680 / 50 batches, accepted + both territory variants together |
| Deep leaned_ladders | 1,082,933,248 | 1,062,813,696 | 1.014907 / 26 candidate frames |
| Cold lead24 | 815,661,056 | 875,741,184 | 0.493437 / 47 candidate frames |
| Pen soft_spine_metal | 525,910,016 | 421,756,928 | 1.399913 / 26 candidate frames |
| T1 archived legacy-seed C430 | 2,205,745,152 — failed gate | Not measured | 3.631718 / one current-word frame; current CRC revision not measured |
| A crossing A5584 | 2,960,359,424 — aborted before first completed image | Not measured | Not measured |

Night rows include one shared AOV, all compositions/masks and beacon projection, excluding JPEG output. Trap rows include render, validation, output/hash and resource/source checks; Map rows also include diagnostics and field archives. Deep/Cold/Pen/T1 intervals include rendering, grading where applicable, output and checks. Factory construction is outside these row timers; RSS includes it. These are bounded study measurements, not production throughput benchmarks or estimates.

Prior sampled equality succeeded at Reveal C3040, Watch C4599, Trap C2479, Map C3724, Deep C4360, Cold C3840 and Pen C5560. Deep's original recorded equality covered finished RGB and alpha; the command above additionally checks HDR. All 24 sampled Cold frames C3840–3863 matched the accepted float hashes and JPEG bytes. T1's successful pixel comparison belongs to its archived legacy hash seed, whose process still failed the memory gate; it does not cover the public CRC32 revision. A crossing has no equality result.

## EDIT consequences and acceptance questions

Replace the selected shot's take stem and `folders` entry with the new rendered folder only after delivery validation and the director's adoption decision. Preserve absolute frame numbers and `off=0`; book take `matte` must name the matching sibling folder. Do not overwrite the accepted folders or rewrite `BAKED_TEXT` to describe pixels that have not been delivered.

| Candidate | Consequence |
|---|---|
| Reveal / Watch night | Replace the two corresponding take stems. Ignition and cut ranges stay fixed. This is a sequence look decision across Reveal → Beacon Run → Watch; Beacon Run was unavailable and has no tested night treatment. Inspect both joins before adoption; these two commands do not resolve that gap. |
| Trap | Replace `embers_C5_trap` for C2320–2639. The same event frames remain; verify withdrawal, return and the final paired leaders in motion. No new sound or caption timing follows from the switch. |
| Map | Replace `map_last_beacon_C`. Preserve holdout timing C3724–3783, catch C3784–3791 and all-lit state from C3792. Whole-territory readability improved in the 960 still study; the graphic border style and incoming cut remain review questions. |
| Deep | Replace `book_C5_deep_abandoned` and its matte. Timing is unchanged. Both equipment designs change; captioned readers understood leaving the gold but did not explicitly identify miners disappearing or the lamp/ladders as changes. Reassess against the recovered earlier mine footage and current two-line caption placement. |
| Cold | Change `edit/edl_v3.py` `COLD_CUT` from 3840 to 3816 and select the lead24 stem. The Map then ends at C3815, retaining 24 all-lit frames; all forges still shut down at C3848. R15 spans [3740,3836), so its last 20 frames cross onto lit Cold. Review that caption/shot boundary and preserve the score's C3848 silence; no sound change was made by this study. |
| Pen | Replace `book_C5_pen` and its matte; keep C5440–5679. Metal was identified as a pen at 480 pixels; the softened gutter alone did not establish that viewers saw a book. Current R22 is EDIT's two-dimensional caption, not baked text. Inspect its actual placement in the composite. |
| T1 current words | Replace only the C320–559 mountain shot's `book_C` take/matte with the new stem. Keep R02 `in_picture`, [400,540). After delivery, record the new stem/range and exact current wording in `edit/titles.py` `BAKED_TEXT`; its old `book_C` record describes the old delivered wording. The separate `no-t1` option would require an EDIT-set caption and is not adopted by this recipe. |
| A rope / rock | Replace only the crossing take for A4880–5839 after validation. A finals remain held pending the ending decision. The rope's terrain clearance is sampled; the rock's unchanged 0.42 m analytic step height does not prove actual boot support. Inspect the full motion, especially A5040–5063 and A5578–5601, contact, shadow and both shot boundaries. |

Validate frame coverage, dimensions, JPEG decoding, mattes, source identity and frozen accepted assets after rendering. Then inspect native frames, 24-frame strips and actual playback of the joins; still or model-reader findings do not substitute for motion or audience approval. Re-run the current EDIT acceptance checks after any stem/timing/text-metadata change. This runbook performs none of those adoption steps automatically.
