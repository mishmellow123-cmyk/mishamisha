# C v5.2 ink pages

Three opt-in shots use the existing book renderer, paper, ink, gilt and film finish. `book_c.py` and its old timeline are unchanged, including BURN's `--no-burn` path. New entrypoint: [book_c_v5.py](../../shots/map/book_c_v5.py).

| Shot | Absolute C frames | Output stem | Picture |
|---|---|---|---|
| 10, Refusal | 2080–2319 | `book_C5_refusal` | A gloved palm offers the ring; a hooded figure turns away and raises a hand. The drawing appears stroke by stroke, then holds. |
| 18, Deep abandoned | 4240–4479 | `book_C5_deep_abandoned` | The original mine drawing, finished and still, with miners removed, empty ladders and a lantern at the entrance. The original gilt vein remains. |
| 22, Pen | 5440–5679 | `book_C5_pen` | A wooden dip pen with ferrule and split nib rests across the blank spread. Camera and hearth move slowly. |

Captions remain EDIT's. No new caption or title is baked in. PEN uses the permitted insert: it starts on the blank spread, so EDIT must join it to the preceding page turn. It does not animate a leaf through a stationary pen.

The refusal reuses the existing variable-width pen strokes and hatching. Review rejected the first hood as a beak/helmet and the sleeve hatching as a comb; the revised cowl has a dark opening without a face, irregular cloth folds and a broken hem. The mine subclass consumes the original miners' random draws while discarding their strokes; this keeps the later rock and gold geometry exact. Its new ladders and lantern use heavier local ink so they survive the full-page camera.

The pen has world-space volume, interpolated normals, metal/wood shading, perspective-correct camera-forward depth and a soft shadow cast toward the existing hearth light. Its resting pose minimises the shaft's gravitational height subject to clearance over the cockled paper. The first simple lifted pose left the nib suspended; the revised pose has supports on both sides of the shaft's centre of mass.

## Reproduce

Run from `the-long-dawn/`, with NumPy, Numba, SciPy, OpenCV and Pillow installed:

```sh
NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 shots/map/book_c_v5.py --shot refusal --frames 2080-2319 --out /absolute/output/book_C5_refusal --format jpg
```

Select `deep_abandoned` or `pen` and its corresponding frame range for the other shots. `--out` must be absolute; the driver writes a matching `<out>_matte` folder. Default format is PNG; `--format jpg` writes atomic JPEG q95, 4:4:4 files. `--scale .5` gives 960×402 tests; default is 1920×804. `--skip` skips only when both RGB and matte files of the requested format already exist. It does not validate those existing files.

Separate job JSONs declare the exact 240-frame ranges and fresh output stems: [Refusal](../../cloud/jobs/book_C5_refusal.json), [Deep](../../cloud/jobs/book_C5_deep_abandoned.json), [Pen](../../cloud/jobs/book_C5_pen.json). They run on this Codex branch. No job was launched by this PR; no EDIT, sound or farm code changes are included.

## Verification

The focused suite checks absolute timeline boundaries, output/job contracts, deterministic stroke construction, caption clearance, preservation of every non-miner geometry stroke, the pen's volume and paper clearance, support on both sides, camera-forward depth interpolation, foreground occlusion and JPEG 4:4:4 output. Existing library/output and cloud exit suites also pass. The mine test rejects a mutation that stops consuming the removed miners' RNG; the support test rejects the old floating-nib slope.

```sh
python3 -B -m unittest discover -s shots/map/tests -p test_book_c_v5.py -v
python3 -B -m unittest discover -s lib/tests -p 'test_*.py' -v
python3 -B -m unittest discover -s cloud/tests -p 'test_*.py' -v
```

The final source passed **21 tests** (10 focused page contracts and 11 existing library/cloud tests) with Numba JIT enabled. [Test log](tests.log). Tests establish the stated numerical contracts; EDIT's joins remain a separate review.

## Render review

All three full-resolution stills were inspected at 1:1 and at 480×201; every frame in the three 24-frame strips was inspected in its sheet. The figure gesture and pen contact hold; the mine preserves its original layout. Lantern and ladder details remain small at 480 pixels wide. The films' captions and adjacent-shot joins were not tested here.

| Shot | Native still | 24 consecutive frames | Full sample seconds | Peak RSS bytes |
|---|---|---|---:|---:|
| refusal | [2272](refusal-full.png) | [2192–2215](refusal-motion-sheet.png) | 15.308921 | 681132032 |
| deep_abandoned | [4360](deep_abandoned-full.png) | [4360–4383](deep_abandoned-motion-sheet.png) | 18.406665 | 1333444608 |
| pen | [5560](pen-full.png) | [5550–5573](pen-motion-sheet.png) | 11.362073 | 660553728 |

These single full-resolution samples include asset construction and overlap other bounded workers; they are not an isolated throughput benchmark. The three 960×402 strips contain **72/72 decoded frames and 69/69 adjacent pairs**, with no black, constant, held-duplicate or pop candidates under the existing checker. That heuristic is a review aid, not a visual quality guarantee. [Exact receipts, source hashes and per-frame measurements](verification.json).

## Delivery status

Source for all deliveries is `ebcdd0a`; review-only commits do not change it.
Each completed row has decoded 1920×804 JPEG q95 4:4:4 RGB frames and matching mattes, independently rechecked SHA256 hashes, and a silent 24 fps preview.

| Shot | RGB / mattes verified | Actual delivered samples | Receipt / full-shot QC |
|---|---:|---|---|
| 10 refusal | 240 / 240 | [24 sampled frames](refusal-delivery-sheet.jpg) | [Receipt](refusal-delivery.json), [QC](refusal-delivery-qc.json) |
| 18 deep_abandoned | 240 / 240 | [24 sampled frames](deep_abandoned-delivery-sheet.jpg) | [Receipt](deep_abandoned-delivery.json), [QC](deep_abandoned-delivery-qc.json) |
| 22 pen | 240 / 240 | [24 sampled frames](pen-delivery-sheet.jpg) | [Receipt](pen-delivery.json), [QC](pen-delivery-qc.json) |

Frames live under `~/ldfarm/out/book_C5_<shot>/` and its `_matte` sibling. Previews and original full receipts live under `~/ldfarm/comms/files/book_C5_<shot>_preview.mp4` and `book_C5_<shot>_receipt.json`.

The 24-frame motion strips and native stills above were visually reviewed before final rendering. The complete final sequences await the later visual audit; no adjacent-shot joins or final captions have been approved here. Deep’s lantern and ladders remain small at 480 pixels wide. PEN is a supported physical insert onto blank pages; the incoming page turn belongs to EDIT.

Frame-QC candidates are review flags, not calibration or artistic approval. [Delivery packaging/provenance](delivery-provenance.json), [runtime versions](runtime.json).
