# Candidate suite — original comparison pending

This package contains **candidate-only results** from the frozen `61cf28b390cf00fdf860401127ca3e7f6cf33491` runtime with the SP-local noise wrapper. It excludes PR8's A14 catch changes. [Source inventory](sources.json) records exactly one changed runtime file, `sdfppl.py`; these previews belong to that frozen study configuration.

The [candidate receipt](candidate-cold-receipt.json) records **348.658020 seconds** for the suite and **47.584665 seconds** for the first `SP.render` call, a 96×96 cloth fixture that includes compilation and rendering. That call had no prior SP signature, one cache miss and no cache hits. Later material fixtures and the five film frames reused the same SP signature. Each frame ran once plus three warm repeats; its unquantized linear and finished RGB hashes matched exactly across all four runs. The [complete log](candidate-cold.log) and receipt retain individual timings, image/depth/mask hashes and cache counters. Total time includes other kernels and capture overhead.

Numba, BLAS/OMP/MKL/VECLIB were limited to one thread; OpenCV used `setNumThreads(0)` (getter one). Python was 3.13.9, Numba 0.62.1, llvmlite 0.45.1 and NumPy 2.3.5 on macOS. The original-process run was launched after candidate SP compilation, while candidate film frames were still running. Its receipt creation timestamp was 191.712453 seconds after the candidate's; the candidate suite lasted 348.658020 seconds. **The later candidate timings overlap that workload.** There was no CPU pinning or load isolation, and this package does not establish a speedup ratio.

The [fresh-process cache reload](candidate-cache-receipt.json) also completed: its first SP call took 0.014632 seconds with one cache hit, and the whole suite took 240.342568 seconds. All raw outputs and full-frame hashes match the cold candidate run exactly; warm repeats remain exact. This phase also overlapped the original compiler process. [Reload log](candidate-cache.log).

## Finished previews

All five PNGs are 960×402 at render scale 0.5 and supersampling 1.5. A14 used the actual motion-blur path. Conversion from the finished arrays is exactly `rint(clip(RGB, 0, 1) * 255)` to uint8, with no added gamma, exposure, contrast, crop or resizing. [Preview provenance](preview-receipt.json) records the original NPY/float-RGB hashes and PNG hashes; its contact-sheet entry refers to the local preview sheet, which is not included in this package.

| Frame | PNG |
|---|---|
| A14 4202 | [View](A14-4202.png) |
| A14 4208 | [View](A14-4208.png) |
| A14 4239 | [View](A14-4239.png) |
| A15 4240 | [View](A15-4240.png) |
| A15 4360 | [View](A15-4360.png) |

The implementing session inspected all five frames above at native resolution. They show the sampled compositions and watcher poses; no complete-motion or original-renderer equivalence claim follows from this inspection.

## Exact identity and remaining checks

Candidate `sdfppl.py` SHA256: `9140e4ccd060db8895a2952ea5aab74a3c444bda12e29fd43ab8ec12909b2855`. Original `sdfppl.py` SHA256: `9450d9becd261e4eb65487b1f4fecafb4996e4da2a652632a5e64c0c5a906c2a`. Harness SHA256: `d22693e1e1f2f8539dca7ebf20c07ef757251fcf5c3e8310f38ea4868e83075c`. Full runtime source/input hashes are in the source inventory and receipt. [Packaging provenance](packaging.json) preserves original and packaged artifact hashes; only machine-specific workspace prefixes were normalized in text.

[Pending checks](pending-checks.json) records the successful candidate cache reload and reserves the still-pending original-versus-candidate raw image/depth/hit-mask comparison, original cache reload and adoption verdict. Warm repeatability of one candidate does not settle those comparisons. The raw film arrays remain locally under `work/compiler-study/bench/suite-61cf28b/results/suite/candidate/cold`; no NPY or cache files were copied into Git. The earlier [pilot evidence](../pilot/README.md) remains separate and unchanged.
