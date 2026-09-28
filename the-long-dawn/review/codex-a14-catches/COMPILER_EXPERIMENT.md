# Separate compiler-performance proposal

No renderer change was adopted. A disposable runtime changed only `sdfppl.gnoise3` from forced inlining to a non-inlined callable around the same Python function. A fresh `SP.render` dispatcher retained its original parallel/math options. Both experimental dispatchers used `cache=False`; other modules retained their original noise dispatcher, and all remaining cache writes were isolated.

One real A15 frame, A4240 at 960×402, produced exactly the same finished unquantized RGB hash and PNG bytes as the authoritative integration frame. The implementing session independently checked the saved float array and PNG against those recorded hashes. Source was `9b9027b`, incorporating production `6ae7180`; the later RUN-A4 change `8bcc818` was not part of this experiment.

The first actual SP call (compile plus execution) took **33.735 seconds**; the complete supervised process took **45.605 seconds**. It used one Numba thread and reused read-only copies of unrelated kernel caches, with every SP cache entry excluded. The original production-style check took 2050.893 seconds for its first frame, including compilation, rendering and load pauses, using two Numba threads. These unmatched runs do not establish a controlled speedup ratio. Warm throughput was not measured.

Runtime: Python 3.13.9, Numba 0.62.1, llvmlite 0.45.1, NumPy 2.3.5, OpenCV 5.0.0.93. BLAS/OMP/MKL/VECLIB limits were one; OpenCV was sequential. The ten-minute timeout did not fire. No experimental process remains. [Full receipt](receipt-compiler-experiment.json), [comparison](comparison-compiler-experiment.json).

The tested override, after importing the real scene and before its first SP draw, was:

```python
original_noise = SP.gnoise3
original_render = SP.render
noise_options = dict(original_noise.targetoptions)
noise_options['inline'] = 'never'
render_options = dict(original_render.targetoptions)
SP.gnoise3 = numba.jit(cache=False, **noise_options)(original_noise.py_func)
SP.render = numba.jit(cache=False, **render_options)(original_render.py_func)
```

Keep this separate from the accepted shot changes. Before proposing a production patch, reconcile current shader ownership and reproduce in a fresh output/cache directory. Keep other noise consumers unchanged. Compare representative cloth, glass/horn and rough-stone paths, deterministic coordinates/seeds including negative values and cell boundaries, and full A14/A15 frames against unchanged source. Report every unquantized difference. Measure cold compilation and repeated warm rendering with matched inputs, runtime, threads, cache state and load; rerun the relevant compiled suite. One identical watcher frame does not establish all-material, all-shot or cross-platform equivalence.
