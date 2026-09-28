# SP compiler comparison harness

`benchmark.py` compares the pinned `61cf28b` renderer with the current worktree's `sdfppl.py`. Preparation copies the reference's Python sources and committed NumPy inputs into two isolated trees; only the candidate SP file differs. It never copies Numba caches or rewrites the reference's decorators. Each run starts a separate process, imports from its own tree and uses its own cache. Shared `mt.noise` remains the pinned original in both.

The material mode calls the real `SP.render` on twelve small diagnostic cases: cloth at two distances, cloth with foreground depth and with zero fog, iron/wood, ordinary glass, horn, small emissive glass, the actual `stone_ring` helper, and a rough box with displacement enabled and disabled. `lantern_v3` supplies the horn geometry. These fixtures are numerical probes, not proposed film designs. The raw float image, modified forward-depth buffer and Boolean mask of depth values changed by each SP call are retained. The mask distinguishes changed coverage from numeric depth differences; it does not count hits that write an identical depth. Every case must alter pixels; paired controls must demonstrate horn RGB, displaced depth, fog and foreground occlusion. A gamma preview supports inspection; it is not used for metrics. A14/A15 modes call their complete renderers with actual scene tables, cameras, terrain, fire, finishing and A14 motion blur; they additionally capture each real SP call's inputs and float outputs.

Each case runs once plus three identical warm repeats by default. The first SP call in a cold process includes compilation and rendering; subsequent same-signature calls measure warm execution. The separate `cache-hit` phase launches a new process against that variant's populated cache and rejects a missing SP cache hit or any output/input difference from that variant’s cold run. Full-frame timings also contain other kernels and capture overhead; use the recorded SP call times for the compiler comparison. Cold compilation time cannot be inferred by subtracting arbitrary full-frame timings.

All runs use one Numba thread, sequential OpenCV and BLAS thread limits of one. Source/input hashes, call settings, Python/library versions, signatures, cache hits/misses and process peak RSS are recorded. The wall-clock timeout terminates only the launched process group. No CPU affinity or background-load isolation is claimed. Run the variants sequentially under comparable machine load; do not interpret a one-off timing difference as a throughput estimate.

From the repository root, with the validation Python on `PATH`:

```sh
python -B the-long-dawn/review/compiler-study/benchmark.py prepare --repo . --out /absolute/new/benchmark-pair
python -B the-long-dawn/review/compiler-study/benchmark.py run --out /absolute/new/benchmark-pair --variant reference --mode material --phase cold --timeout 1200
python -B the-long-dawn/review/compiler-study/benchmark.py run --out /absolute/new/benchmark-pair --variant candidate --mode material --phase cold --timeout 1200
python -B the-long-dawn/review/compiler-study/benchmark.py run --out /absolute/new/benchmark-pair --variant reference --mode material --phase cache-hit --timeout 1200
python -B the-long-dawn/review/compiler-study/benchmark.py run --out /absolute/new/benchmark-pair --variant candidate --mode material --phase cache-hit --timeout 1200
python -B the-long-dawn/review/compiler-study/benchmark.py compare --reference /absolute/new/benchmark-pair/results/material/reference/cold --candidate /absolute/new/benchmark-pair/results/material/candidate/cold --output /absolute/new/benchmark-pair/material-comparison.json
```

For real frames, use `--mode A14 --frames 4202,4208,4239` or `--mode A15 --frames 4240,4360`, with explicit `--scale 0.5 --ss 1.5`. Each mode has separate per-variant caches. `--repeats 1` reduces full-frame work to two calls per frame. Preparation and `--help` do not import a renderer. A material render still compiles the entire SP kernel: a small image does not make that initial compilation cheap. Long runs require the coordinator's resource slot.

Comparison verifies source isolation and matching inputs/settings before comparing every saved float element; any output difference returns nonzero. Warm repeats must reproduce their first call exactly. `test_benchmark.py` includes a known changed pixel and mismatched-input/source negative controls and imports no renderer. These sampled cases cannot establish all-input equivalence or creative acceptance.
