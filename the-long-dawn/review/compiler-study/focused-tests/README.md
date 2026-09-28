# Compiled test evidence

The candidate passed **64/64 RUN tests: 56 existing tests plus 8 new noise-dispatch tests**, with no failures, errors or skips. This branch does not contain PR8's A14 catch tests. The runner confirmed `SP.render.signatures == []` afterward: **the full figure renderer was neither compiled nor executed** by this suite.

The complete [suite log](full-suite.log) records 92.540 seconds in unittest and 99.80 seconds command wall time (94.19 user, 3.45 system). The run started with an empty private Numba cache, JIT enabled, Numba and BLAS/OMP/MKL/VECLIB limits of one, and OpenCV `setNumThreads(0)` after imports (getter one). These are test-suite timings; they do not measure renderer speed.

The [receipt](suite-receipt.json) records Python 3.13.9, Numba 0.62.1, llvmlite 0.45.1, NumPy 2.3.5 and OpenCV 5.0.0.93, along with every RUN source/test hash. HEAD at measurement was `5890f876a4eec9b1577bec5fdd4eada41a455169`; the candidate changes were uncommitted then. The [shader](../../../shots/run/sdfppl.py) SHA256 was `9140e4ccd060db8895a2952ea5aab74a3c444bda12e29fd43ab8ec12909b2855`; [new tests](../../../shots/run/tests/test_sdf_noise_dispatch.py) SHA256 was `985f3cacf4e4aedbf40dfe63c3216587da3bf5347a18af526df1f60b5e9d93f1`. Those hashes remained unchanged during verification.

## Numerical coverage and limits

The [focused log](focused-tests.log) preserves the eight-test run (27.902 seconds), the forced-inline negative control and every differing helper value. All 1,500 scalar samples across the 15 actual shader seeds match exactly, including negative and large coordinates, lattice boundaries and adjacent floating-point values. The 1,000 actual cloth/stone noise expressions extracted from the shader and 225 samples through a strict-math caller also match exactly.

Composed helpers show small float64 differences: horn differs in 29/64 values, at most 3 ULPs / `1.6653345369377348e-16`; rough box differs in 1/44 values, at most 1 ULP / `1.734723475976807e-18`. The logs retain each actual/expected pair. Helper regression limits are these observed fixture ceilings. A deliberate `+1e-6` noise mutation through each actual helper is rejected, and the forced-inline policy mutation produces one expected test failure. This evidence does not establish unchanged trace hit masks, depths, full-frame output, other input domains or other compilers/platforms.

## Provenance and reproduction

[Packaging provenance](packaging.json) records the original and packaged SHA256 of each evidence file. Only machine-specific workspace/temp prefixes were normalized in the measured logs and receipt; source hashes, measurements and raw numerical differences were preserved. `<WORKSPACE>` denotes the removed local workspace prefix. The original runner's hash inside the receipt identifies [run_suite_original.py](run_suite_original.py), a byte-identical archival copy that retains its historical sibling-directory layout.

[run_suite.py](run_suite.py) is a portable derivative, added after measurement. It accepts a repository root and new output directory; its test selection, shader source pin, thread settings and no-full-render assertions match the original runner. From the repository root, using a Python environment with the dependencies above:

```sh
python3 -B the-long-dawn/review/compiler-study/focused-tests/run_suite.py --tree . --out /tmp/sdf-noise-suite-rerun
```

Choose an output directory that does not exist. The runner creates a private cache there and writes a new receipt; capture stdout/stderr separately if another log is required. Existing evidence and generated cache files were not overwritten or rerun for this package. The original cache is intentionally excluded.
