# One-second reference-worker diagnostic

At 2026-09-28 06:11:13 UTC, the process command identified PID 98102 as this study's `benchmark.py _worker --variant reference --mode suite --phase cold`. A one-second macOS `sample` completed successfully. The [receipt](reference-98102-20260928T061113Z.json) retains the command verification and `ps` snapshots; the [native stack sample](reference-98102-20260928T061113Z.sample.txt) retains the full call graph.

All 787 main-thread samples include Numba's dispatcher `compile_and_invoke`. Python set operations dominate that interval: 696 samples pass through `PyNumber_And`, and 595 through `set_intersection` into `set_contains_entry`. Nine TBB worker threads were waiting on semaphores. This supports active Python-side Numba compilation during the sampled second. The native stack does not identify the exact Python compiler pass; the root cause of the long compilation remains unproven.

| Observation | Elapsed process time | Memory reading |
|---|---|---|
| `ps` before sample | 08:26 | RSS 3,042,288 KiB |
| `ps` after sample | 08:28 | RSS 2,525,216 KiB |
| macOS `sample` | During the one-second sample | Physical footprint `8.1G`; peak `8.2G`, as displayed |

`ps(1)` defines RSS in 1024-byte units. Physical footprint is a separately reported metric; these values are not interchangeable. The post-sample `ps` CPU reading was 68.7%. This snapshot does not measure the duration of a compiler pass or predict completion.

[Packaging provenance](provenance.json) records SHA-256 hashes and byte counts for each original and packaged artifact. Packaging replaces the session workspace prefix with `<WORKSPACE>`, the local Anaconda root with `<PYTHON_RUNTIME>`, and user-home prefixes with `<USER_HOME>`; existing macOS wildcard redactions remain. System-library paths, sample counts, symbols, addresses, timestamps and numeric observations are preserved. The original local files were unchanged.

This diagnostic accompanies the draft compiler study. It provides no rendered-output equivalence verdict or controlled performance ratio. Packaging performed no new sampling, renderer work or process lifecycle action.

## Second one-second sample

The [second sample](reference-98102-20260928T0637Z.sample.txt) is timestamped **2026-09-28 06:37:20.577 UTC**, with launch time **06:02:47.534 UTC**. All **752 main-thread samples** include `compile_and_invoke`; **710** pass through `PyNumber_And`, and **690** through `set_intersection` into `set_contains_entry`. It again shows Python set operations within Numba compilation. The sample reports physical footprint and peak as **`9.0G`**. This second package contains no adjacent `ps` snapshot and supplies no new RSS reading for the table above.

[Separate provenance](provenance-0637.json) records the original and normalized sample hashes. The exact compiler pass, causal explanation and advancement to any later compilation stage remain unproven; no LLVM-progress or rendered-output verdict follows. The first provenance file now pins this appended README and retains its previous README hash in `readme_history`. No additional sampling or process action occurred during packaging.
