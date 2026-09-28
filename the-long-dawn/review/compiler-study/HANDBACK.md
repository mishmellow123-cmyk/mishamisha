# Compiler-study handback

PR [#10](https://github.com/mishmellow123-cmyk/mishamisha/pull/10), `codex/sdf-noise-compile`, remains **draft and unadopted**. The candidate completes its validation suite, but the original worker timed out after 2400 seconds without returning its first SP render. Image/depth/hit-mask equivalence remains unresolved. All study processes have exited.

## Change and rationale

The only production-code change is [sdfppl.py](../../shots/run/sdfppl.py#L27): an SP-local cached noise wrapper with Numba IR inlining disabled and `fastmath=True` explicit. It still calls the shared noise implementation; terrain consumers retain their own binding. LLVM may inline machine code. The experiment targets expensive compilation from repeated noise IR expansion; the precise compiler pass responsible for the original delay is unproven. No rendering recipe, motion control, farm job or EDIT selection changed.

The study freezes production `61cf28b390cf00fdf860401127ca3e7f6cf33491`, then changes only `sdfppl.py`. The frozen RUN/lib/montage sources remain unchanged at production `7f6ffdd`. PR8's catch/fire changes are excluded from these film-frame results.

## Verified

- **64 compiled RUN tests pass**: 56 existing plus eight numerical regressions. Scalar noise and sampled cloth/stone coordinates agree exactly. Composed horn/rough-box helpers have documented differences of up to three/one float64 ULPs. The tests retain those values and reject substantive mutations; they do not prove all-input equivalence. See [numerical evidence](focused-tests/README.md).
- The [candidate suite](suite-results/README.md) covers 12 live material/occlusion fixtures and A14 4202/4208/4239 plus A15 4240/4360, with three warm repeats each. One SP specialization serves all cases. Full-frame linear/finished color and SP color/depth/masks repeat exactly; a fresh-process cache reload reproduces all saved outputs. All five native previews were inspected.
- Two [audit records](audit/README.md) separately check the harness/publication and actual saved arrays: 122 files, 61 cold/cache pairs, 12 live fixtures and five faithful PNGs. Their limits and denominators are explicit.
- Candidate first SP compile-plus-render call: **47.584665 seconds**; whole cold suite: **348.658020 seconds**. Reference launcher elapsed: **2400.086786 seconds**, terminated at its limit. These observations do not establish isolated throughput or a speedup ratio; workload overlap is documented.

## Unresolved and prepared

The exact question is whether this wrapper changes rendered RGB, depth or hit masks relative to the current original shader. The original has **zero completed calls and zero saved outputs** in this attempt. Its partial receipt says `starting` because that was the final worker checkpoint; the launcher and traceback establish timeout. Original cache reload remains unrun.

Two [one-second native samples](diagnostics/README.md) show Python set operations inside Numba compilation. Neither identifies the exact Python pass. A bounded cache inventory found no usable original cache in the searched project/farm locations; older source caches were not substituted.

A future attempt should first investigate the long compiler pass or allocate an explicitly longer resource window, then use [benchmark.py](benchmark.py) to prepare a fresh output directory. Reproduce suite mode with frames `4202,4208,4239,4240,4360`, scale `0.5`, supersampling `1.5`, size `96`, and three warm repeats. `run` supplies private caches and one-thread settings; `compare` requires matching source/runtime/input evidence. The original cold phase must complete before its cache-reload phase. Preserve any numerical difference. If the platform or runtime changes, rerun both variants there. Adoption and any farm launch remain the director's decision.

Local frozen sources, raw arrays and caches remain under `work/compiler-study/bench/suite-61cf28b` in the Codex workspace. Published receipts preserve source/harness hashes and normalized paths. No raw film arrays or caches were added to Git.

## Other completed requests

PR8 remains ready at `29b0d20`, with 64 compiled compatibility tests. Its actual farm A4240/source-receipt comparison still awaits the artifact promised through `~/ldfarm/comms/files`; the prepared A15 companion addresses the shared fire-table dependency. The v4 script read and capability assessment were appended to `TO_CLAUDE.md` and copied into `comms/files`. Script wording suggestions are proposals, not adopted edits. The mailbox is checked after each completed task.
