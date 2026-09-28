# Compatibility update — 2026-09-28

PR #8 now integrates production `61cf28b` in merge `1f9a38c`. All **64 compiled RUN tests pass on that exact merge**; the source hashes and HEAD were unchanged throughout the run. JIT enabled, Numba two threads, BLAS one, sequential OpenCV; the run reused this audit's private cache. Its 0.696-second test time is not a cold-render measurement. [Log](merged-compiled-run-suite.log), [receipt](merged-suite-receipt.json), [source hashes](merged-suite-inputs.json).

Since the previously integrated `70d102b`, the only `sdfppl.py` change adds optional `reach_w=None` construction controls. All 11 compiled SP function ASTs are identical; removing the default-inactive additions restores the previous traveller/seated construction ASTs. The watcher does not supply the new option. [Source audit](source-receipt.json). No renderer change was needed for integration.

This adds current-source compatibility evidence, not a new full-frame render. The earlier pixel/source boundaries in the parent review still apply. Claude's 05:40Z mailbox message reports A15 rendered and selected in EDIT; Codex has not yet received actual farm A4240 or unit source receipts. The prepared A15 companion remains the route to a consistent enlarged-fire pair if the director accepts #8. No farm job was launched by Codex.
