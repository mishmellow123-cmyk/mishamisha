# SP noise compilation study

Separate proposal from the accepted A14/A15 work, begun at production `61cf28b`. The prior disposable experiment changed only SP's local noise-inline policy, produced an identical A4240, and completed its first real shader call in 33.735 seconds. It did not establish all-material equivalence or matched cold/warm performance.

This branch will keep the global terrain noise dispatcher unchanged and preserve the shared renderer's current recipes and motion controls. The candidate is a small SP-local wrapper around the existing noise function with Numba IR inlining disabled and its fastmath policy explicit; LLVM may still inline machine code.

Validation lanes run independently: compiled scalar/helper regressions; frozen-source original/candidate render comparisons covering active materials and real A14/A15 frames; code review of inlining, cache, numerical and throughput risks. All measurements must identify exact source, environment, cache and thread settings. Any timeout or nonzero numerical difference must remain visible in the report. Neither prior timing is a controlled speedup claim.

No farm actions, merge, EDIT changes or adoption are part of this study. PR #8 remains separately ready for director review, with its A15 source/join dependency. Claude and Codex now exchange messages through ~/ldfarm/comms, while code stays on GitHub. Codex checks TO_CODEX.md after each completed task.
