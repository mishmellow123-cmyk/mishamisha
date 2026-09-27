# Temporary handback: render failures and picture readiness

Prepared 27 September 2026 against `claude/long-dawn-v2` at
`07d5dca6d6111ad2fb5ec4e0b40008e5b22072ec`, in an isolated checkout on
`codex/long-dawn-handoff-validation`. The active film is the v3 trilogy; the repository's default branch
and original 119-second README do not describe this state.

## Integration first

The original production checkout has unpublished changes. Its supplied Git status includes `cloud/farm.py`,
`edit/NOTES_v3.md`, `edit/edl/edl_A.json`, `edit/edl/edl_B.json`, sound code and event files, and
`shots/run/reveal_a.py`, plus replacement blue-hour/crossing job files and substantial untracked media/review
work. Preserve and inspect that diff before integration. This patch keeps its notes in this separate file
to avoid editing the active department notes.

Farm units fetch the production branch at dispatch. Reconcile running requests and decide a rollout boundary
before applying a shared-driver change; do not reset the active checkout or restart/resubmit jobs merely
because the model session paused. Successful command-exit notices in the exported conversation continued
after the model limit, but neither those notices nor a worker's emptied delivery directories prove the
current output inventory.

No production checkout, render process, cloud node, shot design or media asset was changed by this patch.
No full film or Blender shot was rendered during validation. The original laptop's current farm receipts,
uncommitted diff contents and latest masters were unavailable to this checkout.

## Changes

`shots/montage3d/render.py` now returns failure when Blender fails, post-processing raises, or requested frame
notifications do not result in successful post-processing. It drains the poster queue before checking the
result, preserves completed partial outputs, reaps the child on driver exceptions, and removes only a lock
created by that invocation. A regression forces the producer/consumer interleaving that previously allowed
the poster to miss queued work during shutdown. `melt_local.py` retains its additional file-existence check.
The farm's decoding, dimension and completeness checks remain independent.

The driver's guarantee is successful child exit plus successful return from `post_frame()` for every
requested frame in this invocation. It does not independently decode images or certify auxiliary outputs:
Ring-mask helpers can return without a mask if its AOV is absent, and the farm separately validates those
expected masks. The unchanged legacy `cloud/run_job.py` can still call a job complete after logging a
nonzero renderer exit if every expected image was pushed.

`lib/look.py` now checks the PNG encoder's boolean result before publishing its temporary file. An encoding
or rename failure preserves the destination, raises the original error, and attempts temporary-file cleanup.
Dithering and pixel processing are unchanged. Tests inject failure states; no production disk or encoder
failure was reproduced.

The EDL explicitly declares provisional takes. Delivery QC and H9 use this metadata instead of inferring
eligibility from descriptive prose. All-present provisional footage can still appear in a partial master,
but it cannot establish picture-source completeness or trigger the watcher's completion notice. Approved
reuse stays eligible; no timeline, take priority or creative approval has changed.

Specifically, `final_eligible=False` marks the five provisional take families, and
`UNDER_FINAL_ELIGIBILITY` marks the provisional `embers_C3_half` folder when used beneath a book layer.
H9, QC and watcher signatures share `assemble.provisional_sources`, which follows selected sources per
frame, including held under frames and matte presence. The metadata lookup mirrors the unchanged
compositor; tests exercise its actual source reads to pin parity. Eligibility-only changes refresh QC
without invalidating picture segment caches. An arriving final under-layer changes the affected picture
key and the watcher signature as required.

`complete` is a picture-source signal. Audio, finish backlog and creative approval retain their separate
checks. Coverage is based on source lookup: `--qc-only` inspects current source folders, so this flag does
not attest to the provenance of a previously encoded master. Missing or unreadable matte/under-layer
behavior remains an existing limitation outside this correction.

## Validation

29 focused tests passed: 11 driver tests, 6 PNG I/O tests and 12 readiness tests. Shell syntax and whitespace
checks passed. The driver suite against the original `07d5dca` implementation produced 10 failures and one
passing control. The PNG suite against that original helper failed four tests (five failure records,
including two subcases), with two controls passing. Bypassing provisional-source reporting made both the
C6 take and C9 under-layer regressions fail. The C9 regression also failed before its under-layer fix.

Independent review confirmed unchanged compositor code and all five picture-pipeline hashes, and compared
all 35 C fixture segment keys against the baseline source-tracking implementation: identical. The tests
also establish that a final under-layer arriving later changes the affected picture key. This comparison
used controlled inputs, not the original laptop's cache. Readiness tests emit existing unclosed-file
`ResourceWarning`s from unchanged file-reading code.

The renderer contract tests use real child processes and the real poster thread, with image work replaced
by controlled fixtures. The readiness tests use the actual EDL and selection/QC/watcher code with synthetic
frame indexes and isolated media measurements. These establish software contracts, not production coverage
or visual/audio quality.

Run from the repository root in the project Python environment:

```sh
python -B -m unittest discover -s the-long-dawn/shots/montage3d/tests -p test_render_exit.py -v
python -B -m unittest discover -s the-long-dawn/lib/tests -p test_look_io.py -v
python -B -m unittest discover -s the-long-dawn/edit/tests -p test_picture_readiness.py -v
bash -n the-long-dawn/edit/refresh_watch.sh
git diff --check
```

## Resume the films

Read `BIBLE_V3.md`'s H5 calls and subsequent departmental notes alongside the original laptop's
`_local_logs/PACING.md`, `_local_logs/handoff/LAUNCHED_BY_MAIN.md`, current farm receipts and frame inventory.
The recovered brief retains A's uncoded giants and separately labelled coded-towers alternate, B's
single-night Vigil, C's book/ink language and invented Ring script, silhouettes/gloves/red shawl, distinct
landforms, and the white lantern core/match cut. The council's possible renderer rebuild was a proposal,
not an adopted architecture change.

Historical unresolved work below needs fresh review against the latest local assets:

- C's council was rejected visually; timing renders do not settle its stone, figures, torches or fire.
- A's seventh fire and watchers need visual work; blue-hour and crossing follow-ups concern figures,
  landscape and lantern construction. Preserve the accepted white core.
- C's Ring-melt JOB READY arrived after the model limit; verify launch/delivery before resubmitting it.
  C's dedicated hands/flint, fire test and flame-letter legibility also remain review items.
- C's replacement forging render for frames 1040–1439 was recorded as launched at 21:14Z; identify its
  output before assessing a superseded render.
- B's major jobs were mostly launched, with cairn/dusk follow-ups. The recorded-effects sound version
  reportedly passed technical checks but awaited adoption; A/C effects work remained.
- Finish uses the documented `250D_2383_fire` choice. Rebuild from current assets, clear the finish backlog,
  inspect picture in motion with sound, and run the intended fresh-context critic. Historical technical
  QC passes included partial masters with stand-ins and do not establish completed films.

These are reconstructed historical findings, not fresh judgments of the latest films. The supplied
conversation is not included in this branch.
