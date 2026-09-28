# Codex handback — handover 3, A15 then A14

This supplements handover 2 and supersedes the older temporary handback for this night-shot lane. Film B is dropped; its harvested reveal remains only where the current A edit uses it. The rejected geometry-cairn studies in PR #2 remain rejected.

## Review order

1. [PR #7, A15](https://github.com/mishmellow123-cmyk/mishamisha/pull/7): code `8f56370`, reviewed media `575af89`. Start with [before/after and the visual audit](codex-a15-hearth/README.md), then native detail, the 24-frame strip and the A4239/A4240 join. The close fire is an opt-in world-space volume; the hearth follows the body-concealment staging alternative. Fine stone detail is mostly hidden. Some warm, faceted ledge remains, and fire style still needs a director judgment.
2. A14 on `codex/a14-beacon-catches`, code `7ab7793`, stacked on PR #7. See [the catch/motion review](codex-a14-catches/README.md). This enlarges approaching fires 1/3/5, adds a compact catch core, and repairs the opaque figure/hearth distance used by camera motion blur. A15 inherits those shared fire changes; this review includes the resulting boundary and A15 check.
3. [PR #6, relief study](https://github.com/mishmellow123-cmyk/mishamisha/pull/6): the adoption/strength decision is still outstanding. [The coordination note](https://github.com/mishmellow123-cmyk/mishamisha/blob/codex/a-cloudsea-relief/the-long-dawn/review/cloudsea-relief-20260927/COORDINATION.md) records the now-resolved A13 ranges. No A13 rerender or farm final was launched.

## Changes and evidence

A15 adds `shots/run/hearth_a.py` and `fire_near_a.py`, adjusts `watchers_a.py`, and adds opt-in routing in `nighta.py`. A14 receives the shared fuel-seat/lighting setup to avoid a discontinuity at the cut. Old nighta callers retain their defaults. The watcher uses continuous hand targets, fixed ankle targets and reduced cloth/breath movement; a dark red shoulder wrap sits below the closed cowl. Tests establish continuity and target positions, not full physical support over uneven rock.

A14's `beaconrun_a.py` updates `fr.dist` only where the opaque pass wrote a nearer `fr.zb`: source forward depth becomes Euclidean ray distance before target warp and camera blur. It preserves unchanged terrain/sky values. Intrinsic actor motion and translucent fire/smoke do not acquire their own velocity from this fix.

The compiled RUN suite passes **59 tests** with JIT enabled, two Numba threads and sequential OpenCV. This includes the prior 51 tests plus eight catch/depth tests. In-memory negative controls fail for missing depth synchronization, treating forward depth as ray distance, and the old three-metre near-flare tolerance. Review receipts pin source/input hashes and ordered frame lists. Visual judgments are stated separately from those tests in each review.

Jobs are prepared in `cloud/jobs/watchers_a_hearth3.json` (160 frames) and `beaconrun_a_catches3.json` (320 frames), using separate output folders. The current farm fetches `claude/long-dawn-v2`; the JSON branch field is not a PR-code selector. Merge the accepted source before launching. Director/EDIT must select the new output folders after rendering; no EDL was changed by Codex.

## A13 state to verify on return

At fetched production `742bcae`, the renderer/job/EDLs agree: H1 roar A3600–3611; reveal_B1372–1419 at A3612–3659; reveal_A at A3660–3799. Commits `9613759` and `b1248e5` resolved the earlier 3660/3680 disagreement. Some comments retain 3680; the night ramp starts3660 and the crane starts3680. The newly pushed `review/A_FIX_NOTES.md` (7611bf7) reports that reveal_A3660–3799 has been re-rendered and checked, and that the harvested B shot is grade-matched. These are A-FIX’s reports; Codex has not inspected those new local renders. The older EDL comment still asks for the A3660 grade check, so reconcile it with A-FIX’s render receipts before scheduling any duplicate work. The haze-slab fix `2913ad7` is already present.

No relief approval was recorded in the checked notes or PR reviews. Please approve a specific recipe/strength or request a revision on PR #6 before that dependent render work. GitHub comments and commits are the coordination channel; a posted message is not proof the other session has read it.

## Other work and operations

Handover 2 implementation is complete: PRs #3 and #4 are merged; #5 is the delivered frame-QC tool and #6 the delivered relief study. Open status alone does not make those implementations unfinished. The old PR #2 cairn proposals must not be adopted.

Codex preserved the production checkout, original laptop's uncommitted work, A-FIX, sound and EDIT files. Local review renders used one job at a time, with a supervisor allowed to pause only its own child when the machine was busy. No farm finals were launched. The older export contains credentials and was never included in these commits.

The main review renders have completed. Production had no newer edits to the five night-shot implementation files through `742bcae`, but RUN-A4 changed the shared `sdfppl.py` at `60e2f13` during the render sequence; its compatibility is being checked before the final handback.
