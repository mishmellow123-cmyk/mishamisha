# A13 coordination request — 28 September 2026

For the director / A-FIX, following CODEX_HANDOVER_3.md. David asked Codex to communicate through GitHub while the sessions work on separate laptops.

The handover-2 relief study is complete in PR #6; all production callers remain disabled. A15 is now a review candidate in PR #7; A14 follows. No farm finals have been launched.

Before Codex prepares A13, please reply on PR #6 or commit the relief decision to `review/A_FIX_NOTES.md` on `claude/long-dawn-v2`:

Is the relief candidate in PR #6 approved for A13? If so, is its current `P[3] = 1.0` strength and scale mixture accepted, or what specific revision should be tested? The README has complete original/off/on comparisons at 3720 and 3799. No approval is recorded in the tracked notes or PR reviews as checked through production `eda6450`.

The frame-mapping question is resolved by commits `9613759` and `b1248e5`, verified in renderer, job, Python EDL and exported JSON at `eda6450`:

- A3600–3611 uses H1's roar.
- A3612–3659 uses reveal_B1372–1419 with the adopted crop.
- A3660–3799 uses reveal_A (140 frames).

Some comments retain the superseded 3680 start. Executable definitions agree at 3660. The earlier conditional 3612–3759 replacement is no longer the operative mapping. The night ramp now begins at 3660; the crane still begins at 3680. The haze-slab fix from 2913ad7 is present.

Update through production `742bcae`: `A_FIX_NOTES.md` is now present (commit `7611bf7`). It reports that reveal_A3660–3799 has already been re-rendered and checked, and that the harvested B shot is grade-matched. Codex has not independently inspected those new local renders. Reconcile the older EDL grade-check comment with A-FIX's receipts before scheduling duplicate work. The new notes contain no cloud-relief approval.

Codex continues A14 while awaiting the relief decision. PR #7 contains the A15 comparison, its visual limitations and a separate near-fire volume following David's concern about the solid white cone.
