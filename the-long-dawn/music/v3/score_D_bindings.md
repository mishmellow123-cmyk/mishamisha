# Cut D score bindings

`score_D_bindings.json` is the editable score table. It includes every committed
editorial event and every score name. An alias has an `anchor`, without another
frame value; edit its canonical row once. `barmap_D.json`, `cues_D.json`,
`events_D_measured.json` and `events_D_round4_measured.json` remain read-only
inputs, pinned by SHA256.

Use the table explicitly:

```sh
"$O/tools/onepy" "$PY" music/src/render_D.py --render --events music/v3/score_D_bindings.json --output-dir "$O/codex/scored/phase3"
"$O/tools/onepy" "$PY" music/src/verify_D.py --bindings music/v3/score_D_bindings.json --wav "$O/codex/scored/phase3/score_D_draft.wav" --opening music/out/v3/sound_AP2_score.wav --receipt "$O/codex/scored/phase3/render_D_receipt.json" --json "$O/codex/scored/phase3/verification_D.json"
```

Omitting the binding flag preserves the Phase 1c composition. Both commands also
accept the other command's flag spelling. Receipts record the table hash and its
editorial source hashes. A changed event requires a full `--render`.

## When a new shot arrives

Measure the event in `renders/<stem>/` using the shot's **D frame numbers**.
Record the actual method and frame asset SHA256. Merely finding the directory
does not supply a measurement. Update one canonical event row:

- `frame`: the measured integer D frame, inside the event's committed timing
  window and its score section. A checked native `render_D` receipt may cross
  the estimate's bar; an unmeasured edit remains confined to that bar. Without
  a committed window, the original bar remains the bound.
- `status`: `measured`.
- `source`, `measured_ref`, `measurement_scope`: the actual measurement and its
  location/scope; distinguish a native image from a proxy or generated mask.
- `measurement`: an object with `kind: "render_D"`, `stem`, `frame`,
  `frame_file` (basename such as `f_04324.png`, carrying the actual frame number),
  `sha256`, and `method`. The basename here illustrates the format; it is not a
  measured kindle frame.

Leave `picture_frame` null in the table. The loader derives it only after checking
the frame receipt and asset bytes. Retain the `editorial` object as the historical
source evidence. Then run the two commands above. Changing a shot's file bytes
invalidates its receipt until the event is measured again.

| Picture event | Edit this row | Score follows |
|---|---|---|
| Crown kindle | `crown_beacons` | `crowns_kindle`, both CALL horns |
| Gap's first strike | `gap_hammer_lands` | `gap_anvil`; the fixed cut remains `gap` |
| Holdout contact | `holdout_first_contact` | `holdout_hammer`, `holdout_anvil` |
| Last beacon's first catch | `last_beacon` | `last_beacon_catch`, horn answer |
| Last beacon fully alight | `all_lit` | D resolution |

Catch and fully-alight are separate observations. Update each if its measured
frame changes. The NEW gap and holdout rows enforce their specified render stems;
the reused map can receive a replacement render measurement with its actual stem.
The fixture in `music/tests/fixtures/d_binding_render_measurements.json` contains
synthetic measurements used only by tests. It proves that the canonical row edit
moves the score's actual notes, and that a missing/changed asset or unsupported
measurement claim fails.

## Scope retained from cutd

AP2's opening remains inherited **pass 1**, with its delivered PCM protected.
C's measured timings retain their source offsets. The drawn-ring burn's
D1683/D1748 rows measure generated cover at 480×201, not finished native frames;
the pedal starts on the fixed D1680 cut. AP2's measured full walking speed at
D6736 is distinct from its authored step/harmony at D6740. The voice-space
markers describe a composed reservation, with no approved private voice placement.

Initial Phase 2 reconciliation adopts cutd's estimates for inscription stop,
trap changes, letters illuminated by lamps and the last-leaf marker. Native
measurements may replace those estimates within the committed shot windows;
the bar grid and section boundaries remain fixed. New score-only marker rows
need an explicit `baseline_frame` and `timing_window` to move across a bar, plus
the same checked native receipt. They cannot leave the baseline's score section.
If the owner changes the music editorial JSON again, reconcile its table and
review the source hashes before using these bindings; a stale table fails
instead of silently reading new inputs. EDL-only revisions use the explicit
re-pin preflight below.

Round 4 (`3858589`) supplies two distinct Deep exit measurements: native source
mask opening at D2953, and the runtime tail's same-frame equality to the incoming
Brink at D2980 **before finish**. The selected source masks never clear fully;
their clearance and visible-glow onset remain null. The ridge-to-map burn opens
at D5183 and clears at D5212 on the 480×201 generated cover grid. D5182 is an
engine parameter. The loader checks the canonical references against receipt
leaves and checks those summaries against the recorded samples.

These observations leave the composed Brink entry at D2960 and map entry at
D5180. Neither entry is aliased to a cover event. A cover threshold cannot
silently become a finished-picture accent. The Deep burn-through row no longer
requires the raw `embers_D_brink` render stem: that plate does not contain the
exit transition whose cover the row measures.

## Phase 3 native deliveries

The table now carries native JPEG receipts for crown ignition, gap impact,
holdout cap response, the inscription turn, the vision weld/push, the trap,
selected forging/race/Brink pulses, five ridge catches, the old Ring's fall and
flare, and the first solar rim. Each row states its own operational criterion;
a minimum-motion frame or thresholded crest rise is not a claim of exact
subpixel physical contact. The unfinished lamps remain estimated because their
illumination is continuous and the combined event has no isolated first frame.

`old_fire_ring_falls` and the new `old_fire_letters_flare`,
`old_fire_letters_full`, `old_fire_letters_out` are separate picture observations.
Their measurement does not add accents to the existing old-story phrase.
`trap_surge` now anchors `trap_neighbour_rise`: the pixels establish a crest rise,
not the distinct editorial event of gold redistribution. The four-note surge
phrase shortens when necessary to finish before the return anchor; its pitches
and number of notes are retained.

Five added ridge fires are measured. The legacy `beacon_catch_06` name remains
for an authored closing response; it does not assert a sixth added fire. The
watch's three harmonic changes are authored phrase markers, and the last-leaf
marker is musical: the delivered leaf has static lettering.

An original giant-stroke accent can carry `score_action: "omit"` and an
`omission_reason`. For an unsupported strike, keep the event estimated and place
the inspected frame receipt in `absence_evidence` (source, reference, scope and
`measurement` in the same render-receipt format). The forging omits the five
unsupported accents at D2080/2120/2160/2200/2360. No onset is invented for them.
The remaining forging accents retain their pitches and measured frames.

The Brink's pulses remain measured observations, but their glass accents are
omitted under `omission_kind: "no_gap_narrowing"`. The treatment assigns glass
to narrowing; the renderer's `state()`/`theta_range()` hold the gap at 55 degrees
through this shot. This is source geometry, not an image-space angle measurement.
The separate absence evidence pins `shots/embers/d_vision.py` by hash, in addition
to the native pulse frame. Taiko, anvil and roll continue. The loader restricts
this decision to the Brink's original glass markers.

The renderer now records the exported WAV's SHA256. Verification compares that
hash, the score-driving event frames and play/omit actions, and the implementation
hashes. An old WAV cannot acquire new musical timings through a new table. A
metadata-only promotion at unchanged musical frames may retain the audio; the
verification output exposes both the render's table provenance and the current
table provenance. Run verification with the render receipt. Changed musical
bindings or changed implementation require a full render.

## Phase 4 editorial re-pin

After the owner adopts an EDL-only revision, run this **one command** from the
repository root (`O` and `PY` are the same private output root and interpreter
used above):

```sh
CUTD_LOCAL_RENDERS="$O/codex/cutd/r5/local_renders" "$O/tools/onepy" "$PY" music/src/repin_D.py --write --receipt "$O/codex/scored/phase4/editorial_repin.json"
```

The command validates native frame receipts, checks the exported D EDL against
the production `assemble.edl_doc("D")` result, and verifies the four music JSON
pins, canonical evidence, measurement summaries, section grid and treatment
entries. It then changes only the table's `edit_revision` field; the `events`
object remains byte-identical, including aliases and measurement scopes. A
simultaneous table edit aborts the write. The receipt records the old and new
generator/export hashes with repository-relative names.

The first pin records the reviewed adoption's shot and source mapping. Subsequent
re-pins reject changes to shot bounds, source stems, offsets, holds, need gates,
baked-caption declarations or protected transitions. The sole transition
exception is the single D31-to-D32 transition at cut D8640, which must span that
boundary and remain within D8320–8879. Other transitions inside those shots stay
protected. This exception permits the announced page-turn replacement; it does
not measure that transition or caption 14's future write-on. A changed music
JSON or protected mapping requires explicit binding review; the command cannot
silently bless it by refreshing hashes.

Before rendering or verifying against a pinned edit, use the read-only preflight:

```sh
CUTD_LOCAL_RENDERS="$O/codex/cutd/r5/local_renders" "$O/tools/onepy" "$PY" music/src/repin_D.py --check
```

`music/tests/test_d_repin.py` runs this production check and negative controls
for stale exports/pins, altered source clocks, moved cuts, false receipt
summaries and unrelated transitions inside the exception's frame window. The
re-pin is a separate editorial audit; it never changes a render receipt, audio
hash or measured musical timing. An unchanged musical table can retain the
Phase 3 WAV and its original provenance. Re-render only when a musical timing
moves by more than one frame, as required for Phase 4.

## Phase 5 picture lock

At `8fc48c4`, D31-to-D32 is a dissolve over D8628–8651 (half-open window
`[8628, 8652)`, cut D8640). This fits the existing transition exception; the
protected shot/source signature is unchanged. The current Deep composites use
the round 7 local-render root.

Re-measure each changed race or old-fire observation on its replacement JPEGs
and update that row's frame, status, scope and asset receipt through the one-table
path above. Then validate **all native picture receipts and the editorial revision
in one command**:

```sh
CUTD_LOCAL_RENDERS="$O/codex/cutd/r7/local_renders" "$O/tools/onepy" "$PY" music/src/repin_D.py --write --receipt "$O/codex/scored/phase5/editorial_repin.json"
```

The picture receipts are already part of the table hash recorded by this command.
It verifies their actual asset bytes before updating editorial pins; it cannot
refresh a stale JPEG hash, infer an onset or turn an estimate into a measurement.
A changed frame set therefore requires new observation evidence even when the
measured frame number stays the same. Use the same command with `--check` in
place of `--write` for subsequent read-only preflight.

A retained source-clock cue can remain `status: "estimated"` with
`role: "source_clock"` when its nominal frame has no discrete optical onset.
Store the inspected snapshot under `frame_evidence` (its own `frame`, `source`,
`measured_ref`, `measurement_scope` and native `measurement` receipt), and pin
the clock's `shots/embers/c_d.py` source bytes under `clock_source` (`path` and
`sha256`). The re-pin command validates that snapshot and code pin separately;
it leaves the estimated row and its null `picture_frame` untouched. These
fields require the explicit source-clock role and cannot substitute for a
measured-onset receipt. Missing or changed snapshot/source bytes fail preflight.

## Phase 6 reviewed picture transitions

The ordinary command retains all previous guards. A separate, explicit path can
adopt reviewed transition changes at cuts **4080, 4240, 4560, 5840, 6080 and 6640**.
It cannot change shot boundaries, take/source mappings, the four music JSONs,
measured event rows, or any other protected transition. The existing D8640
exception is unchanged.

After cutd's final windows have been inspected, write a private JSON approval
with schema `long-dawn/score-D-reviewed-transitions/1` and a `changes` array.
Each entry has exactly `cut`, `before`, and `after`. Both sides are lists of
zero or one transition record: `before` must equal the record in the table's
current `edit_revision.protected_signature.protected_transitions`; `after`
must equal the final generated D EDL record. Use every field except the human
`note`, as the existing protected signature does. An empty list means a hard
cut. Include every changed protected join and no unchanged joins. Do not fill
this approval from proposed windows while the final picture is pending.

Run the adoption in one command (`CUTD_LOCAL_RENDERS` must name the final
reviewed local-render namespace):

```sh
"$O/tools/onepy" "$PY" music/src/repin_D.py --write --reviewed-transitions "$REVIEW" --receipt "$O/codex/scored/phase6/editorial_repin.json"
```

The command checks the old and new records independently before replacing the
protected signature. Missing approval, a stale old record, a mismatched new
record, an unlisted cut, a duplicate, or an unapproved transition change fails.
Reviewed windows must stay within the adjacent fixed shots. The private
approval's normalized hash and exact records are recorded under
`edit_revision.transition_review`; no workstation path is copied into the table.
The event object stays byte-identical. Existing picture receipts are validated
first, so this command cannot bless stale native frame hashes.

After adoption use ordinary `--check` without the approval. Reusing the same
approval against the new baseline fails because its `before` records no longer
match. The approval authorizes editorial revision only; it supplies no measured
picture onset, changes no score notes, and cannot relabel an old WAV as a new
render. Synthetic tests exercise the command with deliberately fictional
transition windows; those fixtures are not final picture decisions.

## Phase 6 score activation

The musical transition treatment is opt-in. Without `--transition-pass`, the
default score identity is preserved exactly. With the flag, `transitions_D.py`
adds sustained links and adjusts attack/release envelopes around the joins;
existing picture hits retain their bound frames. Its musical windows include
preparation before the cut and can be broader than the picture transition.
They do not prescribe cutd's final EDL windows, which remain pending owner
adoption.

Render into a new private directory, then verify with the same flag and binding
table:

```sh
"$O/tools/onepy" "$PY" music/src/render_D.py --render --transition-pass --bindings music/v3/score_D_bindings.json --opening-source music/out/v3/sound_AP2_score.wav --output-dir "$O/codex/scored/phase6/render"
"$O/tools/onepy" "$PY" music/src/verify_D.py --transition-pass --bindings music/v3/score_D_bindings.json --wav "$O/codex/scored/phase6/render/score_D_draft.wav" --opening music/out/v3/sound_AP2_score.wav --receipt "$O/codex/scored/phase6/render/render_D_receipt.json" --json "$O/codex/scored/phase6/render/verification_D.json"
```

The new render receipt pins the activation flag, treatment manifest and
`transitions_D.py` hash. Verification checks the option and implementation
against that receipt; the retained Phase 3 WAV cannot serve as evidence for
the activated treatment.
