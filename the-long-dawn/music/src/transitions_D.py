"""Opt-in D transition orchestration; apply only after the complete D build.

The carrier lines connect common tones while named picture attacks retain their
positions. Percussive entries use explicit velocity ramps: a part dynamics curve
alone does not change one-shot attacks with explicit velocities. No donor module
or protected opening PCM is changed here.
"""
from copy import deepcopy

import kit_v3 as K
from dsl import dyn_at

PROFILE = "D-transitions-phase6-v1"


def _factor(points, frame):
    if frame < points[0][0] or frame > points[-1][0]:
        return 1.0
    for (a, low), (b, high) in zip(points, points[1:]):
        if a <= frame <= b:
            u = (frame - a) / (b - a)
            u = u * u * (3 - 2 * u)
            return low + (high - low) * u
    return points[-1][1]


def _expression(S, names, points):
    """Multiply both explicit onset velocities and continuous sampler dynamics.

    Sustained sampler curves normalize to the explicit onset velocity. Scaling
    both gives the intended envelope without cancelling the quiet first attack.
    Synth one-shots here read explicit velocity; keep their dynamics untouched.
    """
    lo, hi = points[0][0], points[-1][0]
    for name in names:
        part = S.parts[name]
        original = list(part.dyn)
        for note in part.notes:
            if note.vel is not None:
                note.vel *= _factor(points, note.start * 20)
        if part.kind != "sampler":
            continue
        frames = {b * 20 for b, _ in original}
        frames.update(f for f, _ in points)
        frames.update(range(int(lo), int(hi) + 1, 4))
        # Anchor the unchanged curve outside this local treatment.
        frames.update((lo - .01, hi + .01))
        part.dyn = [(f / 20, dyn_at(original, f / 20) * _factor(points, f))
                    for f in sorted(frames)]


def _carry_tail(S, names, start, end, *, hold_to=None):
    """Let an existing final chord release across the join, without a new attack."""
    hold_to = hold_to if hold_to is not None else (start + end) / 2
    for name in names:
        part = S.parts[name]
        level = dyn_at(part.dyn, start / 20)
        extended = False
        for note in part.notes:
            if note.start * 20 < start and (note.start + note.dur) * 20 >= start:
                note.dur = max(note.dur, end / 20 - note.start)
                note.kw["rel"] = .12
                extended = True
        if not extended:
            raise ValueError(f"{name}: transition carrier has no existing note at {start}")
        part.dyn = [(b, v) for b, v in part.dyn if b * 20 < start]
        part.dyn.extend(((start / 20, level), (hold_to / 20, level * .94),
                         (end / 20, .02)))


def _bridge(S, stem, voices, first, last, levels):
    names = []
    for i, (seat, pitch) in enumerate(voices):
        name = f"transition_{stem}_{i}"
        if name in S.parts:
            raise ValueError("D transition treatment cannot be applied twice")
        part = S.add(name, seat, gain_db=K.SEATS[seat].get("gain_db", 0) - 3,
                     humanize_ms=0, send=.25)
        part.n(pitch, first / 20, (last - first) / 20, None,
               sync=True, antic=0., fadein=.18, rel=.12)
        part.d(*[(f / 20, level) for f, level in levels])
        names.append(name)
    return names


def _notes(S, names):
    return {name: deepcopy(S.parts[name].to_dict()) for name in names}


def apply(S, bm):
    """Apply the explicit profile and return its reproducible treatment manifest.

    Existing note starts and pitches never move. The added lines are composed
    support, not newly measured picture events. Exact cutd windows remain a
    separate editorial pin; these musical windows may prepare a cut earlier.
    """
    if getattr(S, "transition_manifest", None):
        raise ValueError("D transition treatment cannot be applied twice")
    if (bm.bars, bm.frames) != (115, 9200):
        raise ValueError("D transition treatment requires the fixed v2 grid")
    protected_names = ("crowns_left", "crowns_right", "watch_D")
    protected = _notes(S, protected_names)
    original_attacks = {name: [(n.start, n.pitch) for n in part.notes]
                        for name, part in S.parts.items()}
    silence = deepcopy(S.hard_silences)
    pcm = deepcopy(S.pcm_regions)
    event_frames = {e["id"]: e["frame"] for e in bm.events}
    joins = []

    # The inscription's quiet D pedal becomes the working bass before the
    # forging enters. Its first measured/nominal percussion attacks stay put.
    carriers = _bridge(S, "forging", (("cb_q", "D2"), ("vc_q", "A2"), ("vla_q", "D3")),
                       2000, 2112, [(2000, .035), (2050, .11), (2080, .19), (2100, .14), (2112, .02)])
    _carry_tail(S, ("inscription_bed_cb_q", "inscription_bed_vc_q"), 2068, 2104, hold_to=2088)
    _expression(S, ("race_taiko", "race_anvil", "race_ring", "race_ring_double", "race_low", "race_trem"),
                [(2072, .38), (2080, .38), (2120, .72), (2160, 1.)])
    joins.append(dict(cut=2080, window=[2000, 2160], carriers=carriers,
                      treatment="D/A preparation, carried inscription bass, explicit race attack crescendo"))

    # The exposed line speaks after the gap; soften its attack without prelap.
    refusal = S.parts["refusal_solo"]
    first = next(n for n in refusal.notes if round(n.start * 20) == 3520)
    first.kw["fadein"] = .5  # 12 frames; starts and prescribed silence unchanged
    joins.append(dict(cut=3520, window=[3520, 3532], carriers=["refusal_solo"],
                      treatment="Twelve-frame attack fade on the existing exposed A4; no prelap"))

    cadence = next(n for n in refusal.notes if round(n.start * 20) == 3720)
    cadence.dur = (3786 - 3720) / 20
    # This part had no dynamics; preserve each earlier explicit velocity and
    # release only the last D4 as the burn clears.
    refusal.d((3720 / 20, .6), (3768 / 20, .6), (3786 / 20, .03))
    carriers = _bridge(S, "trap", (("cb_q", "D2"), ("vc_q", "D3")),
                       3750, 3800, [(3750, .035), (3760, .11), (3778, .14), (3800, .02)])
    _expression(S, ("trap_forge", "trap_others", "trap_bass", "trap_taiko", "trap_horns"),
                [(3760, .32), (3786, .75), (3806, 1.)])
    joins.append(dict(cut=3760, window=[3750, 3806], carriers=["refusal_solo"] + carriers,
                      treatment="Cadential D4 carries through burn; low D prelap and explicit trap attack ramp"))

    # The ascending violin's last A stays in the air as the foreground moves
    # away; the same D supports the distant pulse and the later crowns pedal.
    leader = S.parts["trap_leaders"]
    high_a = max(leader.notes, key=lambda n: n.start)
    high_a.dur = (4132 / 20) - high_a.start
    high_a.kw["rel"] = .15
    # Keep the reduction through the sample's release, which ends at 4135.6.
    # Returning to the original curve at the written note end revives its tail.
    _expression(S, ("trap_leaders",), [(4068, 1.), (4092, .75), (4132, .08), (4140, .08)])
    _expression(S, ("trap_forge", "trap_others", "trap_bass", "trap_taiko"),
                [(4040, 1.), (4080, .5)])
    carriers = _bridge(S, "glow", (("vc_q", "D3"), ("vln2_q", "A4")),
                       4068, 4252, [(4068, .035), (4092, .15), (4180, .13), (4240, .1), (4252, .02)])
    joins.append(dict(cut=4080, window=[4040, 4252], carriers=["trap_leaders"] + carriers,
                      treatment="A5 line crosses the dissolve; D/A support links distant pulse to crowns"))

    # Prepare the major third before the second CALL. Its two sources retain
    # identical timing and gain; the first kindle CALL and WATCH stay exact.
    second_call = bm.ev("two_fires")
    for name in ("crowns_left", "crowns_right"):
        notes = [n for n in S.parts[name].notes if second_call <= n.start < second_call + 4]
        expected = [n for n in protected[name]["notes"]
                    if second_call <= n["start"] < second_call + 4]
        if len(notes) != 3 or len(expected) != 3:
            raise ValueError("second CALL requires its existing three-note phrase")
        for note, saved, gain in zip(notes, expected, (-8., -4., 0.)):
            note.gain_db += gain
            saved["gain_db"] += gain
        notes[0].kw["fadein"] = .5
        expected[0]["kw"]["fadein"] = .5
    # The full-part comparison below permits only these exact note-level
    # changes: kindle notes, instrument, seed, pan, sync and durations stay put.
    carriers = _bridge(S, "ridges", (("cb_q", "D2"), ("vc_q", "A2"), ("vln2_q", "F#4")),
                       4500, 4580, [(4500, .035), (4532, .13), (4554, .20), (4566, .18), (4580, .02)])
    _carry_tail(S, ("crowns_pedal",), 4540, 4572, hold_to=4560)
    _expression(S, ("ridges_cb_q", "ridges_vc_q", "ridges_vla_q", "ridges_vln2_q", "working_distant"),
                [(4560, .42), (4580, .72), (4600, 1.)])
    joins.append(dict(cut=4560, window=[4500, 4600], carriers=carriers + ["crowns_pedal"],
                      treatment="D major prepared under crowns; second CALL pair enters equally at -8/-4/0 dB with a twelve-frame first attack; kindle CALL and WATCH unchanged"))

    # Only the inner third changes in each book/ember join. Carry the identical
    # D/A voices and let F-sharp/F exchange without a renewed bass attack.
    _carry_tail(S, ("in_step_cb_q", "in_step_vc_q", "in_step_vln2_q"), 5828, 5860, hold_to=5844)
    _expression(S, ("oldfire_cb_q", "oldfire_vc_q", "oldfire_vln2_q"),
                [(5840, .35), (5860, 1.)])
    _expression(S, ("oldfire_vla_q",), [(5840, .35), (5852, 1.)])
    joins.append(dict(cut=5840, window=[5828, 5860],
                      carriers=["in_step_cb_q", "in_step_vc_q", "in_step_vln2_q"],
                      treatment="D/A/D carries into old story; F-sharp yields to the quiet minor third"))

    _carry_tail(S, ("oldfire_cb_q", "oldfire_vc_q", "oldfire_vln2_q"), 6072, 6120, hold_to=6092)
    _expression(S, ("unfinished_cb_q", "unfinished_vc_q", "unfinished_vln2_q"),
                [(6080, .35), (6100, .75), (6120, 1.)])
    _expression(S, ("unfinished_vla_q",), [(6080, .3), (6104, 1.)])
    joins.append(dict(cut=6080, window=[6072, 6120],
                      carriers=["oldfire_cb_q", "oldfire_vc_q", "oldfire_vln2_q"],
                      treatment="D/A/D carries through burn; major third arrives gradually; incomplete Ring unchanged",
                      picture_window_status="proposed musical window; bind final cutd burn separately"))

    # C's D minor cadence and A's Bbmaj7 share D/F/A. The note excerpt used to
    # retrigger all three at the cut. Supply their sound across the dissolve,
    # retaining every AP2 walk/setoff/harmony marker and original note start.
    donor = ("deep_closed_vla_q", "deep_closed_vln2_q", "deep_closed_bsn_c", "deep_closed_line_vc")
    _carry_tail(S, donor, 6616, 6660, hold_to=6644)
    carriers = _bridge(S, "crossing", (("vla_q", "D4"), ("vln2_q", "F4"), ("vln1_q", "A4")),
                       6628, 6668, [(6628, .035), (6640, .13), (6652, .14), (6668, .02)])
    _expression(S, ("crossing_vln2_q", "crossing_vln1_q", "crossing_call_vlaq"),
                [(6640, .32), (6652, .62), (6680, 1.)])
    _expression(S, ("crossing_cb_pizz_go", "crossing_feet_go", "crossing_cycle"),
                [(6640, .55), (6680, 1.)])
    joins.append(dict(cut=6640, window=[6616, 6680], carriers=list(donor) + carriers,
                      treatment="Shared D/F/A links D minor to Bbmaj7; fixed AP2 setoff attacks soften into texture"))

    for name, attacks in original_attacks.items():
        if [(n.start, n.pitch) for n in S.parts[name].notes] != attacks:
            raise ValueError(f"{name}: transition changed an original attack or pitch")
    if _notes(S, protected_names) != protected:
        raise ValueError("transition changed a CALL pair or the continuous WATCH")
    if S.hard_silences != silence or S.pcm_regions != pcm:
        raise ValueError("transition changed protected silence or opening PCM")
    if {e["id"]: e["frame"] for e in bm.events} != event_frames:
        raise ValueError("transition moved a picture binding")
    manifest = dict(profile=PROFILE, joins=joins, event_frames=event_frames,
                    protected_opening=[0, 1440], examined_unchanged=[1440],
                    musical_threshold="Measure rendered score; no unrendered dB improvement is claimed.",
                    default="This module has no effect unless apply is explicitly invoked.")
    S.transition_manifest = manifest
    return manifest
