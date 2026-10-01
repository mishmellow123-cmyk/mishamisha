"""THE LONG DAWN, cut D, treatment v2: 115 bars at 72 BPM.

Phase 1c is a complete composition on a provisional picture-event table.  Every
picture hit resolves through ``bm.ev(name)``; binding measured event frames must
not require recomposition.  ``barmap_D_score`` is a separate opt-in draft map;
the cutd-owned JSON files and every A/C score stay untouched.

The first 1,440 frames are replaced AFTER mastering by AP2's delivered score
PCM (including ignition polish).  Donor note material thereafter is explicitly
reorchestrated/re-rendered, not falsely described as byte-identical audio.
The two complete old-story RING windows are exceptions to the rule for OUR
Ring: ours completes once inside the vision and otherwise lacks its fifth note.
The cor-anglais role uses the available low oboe; stopped brass uses EQ'd horn.
No caption is sung.  Caption authority is D_TREATMENT_v2, including caption 8,
"But whoever won the race, the Ring would rule us all."
"""
from copy import deepcopy
from functools import lru_cache

import numpy as np

import kit_v3 as K
import score_v3_A as A
import score_v3_AP2 as AP2
import score_v3_C as C
import score_v3_C5 as C5
import score_v3_C5P2 as C5P2
from dsl import dyn_at, m
from sound_recipes_D import opening_region
from timeline_v3 import BEAT_S, BarMap

CALL_PARTS = ("crowns_left", "crowns_right")
WATCH_PART = "watch_D"
TRAP_VOICE = "trap_forge"
RING_INTERVALS = tuple(K.RING)
AUDIBLE_DB = -120.0


def _part(S, name, seat, **kw):
    if name not in S.parts:
        S.add(name, seat, **kw)
    return S.parts[name]


def _bind(S, eid, part, label=None, kind="hit", offset=0.0):
    b = S.bm.ev(eid) + offset
    S.picture_bindings.append(dict(event=eid, part=part, beat=b, offset_beats=offset,
                                   status=S.bm.event(eid).get("status", "provisional")))
    S.sync.append((b * BEAT_S, label or eid.replace("_", " "), part, 0.25, kind))
    return b


def _line(p, pitches, durations, start, level, **kw):
    out, t = [], start
    for i, (pitch, dur) in enumerate(zip(pitches, durations)):
        n = p.n(pitch, t, dur, level, legato=i > 0, sync=True, antic=0.0, **kw)
        out.append(n)
        t += dur
    return out


def _phrase(S, name, part, root, start, durations, complete=False, context="ours", event=None, level=.4):
    intervals = K.RING if complete else K.RING[:4]
    notes = _line(S.parts[part], [m(root) + x for x in intervals], durations, start, level)
    S.motif_phrases.append(dict(id=name, motif="RING", context=context, root=m(root), complete=complete,
                               event=event, notes=[dict(part=part, start=n.start, pitch=n.pitch) for n in notes]))
    return notes


def _chords(S, stem, seq, end, level=.2, quiet=True):
    """Four independently bowed voices; seq is (beat, [bass,cello,viola,violin])."""
    seats = ("cb_q", "vc_q", "vla_q", "vln2_q") if quiet else ("cb", "vc", "vla", "vln2")
    for k, seat in enumerate(seats):
        pn = stem + "_" + seat
        p = _part(S, pn, seat, gain_db=K.SEATS[seat].get("gain_db", 0) - 3.0)
        for i, (t, pitches) in enumerate(seq):
            t1 = seq[i + 1][0] if i + 1 < len(seq) else end
            if t1 <= t:
                raise ValueError(f"unordered chord progression in {stem}")
            K.held(S, pn, pitches[k], t, t1, bow=8, vel=level, sync_first=True, antic=0.0)
        p.d((seq[0][0], level), (end - .3, level * .82), (end, .03))


@lru_cache(maxsize=1)
def _donors():
    """Source builds only, with their current ride and anticipation; never render."""
    return AP2.build(BarMap("AP2")), C5P2.build(BarMap("C5P2"))


def _excerpt(S, donor, f0, f1, target, prefix, skip=(), gain_db=0., named=None):
    """Copy sounding notation and expression, preserving source timbre parameters.

    Existing sustains are clipped at the entry; no later donor onset is imported.
    Notes remain musical material, not a claim of identical sampler random history.
    An optional named-event map rebinds source starts to D's independent events.
    """
    a, z = f0 / 20., f1 / 20.
    shift = target - a
    mapping = dict(named or {})
    for pn, p in donor.used().items():
        if pn in skip:
            continue
        ns = [n for n in p.notes if n.pitch is not None and n.start < z - 1e-8
              and n.start + n.dur > a + 1e-8 and n.gain_db > AUDIBLE_DB]
        if not ns:
            continue
        q = deepcopy(p)
        q.name = prefix + "_" + pn
        q.gain_db += gain_db
        q.notes = []
        q.dyn = [(a + shift, dyn_at(p.dyn, a))] + [(b + shift, v) for b, v in p.dyn if a < b < z]
        q.dyn.append((z + shift, dyn_at(p.dyn, z)))
        for n in ns:
            nn = deepcopy(n)
            source_start = max(a, n.start)
            nn.start = source_start + shift
            nn.dur = min(z, n.start + n.dur) - source_start
            nn.kw["donor_start_beat"] = n.start
            nn.kw["donor_cut"] = donor.cut
            if n.start < a:
                nn.legato = False
                nn.kw["antic"] = 0.0
                nn.sync = True
            for source_frame, event in mapping.items():
                if abs(n.start * 20 - source_frame) < 1e-5:
                    delta = S.bm.ev(event) - nn.start
                    nn.start += delta
                    nn.dur = max(.05, nn.dur - delta)
                    nn.sync = True
                    _bind(S, event, q.name, kind="soft")
            q.notes.append(nn)
        S.parts[q.name] = q
        if pn in donor.eq:
            S.eq[q.name] = deepcopy(donor.eq[pn])
    S.reuse.append(dict(source=donor.cut, source_f0=f0, source_f1=f1,
                        destination_f0=round(target * 20), prefix=prefix, method="donor_notes"))


def _opening_tail(S, donor):
    """Only already-playing opening voices may release after the PCM boundary."""
    edge = S.bm.ev("mountain")
    for pn, p in donor.used().items():
        ns = [n for n in p.notes if n.start < edge and n.start + n.dur > edge - 1.0]
        # Ignition's bowed voices only; glass decays already live in the PCM.
        if not ns or pn not in ("cb", "vc", "vla", "vln1", "vln2", "prom_vc", "prom_vla"):
            continue
        q = deepcopy(p)
        q.name = "opening_release_" + pn
        q.notes = deepcopy(ns)
        q.dyn = [(b, v) for b, v in p.dyn if b <= edge] + [(edge + .6, .02)]
        for n in q.notes:
            n.dur = min(n.dur, edge + .5 - n.start)
            n.kw["rel"] = .12
        S.parts[q.name] = q


def _old_story(S, source):
    ev = S.bm.ev
    _excerpt(S, source, 320, 560, ev("mountain"), "oldpage",
             named={360: "mountain_ring"})
    # Complete loops are inherited from C; record the two attacked phrases.
    for pn in ("oldpage_hn_st", "oldpage_hn_st2"):
        p = S.parts[pn]
        for k in range(2):
            start = ev("mountain_ring") + k * 4
            ns = [n for n in p.notes if start - 1e-6 <= n.start < start + 4 - 1e-6]
            S.motif_phrases.append(dict(id=f"oldpage_{pn}_{k}", motif="RING", context="old",
                root=m("D4"), complete=True, event="mountain_ring",
                notes=[dict(part=pn, start=n.start, pitch=n.pitch) for n in ns]))
    # Burn carries only a low pedal and a rubbed harmonic; no new loop.
    _chords(S, "burn", [(ev("old_burn"), ["D2", "Ab2", "D3", "F4"])], ev("inscription"), .13)


def _inscription(S):
    ev = S.bm.ev
    pn = "inscription_ring"
    _part(S, pn, "vc_q", gain_db=3.0, send=.3)
    start, stop = ev("inscription"), ev("inscription_stop")
    unit = (stop - start) / 3.5
    _phrase(S, "inscription", pn, "D3", start, [unit, unit, unit, .5 * unit],
            event="inscription_stop", level=.32)
    _bind(S, "inscription_stop", pn, "the inscription stops with the fifth RING note absent", "soft")
    _chords(S, "inscription_bed", [(start, ["D2", "A2", "F3", "C4"])], ev("forging"), .14)


def _race(S):
    ev = S.bm.ev
    start, deep, brink, end = (ev(x) for x in ("forging", "deep", "brink", "vision"))
    _part(S, "race_taiko", "dr_taiko", gain_db=4.0)
    _part(S, "race_ring", "vc", gain_db=1.0, send=.28)
    _part(S, "race_ring_double", "vla", gain_db=-3.0, send=.28)
    _part(S, "race_low", "cb", gain_db=0.)
    _part(S, "race_trem", "vla_trem", gain_db=-4.)
    _part(S, "race_anvil", "C.anvil", gain_db=-13.)
    _part(S, "giant_glass", "C.glitter", gain_db=-11.)
    _part(S, "brink_roll", "timp_roll", gain_db=2.0)
    # Rhythmic escalation is capped at sixteenths; density and register carry the later rise.
    t, k = start, 0
    while t < end - 1e-6:
        elapsed = t - start
        step = 1.0 if elapsed < 8 else .5 if elapsed < 16 else .25
        if deep <= t < brink:
            step = 1.0  # the imported Deep tick owns the dividing pulse
        intensity = .24 + .21 * (t - start) / (end - start)
        S.parts["race_taiko"].n(60, t, min(step, .8), intensity * (1 if k % 4 == 0 else .69),
                                sync=True, antic=0., maxlen=.55)
        if k % max(1, round(1 / step)) == 0:
            S.parts["race_anvil"].n("D4", t, .25, intensity * .4, size=.35, sync=True)
        t += step
        k += 1
    roots = ("D3", "E3", "F3", "G3", "A3", "Bb3", "C4")
    for i, t in enumerate(np.arange(start, end - .1, 8.)):
        if deep <= t < brink:
            continue
        root = roots[min(i, len(roots) - 1)]
        for pn, octave in (("race_ring", 0), ("race_ring_double", 12)):
            _phrase(S, f"race_{i}_{pn}", pn, m(root) + octave, float(t), [1.5, 1.5, 1.5, 2.0],
                    level=.28 + .026 * i)
    K.held(S, "race_low", "D2", start, end, bow=8, vel=.3, sync_first=True, rel=.12)
    for t in np.arange(start, end, 8.):
        for pitch in ("D4", "Ab4"):
            S.parts["race_trem"].n(pitch, t, min(8., end-t), .17 + .15*(t-start)/(end-start),
                                     sync=True, rel=.12)
    for e in S.bm.events:
        if "giant_stroke" in e.get("tags", []):
            if e.get("score_action") == "omit":
                continue  # native absence evidence is checked by BoundMap
            if deep <= S.bm.ev(e["id"]) < brink:
                continue  # parchment Deep has its tick, no visible giant stroke
            label=("the giants continue striking against the held gap"
                   if S.bm.ev(e["id"]) >= brink else "a giant's stroke reaches the gap")
            t = _bind(S, e["id"], "giant_glass", label, "hit")
            # Preserve each event's pitch if an unsupported earlier hit is
            # omitted; indexing the surviving notes would flip later pitches.
            S.parts["giant_glass"].n("D6" if int(e["id"].rsplit("_",1)[1]) % 2 else "Ab5",
                                     t, .25, .19, sync=True, decay=.22)
    K.held(S, "brink_roll", "D2", brink, end, bow=8, vel=.34, sync_first=True, rel=.1)
    S.parts["brink_roll"].d((brink,.16),(end-.8,.52),(end,.08))
    _bind(S,"forging","race_taiko","the towers begin the competitive pulse")
    _bind(S,"brink","brink_roll","three bars gathering toward the conditional vision","soft")


def _vision_gap_refusal(S):
    ev = S.bm.ev
    pn = "vision_stopped"
    _part(S,pn,"C.hn_st",gain_db=3.8,send=.28,humanize_ms=0)
    S.eq[pn]=(300.,None,6.,1500.)
    finish = ev("vision_complete")
    start = ev("vision")
    unit = (finish-start)/3.5
    _phrase(S,"vision_once",pn,"D4",start,[unit,unit,unit,.5*unit,.65],
            complete=True,context="vision",event="vision_complete",level=.66)
    for n in S.parts[pn].notes:
        n.kw["rel"] = .12
    _bind(S,"vision_complete",pn,"the withheld fifth note, once, inside IF IT CLOSED","soft")
    _chords(S,"vision_floor",[(start,["D2","Ab2","D3","C4"])],finish+.65,.42,quiet=False)
    _part(S,"gap_anvil","C.anvil",gain_db=-4.,send=.2)
    # The cut and its preceding silence stay fixed. A measured contact can move
    # the anvil within the new shot without moving either editorial boundary.
    try:
        S.bm.event("gap_hammer_lands")
        gap_hit="gap_hammer_lands"
    except KeyError:
        gap_hit="gap"  # preserve the phase 1c / legacy override map
    S.parts["gap_anvil"].n("D4",_bind(S,gap_hit,"gap_anvil"),.3,.74,size=.75,sync=True)
    p = _part(S,"refusal_solo","C.ca",gain_db=2.,send=.22,humanize_ms=0)
    S.eq[p.name]=(None,3600.)
    # One exposed line. The C# is the dominant's leading tone; its final D is V-i.
    a,turn,cad,end=(ev(x) for x in ("refusal","refusal_turn","refusal_cadence","trap"))
    _line(p,["A4","G4","F4"],[2.,2.,turn-a-4.],a,.32,rel=.12)
    _line(p,["E4","A3","C#4"],[1.,1.,cad-turn-2.],turn,.29,rel=.12)
    p.n("D4",cad,max(.1,end-cad-.35),.27,sync=True,antic=0.,rel=.12)
    _bind(S,"refusal_cadence",p.name,"one low-oboe line closes V-i","soft")


def _trap_and_glow(S):
    ev=S.bm.ev
    start,drop,surge,back,climb,end=(ev(x) for x in ("trap","trap_drop","trap_surge","trap_return","leaders_climb","glow"))
    for pn,seat,gain in ((TRAP_VOICE,"C.vc_sp",0.),("trap_others","C.vla_sp",-2.),
                         ("trap_bass","C.cb_sp",-1.),("trap_taiko","dr_taiko",2.),
                         ("trap_horns","tbn",0.),("trap_leaders","vln1",-1.)):
        _part(S,pn,seat,gain_db=gain,humanize_ms=0)
    pat=["D3","D3","D3","Ab3","D3","D3","Ab3","D3"]
    for k,t in enumerate(np.arange(start,end,.5)):
        v=.29 + .14*(t-start)/(end-start)+(.07 if t>=surge else 0.)
        S.parts["trap_others"].n(m(pat[k%8])+12,t,.44,v*.78,sync=True,antic=0.,maxlen=.32)
        if k%2==0:
            S.parts["trap_bass"].n("D2",t,.5,v*.75,sync=True,antic=0.,maxlen=.42)
            S.parts["trap_taiko"].n(60,t,.5,v*.8,sync=True,antic=0.,maxlen=.55)
    # This single forge has its own exact event boundaries: an off-grid measured
    # sink must cut the last note, and an off-grid return must start a new note.
    for lo,hi in ((start,drop),(back,end)):
        for t in np.arange(lo,hi,.5):
            k=round((t-start)/.5)
            v=.29+.14*(t-start)/(end-start)+(.07 if t>=surge else 0.)
            remaining=(hi-t)*BEAT_S
            S.parts[TRAP_VOICE].n(pat[k%8],t,min(.44,hi-t),v,sync=True,antic=0.,
                                    maxlen=min(.32,remaining),rel=0.)
    for eid in ("trap_drop","trap_return"):
        _bind(S,eid,TRAP_VOICE,kind="soft")
    for t,root,next_change in ((start,"D3",surge),(surge,"Eb3",back),(back,"E3",end)):
        if t+3.5<end:
            # Native bindings can bring two changes closer than the original
            # 70-frame phrase. Fit its same four notes before the next change.
            scale=min(1.,(next_change-t)/3.5)
            if scale<=0:
                raise ValueError("trap changes must remain ordered")
            _phrase(S,f"trap_{t}","trap_horns",root,t,
                    [scale,scale,scale,.5*scale],level=.41)
    _bind(S,"trap_surge","trap_horns","the neighboring forge surges","soft")
    climb_ns=["D5","Eb5","F5","Ab5","A5"]
    step=(end-climb)/5
    _line(S.parts["trap_leaders"],climb_ns,[step]*5,climb,.39,rel=.12)
    _bind(S,"leaders_climb","trap_leaders",kind="soft")
    p=_part(S,"glow_taiko","dr_taiko",gain_db=-3.,send=.5,depth=.85,humanize_ms=0)
    S.eq[p.name]=(None,680.)
    for t in np.arange(end,ev("crowns"),1.):
        p.n(60,t,.65,.27,sync=True,antic=0.,maxlen=.7)
    _bind(S,"glow",p.name,"the same pulse across the valley")


def _paired_call(S,eid):
    t=S.bm.ev(eid)
    for pn,pan in zip(CALL_PARTS,(-1.,1.)):
        p=_part(S,pn,"hn",pan=pan,width=0.,depth=.35,send=.24,gain_db=-.5,humanize_ms=0,seed=7401)
        _line(p,["D4","A4","D5"],[1.,1.,2.5],t,.44,rel=.25)
        _bind(S,eid,pn,"the same CALL, both crowns together","soft")


def _promise_and_ridges(S):
    ev=S.bm.ev
    _part(S,"crowns_pedal","vc_q",gain_db=0.)
    K.held(S,"crowns_pedal","D3",ev("crowns"),ev("two_fires"),bow=8,vel=.19,sync_first=True)
    _paired_call(S,"crowns_kindle")
    _paired_call(S,"two_fires")
    _part(S,WATCH_PART,"vc_q",gain_db=-1.,send=.3,humanize_ms=0)
    K.held(S,WATCH_PART,"D3",ev("two_fires"),ev("terraces"),bow=8,overlap=.16,vel=.16,sync_first=True,antic=0.)
    S.parts[WATCH_PART].d((ev("two_fires"),.16),(ev("watch"),.15),(ev("sunrise"),.17),(ev("terraces"),.14))
    catches=[e for e in S.bm.events if "beacon_catch" in e.get("tags",[])]
    for i,e in enumerate(catches):
        pn=f"ridge_answer_{i%3}"
        _part(S,pn,("hn","hn_far","hn_farther")[i%3],gain_db=-2.-i*.35,humanize_ms=0)
        label=("an authored closing ANSWER follows the five ridge catches"
               if e.get("role")=="composed_response" else "a new fire receives an ANSWER")
        t=_bind(S,e["id"],pn,label,"soft")
        # Two-beat answers leave each 40-frame catch its own entry.
        K.answer(S,pn,"D5",t,mode="major",rhythm=(.5,.25,.25,1.),vel=.29,legato=True,sync_first=True,antic=0.)
    p=_part(S,"working_distant","dr_taiko",gain_db=-11.,depth=.9,send=.5,humanize_ms=0)
    S.eq[p.name]=(None,680.)
    for t in np.arange(ev("two_fires"),ev("map"),2.):
        p.n(60,t,.65,.23,sync=True,antic=0.,maxlen=.65)
    _chords(S,"ridges",[(ev("two_fires"),["D2","A2","D3","F#4"]),
                         (ev("ridge"),["Bb1","F3","D4","A4"])],ev("map"),.17)


def _map_and_instep(S):
    ev=S.bm.ev
    start,dark,hammer,catch,lit,step,end=(ev(x) for x in
         ("map","map_dark","holdout_hammer","last_beacon_catch","all_lit","in_step","old_fire"))
    # C5P2's progression, with the holdout's duration set by D events.
    seq=[(start,["F2","C3","A3","F4"]),(start+2,["D2","A2","F3","A4"]),
         (start+4,["C2","G2","E3","G4"]),(start+6,["A1","F2","C3","A4"]),
         (dark-4,["G1","D2","Bb3","D4"]),(dark-2,["Eb2","Bb2","G3","G4"]),
         (dark,["A1","E3","G3","D4"]),(lit,["D2","A2","F#3","D4"])]
    _chords(S,"map",seq,step,.25)
    p=_part(S,"map_oboe","ob",gain_db=-1.,humanize_ms=0)
    K.answer(S,p.name,"F5",start,mode="major",vel=.26,sync_first=True,antic=0.)
    _bind(S,"map",p.name,"the map receives the previous answer in F","soft")
    _part(S,"holdout_anvil","C.anvil",gain_db=-7.,send=.19)
    S.parts["holdout_anvil"].n("D4",_bind(S,"holdout_hammer","holdout_anvil"),.5,.49,size=.65,sync=True)
    _part(S,"map_last_answer","hn_far",gain_db=-2.,humanize_ms=0)
    K.answer(S,"map_last_answer","D5",catch,vel=.28,sync_first=True,antic=0.)
    _bind(S,"last_beacon_catch","map_last_answer","the holdout catches: answer, then D","soft")
    _bind(S,"all_lit","map_vla_q","A7sus4 resolves into D on the full catch","soft")
    for i,e in enumerate(e for e in S.bm.events if "map_beacon_catch" in e.get("tags",[])
                         and e["id"] != "last_beacon_catch"):
        pn=f"map_reply_{i}"
        _part(S,pn,"hn_far",gain_db=-5.-i,humanize_ms=0)
        t=_bind(S,e["id"],pn,"a map kingdom answers","soft")
        K.answer(S,pn,"F4" if i%2 else "C5",t,vel=.25,sync_first=True,antic=0.)
    _chords(S,"in_step",[(step,["D2","A2","F#3","D4"])],end,.2)
    _part(S,"in_step_cycle","warm",gain_db=-15.)
    K.thinking(S,"in_step_cycle",["D4","A4","D5","E5"],step,end,step=1.,v0=.12,pan_amp=.12)
    _part(S,"in_step_anvil","C.anvil",gain_db=-14.,send=.25)
    _part(S,"in_step_feet","feet",gain_db=-4.,humanize_ms=0)
    for t in np.arange(step,end,4.):
        S.parts["in_step_anvil"].n("D4",t,.5,.26,size=.35,sync=True)
        S.parts["in_step_feet"].n(60,t,1.,.24,sync=True,antic=0.,maxlen=.7)
    # These are phrase breaths in active parts, never an all-stem mute.
    for pn in ("in_step_cycle","in_step_anvil","in_step_feet"):
        p=S.parts[pn]
        p.d(*[(t+off,val) for t in np.arange(step,end,4.) for off,val in ((0.,.28),(3.3,.22),(3.9,.13))])
    _bind(S,"in_step","in_step_anvil","the working strokes agree on one bar")


def _open_ring(S,source):
    ev=S.bm.ev
    _part(S,"oldfire_ring","vc",gain_db=1.,send=.24)
    _phrase(S,"oldfire_complete","oldfire_ring","D3",ev("old_fire")+1.,[2.,2.,2.,1.,1.],
            complete=True,context="old",event="old_fire",level=.35)
    _chords(S,"oldfire",[(ev("old_fire"),["D2","A2","F3","D4"])],ev("unfinished"),.14)
    pn="unfinished_ring"
    _part(S,pn,"vla_q",gain_db=3.,send=.28)
    _phrase(S,"unfinished_rest",pn,"E3",ev("unfinished")+1.,[2.,2.,2.,5.],event="lamps",level=.29)
    _chords(S,"unfinished",[(ev("unfinished"),["D2","A2","F#3","D4"])],ev("deep_abandoned"),.18)
    _part(S,"lamps_harp","harp",gain_db=-1.)
    t=_bind(S,"lamps","lamps_harp","small lamps reveal the open band's letters","soft")
    for k,pitch in enumerate(("D4","A4","E5")):
        S.parts["lamps_harp"].n(pitch,t+k*.5,2.,.16,sync=True)
    _excerpt(S,source,4240,4480,ev("deep_abandoned"),"deep_closed",
             named={4440:"deep_cadence"})


def _crossing_and_watch(S,source):
    ev=S.bm.ev
    # Cutd Round 3's delivered D26 owns this source choice. The picture uses an
    # eight-frame dissolve in its original framing; it is not a spatial lantern
    # match. AP2's set-off, two full-pace steps, narrow-path rest and resumed walk
    # now all belong to the selected excerpt. WATCH remains the same D-owned part.
    _excerpt(S,source,5180,5580,ev("crossing_lantern"),"crossing",skip=("watch",),
             named={5180:"crossing_walk_setoff",5240:"crossing_walk_second",
                    5280:"crossing_walk_full1",5320:"crossing_walk_full2",
                    5360:"crossing_narrow_start",5520:"crossing_walk_resume",
                    5560:"crossing_resume_second"})
    crossing_ids=("crossing_lantern","crossing_walk_setoff","crossing_walk_second",
                  "crossing_walk_full1","crossing_walk_full2","crossing_narrow_start",
                  "crossing_walk_resume","crossing_resume_second")
    destinations=[deepcopy(S.bm.event(eid)) for eid in crossing_ids]
    statuses={e.get("status","provisional") for e in destinations}
    S.crossing_source=dict(source="AP2",f0=5180,f1=5580,status="selected",
                          selection_source="cutd round 3 delivered D26 EDL",
                          destination_event_status=next(iter(statuses)) if len(statuses)==1 else "mixed",
                          destination_bindings=destinations,
                          reason="delivered edit selects A5180–5579; destination statuses and provenance follow the supplied event map")
    start,end=ev("watch"),ev("sunrise")
    seq=[(start,["D2","F3","Bb3","D4"]),
         (ev("watch_feed_1"),["D2","A2","F#3","D4"]),
         (ev("watch_feed_2"),["D2","G2","B3","D4"]),
         (ev("watch_feed_3"),["D2","A2","C#4","E4"])]
    _chords(S,"watch_chorale",seq,end,.18)
    _part(S,"watch_call","hn_far",gain_db=-3.,humanize_ms=0)
    _part(S,"watch_answer","hn_farther",gain_db=-4.,humanize_ms=0)
    K.call(S,"watch_call","D4",ev("watch_feed_1"),rhythm=(1.,1.,2.),vel=.23,sync_first=True,antic=0.)
    K.answer(S,"watch_answer","D5",ev("watch_feed_2"),rhythm=(1.,.5,.5,2.),vel=.21,sync_first=True,antic=0.)
    for eid,pn in (("watch_feed_1","watch_call"),("watch_feed_2","watch_answer"),("watch_feed_3","watch_chorale_vla_q")):
        _bind(S,eid,pn,"a composed phrase moves the watch's harmony","soft")


def _dawn_and_after(S,a,c):
    ev=S.bm.ev
    _excerpt(S,c,4720,5200,ev("sunrise"),"dawn",gain_db=-.5,
             named={4720:"sunrise",4760:"dawn_call",4880:"dawn_answer",5040:"dawn_home",5160:"dawn_tonic"})
    S.beacon_theme=dict(call_event="dawn_call",answer_event="dawn_answer",home_event="dawn_home",
                        tonic_event="dawn_tonic",call_part="dawn_hn",answer_part="dawn_vln1")
    _excerpt(S,a,6000,6240,ev("terraces"),"terraces",skip=("watch",),gain_db=-1.5,
             named={6000:"terraces",6080:"terraces_home",6160:"terraces_cadence"})
    _excerpt(S,c,5200,5440,ev("plenty"),"plenty",gain_db=-.5,named={5200:"plenty"})
    # The copied hymn is clipped to its 12-beat picture span, then released into the leaf.
    _part(S,"lastleaf_solo","vla_q",gain_db=3.,send=.22,humanize_ms=0)
    start,stop=ev("last_leaf"),ev("last_leaf_break")
    unit=(stop-start-1.)/3.
    _phrase(S,"lastleaf_rest","lastleaf_solo","E3",start+1.,[unit,unit,unit,3.],
            event="last_leaf_break",level=.27)
    _bind(S,"last_leaf_break","lastleaf_solo","the RING phrase rests one note short over the open leaf","soft")
    _chords(S,"lastleaf",[(start,["D2","A2","F#3","D4"])],ev("last_pages")-1.,.13)
    # No score or hall is allowed in the voice reservation. It is a musical space, not speech audio.
    blank,prep,cad,end=(ev(x) for x in ("blank","plagal_prepare","plagal","score_end"))
    _chords(S,"pages",[(blank,["D2","F3","Bb3","D4"]),
                        (ev("title"),["D2","G3","Bb3","D4"]),
                        (prep,["D2","G3","B3","D4"]),
                        (cad,["D2","A2","F#3","D4"])],end-.6,.15)
    _bind(S,"blank","pages_vla_q","Bb/D: the watch's first chord returns after the voice space","soft")
    _bind(S,"plagal","pages_vla_q","quiet G/D to D","soft")
    for pn,p in S.parts.items():
        if pn.startswith("pages_"):
            p.d((cad,.17),(end-1.,.055),(end-.6,.02))
            for n in p.notes:
                n.kw["rel"] = .2


def build(bm=None, *, transition_pass=False):
    if bm is None:
        from barmap_D_score import DraftMap
        bm=DraftMap()
    if bm.bars != 115 or bm.frames != 9200:
        raise ValueError("cut D treatment v2 requires exactly 115 bars / 9200 frames")
    S=K.Score("D",bm)
    S.motif_phrases=[]
    S.picture_bindings=[]
    S.reuse=[]
    S.pcm_regions=[opening_region()]
    S.hard_silences=[(round(bm.ev(a)*20),round(bm.ev(b)*20)) for a,b in
                     (("silence_before_gap","gap"),("voice_start","voice_end"),("score_end","film_end"))]
    S.breaths=[]  # donor all-stem breaths would interrupt D's continuous WATCH
    a,c=_donors()
    _opening_tail(S,a)
    _old_story(S,c)
    _inscription(S)
    _race(S)
    _excerpt(S,c,1680,1920,bm.ev("deep"),"deep_race",named={1680:"deep",1760:"tick_8",1840:"tick_16"})
    _vision_gap_refusal(S)
    _trap_and_glow(S)
    _promise_and_ridges(S)
    _map_and_instep(S)
    _open_ring(S,c)
    _crossing_and_watch(S,a)
    _dawn_and_after(S,a,c)
    # Relative short-term loudness bands are an explicit draft contract; the
    # verifier measures these against the rendered film peak, never note count.
    bands={"D01":(-90,-10),"D02":(-30,-7),"D03":(-22,-2),"D04":(-30,-2),"D05":(-20,-2),
           "D06":(-25,-5),"D07":(-32,-6),"D08":(-28,-4),"D09":(-20,-1),"D10":(-16,0),
           "D11":(-20,-1),"D12":(-15,0),"D13":(-16,0),"D14":(-35,-3),"D15":(-30,-5),
           "D16":(-20,-1),"D17":(-35,-7),"D18":(-28,-3),"D19":(-28,-3),"D20":(-30,-4),
           "D21":(-30,-3),"D22":(-28,-4),"D23":(-28,-4),"D24":(-28,-4),"D25":(-30,-4),
           "D26":(-30,-3),"D27":(-30,-5),"D28":(-22,-2),"D29":(-24,-2),"D30":(-28,-3),
           "D31":(-30,-5),"D32":(-40,-5),"D33":(-36,-5),"D34":(-120,-60)}
    S.levels=[(sid,*band) for sid,band in bands.items()]
    S.rules=[("the dawn stays at least 2 LU under the film's loudest",bm.ev("sunrise")*BEAT_S,
              bm.ev("terraces")*BEAT_S,-2.)]
    S.groups={"race_taiko":dict(ceiling_db=-9.,release=.08),"trap_taiko":dict(ceiling_db=-9.,release=.08)}
    # Explicit score gain remains D-only. The protected PCM is copied after it.
    S.fader=[]
    # First render: paired CALL peaks were 2.471 / 2.304 LU below the vision,
    # just above their -3 LU band. This -1 dB mastering ride responds to those
    # readings; orchestration and the private premaster stay intact. Units: frames.
    S.master_rides=[(bm.ev("crowns")*20-40,0.),(bm.ev("crowns")*20,-1.),
                    (bm.ev("two_fires")*20+160,-1.),(bm.ev("two_fires")*20+200,0.)]
    S.ring_families=[sorted({n["part"] for n in r["notes"]}) for r in S.motif_phrases]
    S.watch_parts=(WATCH_PART,)
    S.watch_pitch=m("D3")
    S.call_parts=CALL_PARTS
    if transition_pass:
        from transitions_D import apply
        S.transition_manifest=apply(S,bm)
        S.transition_pass=True
    S.sync_bindings=[]
    for binding in S.picture_bindings:
        p=S.parts.get(binding["part"])
        if p is None:
            continue
        for index,n in enumerate(p.notes):
            if n.pitch is not None and abs(n.start-binding["beat"])<1e-6:
                S.sync_bindings.append(dict(part=p.name,event=binding["event"],note_index=index,
                                            offset_frames=binding["offset_beats"]*20))
    return S


def _actual_phrase(S,record):
    out=[]
    for ref in record["notes"]:
        if ref["part"] not in S.parts:
            return None
        matches=[n for n in S.parts[ref["part"]].notes
                 if n.pitch is not None and abs(n.start-ref["start"])<1e-6 and n.gain_db>AUDIBLE_DB]
        match=next((n for n in matches if n.pitch==ref["pitch"]),None)
        if match is None:
            return None
        out.append(match)
    return out


def check(S,bm=None):
    """Rules on actual notes plus the output contract (rendered checks are separate)."""
    bm=bm or S.bm
    problems=[]
    ev=bm.ev
    allow=((ev("mountain"),ev("inscription")),(ev("vision"),ev("silence_before_gap")),
           (ev("old_fire"),ev("unfinished")))
    found_vision=set()
    for pn,p in S.used().items():
        ns=sorted((n for n in p.notes if n.pitch is not None and n.gain_db>AUDIBLE_DB),key=lambda n:n.start)
        for i in range(len(ns)-4):
            part=ns[i:i+5]
            iv=tuple(round(n.pitch-part[0].pitch) for n in part)
            if iv != RING_INTERVALS:
                continue
            if not any(lo-1e-6<=part[0].start and part[-1].start<hi-1e-6 for lo,hi in allow):
                problems.append(f"COMPLETED RING OUTSIDE OLD STORY/VISION: {pn} at {part[0].start*20:.1f}f")
            if ev("vision")<=part[0].start<ev("silence_before_gap"):
                found_vision.add(round(part[-1].start,6))
    if found_vision != {round(ev("vision_complete"),6)}:
        problems.append(f"VISION MUST COMPLETE ONCE ON vision_complete: {sorted(found_vision)}")
    for r in S.motif_phrases:
        actual=_actual_phrase(S,r)
        expect=K.RING if r["complete"] else K.RING[:4]
        if actual is None or tuple(n.pitch-r["root"] for n in actual)!=tuple(expect):
            problems.append(f"MOTIF NOTES DO NOT MATCH CONTRACT: {r['id']}")
    for eid in ("crowns_kindle","two_fires"):
        rows=[]
        for pn,pan in zip(CALL_PARTS,(-1.,1.)):
            p=S.parts[pn]
            ns=sorted((n for n in p.notes if ev(eid)-1e-6<=n.start<ev(eid)+4),key=lambda n:n.start)
            rows.append([(n.pitch,round(n.start-ev(eid),6),n.dur) for n in ns])
            if p.pan!=pan or p.width!=0 or p.humanize_ms!=0:
                problems.append(f"CALL MUST BE HARD-PANNED AND UNHUMANISED: {pn}")
        wanted=[(m("D4"),0.,1.),(m("A4"),1.,1.),(m("D5"),2.,2.5)]
        if any(row!=wanted for row in rows):
            problems.append(f"CALL PAIR MUST BE IDENTICAL ON {eid}: {rows}")
    for lo,hi in ((ev("silence_before_gap"),ev("gap")),(ev("voice_start"),ev("voice_end")),
                  (ev("score_end"),ev("film_end"))):
        for pn,p in S.used().items():
            for n in p.notes:
                if n.gain_db>AUDIBLE_DB and n.start<hi-1e-6 and n.start+n.dur>lo+1e-6:
                    problems.append(f"NOTE IN HARD SILENCE: {pn} at {n.start*20:.1f}f")
        if (round(lo*20),round(hi*20)) not in S.hard_silences:
            problems.append("HARD SILENCE MASTER MASK MISSING")
    watch=sorted(S.parts[WATCH_PART].notes,key=lambda n:n.start)
    reach=ev("two_fires")
    for n in watch:
        if n.pitch != m("D3") or n.start>reach+1e-6:
            problems.append("WATCH D HAS A GAP OR WRONG PITCH")
        reach=max(reach,n.start+n.dur)
    if reach<ev("terraces")-1e-6:
        problems.append("WATCH ENDS BEFORE THE DAWN ENDS")
    if not S.levels or {x[0] for x in S.levels}!={s["id"] for s in bm.sections}:
        problems.append("LEVEL BANDS DO NOT COVER EVERY SECTION")
    # The independent verifier follows actual attacks across instrument changes;
    # phrase metadata alone cannot catch a terminal note added on another desk.
    from verify_D import ring_problems, beacon_problems
    problems.extend(ring_problems(S))
    problems.extend(beacon_problems(S, dawn_frame=round(ev("sunrise")*20)))
    return problems


if __name__=="__main__":
    score=build()
    issues=check(score)
    print(f"D: {len(score.used())} parts / {sum(len(p.notes) for p in score.used().values())} notes")
    print(f"rule checks: {len(issues)}")
    for problem in issues:
        print(problem)
    raise SystemExit(bool(issues))
