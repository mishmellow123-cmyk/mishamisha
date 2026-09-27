"""THE LONG DAWN v3 - B . THE KEEPER (THE VIGIL), the score: a call without an answer, for one long night.

Locked to music/v3/barmap_B.json (SHOWRUNNER-REV) through music/v3/cues_B.json (cues_v3.py); both LOCKED.
B DECISION (06:40Z) + director 10:05Z: B is THE VIGIL, one night in the same 68 bars.  The chaconne's variations
mark the night's hours, not years (the cue sheet's "year_N" ids are hours 2-9).  No cutaways: B8 and B10 play in
her frame, and the far answers are only pinpricks.  From bar 46 a traveller's child stays and falls asleep against
her.  On bar 63 she presses her fire-steel into the child's palm; HOME (from 62 b1) still arrives as the child wakes.

Palette (BIBLE_V3 section 9 + REVISION 1): her cello, strings, far horns, and the wind as the floor.  No drums,
no glass, no piano.  The quietest score; mf only while the range stands alight in the hand-back's crane.
HER CELLO: VSCO-2 CE has no solo cello, so her voice is the quiet cello section, close and narrow, with the
solo contrabass (thumb position, D3-D4) inside it as one player's core: an exposed line that is doubled and
usually sits on a pad (red team 5.4).  A/B alternatives: music/out/v3/tests/voice_*.wav.

The rules it keeps, checked by analyze_v3 / the notes:
  whoever lights a fire gets the CALL (her cello, always on D; a traveller's torch gets it on a horn); only an
  answer earns the ANSWER (a horn), first heard at the first far pinprick (bar 35); the whole theme (CALL,
  ANSWER, HOME) only at the dawn; the effects own every fire, feed and the sunrise (no stroke: blooms and swells
  only); the dawn is the warmest moment, not the loudest.

THE PLAN (one night)
  B1 dusk        a low D (basses + cellos), her cello steps D4 C4 Bb3 A3 G3 F3 E3 D3, one note per peak going
                 dark, the last on the red point (bar 8 b3); the violas take the fifth (A3) once she passes it
  B2-B3          her D3 held under the wind, gone by bar 11 b3; then only wind (and the lid, the breath, the hiss)
  B4             effects only; the low D enters with the catch (bar 17 b2.8)
  B5             the roar (effects); the CALL on her cello from bar 18 b3 over the low D (D3 A3 D4)
  B6 hour 1      the drone (D2 + A2); the call at bars 20, 22 and 24, each followed by a bar of silence; she
                 feeds the fire (25 b3, effects)
  B7 hours 2-4   the chaconne: a two-bar ground (a tetrachord in half notes), one variation per hour; her call
                 in the first bar, the answer's place empty in the second, a feed on its b3.
                   D C Bb A | C Bb A G | Bb A G F   (down a degree an hour; minor, darkening; storm, clear, fog)
  B8 dead of     in her frame: the ground holds its breath (F, then nothing); one high harmonic (A5); her call
     night       falters (D3, A3, and no octave)
  B9 hours 5-7   G A Bb C | A Bb C D | B C# D E: the ground turns and climbs; the first ANSWER, one far horn, on
                 the first pinprick (35 b1); the traveller's torch takes the CALL on a horn (37 b1); village lights
  B10 her flare  in her frame: she builds the fire up and calls (D pedal); a nearer summit lights its own
                 beacon (41 b1): the ANSWER, nearer, in major for the first time, over G (the Lydian #11 is its C#)
  B11 hours 8-9  E F# G A | A B C# D (D major, one continuous climb): call and answer in dialogue, more voices
  B12 the child  the bass arrives on D, softly; a traveller's child stays; she feeds the fire (47 b1); she sits
                 and calls (48 b1), the child asleep against her; the far horns answer, overlapping, far and soft
  B13 hand-back  the crane: horns near and far exchange call and answer, strings pp -> mf (B's loudest, bar 52);
                 the east greys: A7sus4 -> A, subito p; 275 ms of silence; a bloom on D add9 on the sunrise;
                 the CALL, late, on her cello (54 b3); the ANSWER passed note by note from horn to farther horn,
                 one note per beacon as it pales (D, C#, B, F#, A, D; bars 55-60), each horn then silent;
                 her cello alone, the HOME phrase (B A F# ... D) from 62 b1; on 63 b3, as she presses her
                 fire-steel into the child's palm, the high harmonic of the dead of night returns (A5, ppp: no
                 stroke); HOME arrives on D as the child wakes (64 b1); birdsong (bar 65) over G/D
  B14 title      D major on the downbeat (bar 66), ringing out into birdsong
"""
import kit_v3 as K

PCS = {"minor": (0, 2, 3, 5, 7, 8, 10), "dorian": (0, 2, 3, 5, 7, 9, 10), "major": (0, 2, 4, 5, 7, 9, 11)}
D = 2
K.SEATS.setdefault("hn5", dict(inst="horn", pan=0.38, width=0.25, depth=0.97, send=0.82, gain_db=-3.0, humanize_ms=12,
                               bus="brass"))
K.SEATS.setdefault("hn6", dict(inst="horn", pan=-0.05, width=0.2, depth=1.0, send=0.9, gain_db=-5.0, humanize_ms=14,
                               bus="brass"))


# seconds added to her entry's anticipation (measured: analysis/v3/final_B, sync table).  HOME (62 b1) needs none:
# each layer alone arrives within ~30 ms; the doubled B3 beats in the narrow band, so it is measured on her lead
SYNC_TRIM = {"m.dusk_voice": 0.14, "m.y60_call": -0.02}
# absolute anticipation (s) where the attack table picks the wrong round robin (measured on the stems): HOME's B3
ANTIC_SET = {("home_phrase", "voice"): 0.17, ("home_phrase", "voice_core"): 0.255}

# dB per section, applied to the players' dynamics (kit ride)
CRANE_TRIM_DB = -5.0
RIDE = {"B1": -6.0, "B2": -6.0, "B5": -2.5, "B6": -2.5, "B7": -4.7, "B8": -10.0, "B9": -4.8, "B10": -3.5, "B11": -3.4,
        "B12": -2.0, "B13": 0.0, "B14": -1.7}


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
def triad(bass, mode, major_v=False):
    sc = [(D + x) % 12 for x in PCS[mode]]
    b = bass % 12
    i = sc.index(b) if b in sc else min(range(7), key=lambda k: abs(sc[k] - b))
    ch = [sc[i], sc[(i + 2) % 7], sc[(i + 4) % 7]]
    if major_v and b == (D + 7) % 12:
        ch[1] = (b + 4) % 12                  # the A chord major: the leading note C#
    return ch


def her(S, fn, *a, **kw):
    """her cello: a kit phrase on the lead ('voice'), copied into the core ('voice_core')"""
    n0 = len(S.P("voice").notes)
    end = fn(S, "voice", *a, **kw)
    for nt in S.P("voice").notes[n0:]:
        S.P("voice_core").n(nt.pitch, nt.start, nt.dur, nt.vel, legato=nt.legato, sync=nt.sync, **dict(nt.kw))
    return end


def her_note(S, p, b0, dur, legato=False, sync=False):
    S.P("voice").n(p, b0, dur, legato=legato, sync=sync)
    S.P("voice_core").n(p, b0, dur, legato=legato, sync=sync)


def her_dyn(S, *pts, core=-0.02):
    S.P("voice").d(*pts)
    S.P("voice_core").d(*[(b, max(0.02, v + core)) for b, v in pts])


def night(S, b0, L, ground, mode, level, call_at=1.0, call_rh=(1, 1, 1.5), call=True, answers=(), calls=(),
          pads=("vla_q", "vln2_q"), prev=None, major_v=False):
    """one variation of the chaconne (a night we see): the ground in half notes (basses an octave under the
    cellos), a pad voice-led on its triads (the bass has the root, the pad keeps the third), her CALL, and
    the answers in the second bar.  Returns the pad's last voicing."""
    u = L / 4.0
    for k, g in enumerate(ground):
        t = b0 + k * u
        S.P("cb_q").n(g - 12, t, u + 0.06, legato=True)
        S.P("vc_q").n(g, t, u + 0.06, legato=True)
    # the pad is a halo above her voice (her call lives in D3-D4): violas D4-A4, violins F4-D5 (and A4-G5)
    ranges = [[(62, 69)], [(62, 69), (65, 74)], [(62, 69), (65, 74), (69, 79)]][len(pads) - 1]
    pv = prev if (prev and len(prev) == len(pads)) else None
    last = None
    for k, g in enumerate(ground):
        ch = triad(g, mode, major_v=(major_v and k == 3))
        pv = K.voice_lead([ch], ranges, prev=pv, need=[ch[1]] if len(pads) > 1 else [ch[1]])[0]
        for pn, p in zip(pads, pv):
            S.P(pn).n(p, b0 + k * u, u + 0.08, legato=True)
        last = pv
    for pn in pads:
        S.P(pn).d((b0, level * 0.72), (b0 + L * 0.45, level * 0.86), (b0 + L - 0.3, level * 0.76))
    for pn in ("cb_q", "vc_q"):
        S.P(pn).d((b0, level * 0.92), (b0 + L * 0.45, level), (b0 + L - 0.3, level * 0.9))
    if call:
        c = b0 + call_at
        her(S, K.call, "D3", c, rhythm=tuple(call_rh), sync_first=True)
        v = min(0.4, level + 0.08)
        end = c + sum(call_rh)
        her_dyn(S, (c - 0.05, v), (c + 1, v), (c + 2.2, v * 1.06), (end - 0.3, v * 0.8), (end + 0.5, 0.04))
    for a in calls:                                   # the torches: the CALL passed from horn to horn
        t = b0 + a["at"]
        rh = tuple(a.get("rhythm", (1, 1, 1.5)))
        vel = a.get("vel", 0.32)
        K.call(S, a["horn"], a.get("root", "D3"), t, rhythm=rh)
        S.P(a["horn"]).d((t - 0.05, vel * 0.8), (t + 0.8, vel), (t + sum(rh) - 0.4, vel * 0.85),
                         (t + sum(rh) + 0.5, 0.04))
    for a in answers:
        t = b0 + a["at"]
        rh = tuple(a.get("rhythm", (1, .5, .5, 2)))
        vel = a.get("vel", 0.32)
        K.answer(S, a["horn"], a.get("top", "D4"), t, mode=a.get("mode", "major"), rhythm=rh)
        S.P(a["horn"]).d((t - 0.05, vel * 0.8), (t + 0.6, vel), (t + sum(rh) - 0.4, vel * 0.85),
                         (t + sum(rh) + 0.5, 0.04))
    return last


def pad(S, b0, b1, voicing, levels, legato=False, sync=False):
    """a held chord: voicing = {part: [pitches]}; levels = [(beat, level), ...] applied to every part"""
    for pn, ps in voicing.items():
        for p in ps:
            S.P(pn).n(p, b0, b1 - b0, legato=legato, sync=sync)
        S.P(pn).d(*levels)


# ---------------------------------------------------------------------------
# the sections
# ---------------------------------------------------------------------------
def setup(S):
    for pn in ("voice", "voice_core", "cb_q", "vc_q", "vla_q", "vln2_q", "vln1_q", "cb", "vc", "vla", "vln2", "vln1",
               "hn", "hn2", "hn_far", "hn_farther", "hn5", "hn6", "harmonic"):
        S.add(pn)
    S.add("drone_cb", "cb_q")
    S.add("drone_vc", "vc_q")
    S.add("vla_hold", "vla_q", pan=-0.12)
    S.add("harm_str", "svln", inst="svln_q", pan=-0.4, depth=0.6, send=0.6, gain_db=-4.0)


def build(bm):
    S = K.Score("B", bm)
    setup(S)
    ev = bm.ev
    dusk(S, bm, ev)
    fire_and_night(S, bm, ev)
    vigil(S, bm, ev)
    handback(S, bm, ev)
    end = bm.bars * 4
    for pn in ("hn", "hn2", "hn_far", "hn_farther", "hn5", "hn6"):
        K.breathe(S, pn, 0, end, depth=0.12, min_dur=1.4)
    # the level plan, relative to the crane's mf (analysis v2 of the render): the first half is pp-p, the
    # dawn warm but under the crane
    K.ride(S, bm, RIDE)
    # every attacked note starts early by its own sample's measured attack, so that it ARRIVES on the beat; the
    # quiet cello section's soft bow needs up to 0.54 s (D3), the solo bass core ~0.08-0.3 s, the horns ~0.1 s
    K.anticipate(S, "voice", hi=0.56)
    for pn in ("voice_core", "hn", "hn2", "hn_far", "hn_farther", "hn5", "hn6"):
        K.anticipate(S, pn)
    for pn in ("voice", "voice_core"):              # her legato steps: the crossfade centred on the beat
        for n in S.P(pn).notes:
            if n.legato:
                n.kw["antic"] = 0.02
    # measured trims (analysis of the previous render): a soft entry out of silence under a crescendo arrives late
    for eid, dt in SYNC_TRIM.items():
        b0 = bm.ev(eid)
        for pn in ("voice", "voice_core"):
            for n in S.P(pn).notes:
                if abs(n.start - b0) < 1e-6 and not n.legato:
                    n.kw["antic"] = max(0.02, n.kw.get("antic", 0.0) + dt)
    for (eid, pn), a in ANTIC_SET.items():
        for n in S.P(pn).notes:
            if abs(n.start - bm.ev(eid)) < 1e-6 and not n.legato:
                n.kw["antic"] = a
    for s in bm.sections:
        if s.get("rel"):
            S.levels.append((s["id"], *s["rel"]))
        if s.get("centroid"):
            S.centroid.append((s["id"], *s["centroid"]))
    t = lambda sid: bm.event(sid)["t"]
    S.rules = [("the dawn is warm, not loud: its loudest 3 s at least 2 LU under the film's loudest (the crane)",
                t("sunrise"), t("title") + 10.0, -2.0),
               ("B's first half is the quietest: dusk to the first night at least 6 LU under the loudest",
                0.0, t("year_2"), -6.0)]
    return S


def dusk(S, bm, ev):
    peaks = [e["beat"] for e in bm.find(tag="peak")]                  # 6 peaks + the last red point
    v_in, out = ev("m.dusk_voice"), ev("m.climb_out")
    last = peaks[-1]
    # the low D (basses + cellos), from the silence, gone as the last light goes
    K.held(S, "drone_cb", "D2", 2.0, last + 1.5, bow=8)
    K.held(S, "drone_vc", "D3", 2.4, last + 1.5, bow=8)
    for pn, lv in (("drone_cb", 0.21), ("drone_vc", 0.17)):
        S.P(pn).d((2.0, 0.03), (5.0, lv * 0.7), (8.0, lv), (last - 3, lv * 0.95), (last + 1.4, 0.03))
    # her cello: the call turned downward, filled with steps, one note per peak that goes dark
    line = ["D4", "C4", "Bb3", "A3", "G3", "F3", "E3"]
    times = [v_in] + peaks[:-1]
    for i, (p, t) in enumerate(zip(line, times)):
        t1 = times[i + 1] if i + 1 < len(times) else last
        her_note(S, p, t, t1 - t + 0.05, legato=i > 0, sync=(i == 0))
    # the last step lands on the red point (D3); held through the climb, re-bowed once
    n0 = len(S.P("voice").notes)
    her(S, lambda S_, pn: K.held(S_, pn, "D3", last, out, bow=6.0) or out)
    for pn in ("voice", "voice_core"):
        S.P(pn).notes[n0].legato = True
        S.P(pn).notes[n0].kw.pop("antic", None)
    pts = [(v_in - 0.1, 0.22), (v_in + 1.5, 0.3)]
    for t in peaks[:-1]:
        pts += [(t + 0.05, 0.27), (t + 1.8, 0.32), (t + 3.6, 0.28)]
    pts += [(last, 0.3), (last + 2, 0.31), (last + 8, 0.24), (out - 5, 0.15), (out, 0.02)]
    her_dyn(S, *pts)
    # the violas keep the fifth once her line has passed it, until the last light goes
    a3 = peaks[2]
    S.P("vla_hold").n("A3", a3 + 0.6, last + 1.2 - (a3 + 0.6))
    S.P("vla_hold").d((a3 + 0.6, 0.04), (a3 + 3.5, 0.15), (last - 1, 0.15), (last + 1.1, 0.03))
    S.sync.append((v_in * 60 / 72, "dusk: her cello enters on D4, out of the wind", "voice+voice_core", 0.2, "arrive:62"))
    S.sync.append((peaks[3] * 60 / 72, "dusk: peak 4 goes dark, her cello steps to G3", "voice", 0.2, "pitch:55"))
    S.sync.append((last * 60 / 72, "dusk: the last red point, her cello on D3", "voice", 0.2, "pitch:50"))


def fire_and_night(S, bm, ev):
    catch, roar, c0 = ev("catch"), ev("roar"), ev("call_cello")
    n1 = ev("night_1")
    # the low D enters with the catch and stays under the call as its pad
    K.held(S, "drone_cb", "D2", catch, n1, bow=8)
    K.held(S, "drone_vc", "D3", catch + 0.15, n1, bow=8)
    for pn, lv in (("drone_cb", 0.22), ("drone_vc", 0.17)):
        S.P(pn).d((catch, 0.03), (roar - 0.2, lv), (roar + 1.0, lv * 0.85), (n1 - 1, lv * 0.8), (n1 + 0.4, 0.03))
    her(S, K.call, "D3", c0, rhythm=(1.5, 1.5, 2.8), sync_first=True)
    her_dyn(S, (c0 - 0.05, 0.36), (c0 + 1.5, 0.37), (c0 + 3.0, 0.44), (c0 + 4.5, 0.42), (c0 + 5.8, 0.12))
    S.sync.append((c0 * 60 / 72, "the CALL at the reveal (her cello, on the low D)", "voice+voice_core", 0.2, "arrive:50"))
    # THE FIRST NIGHT: the drone (D2 + A2) and the call three times, each followed by a bar of silence
    y2 = ev("year_2")
    K.held(S, "drone_cb", "D2", n1, y2 + 0.1, bow=8)
    K.held(S, "drone_vc", "A2", n1 + 0.1, y2 + 0.1, bow=8)
    calls = [ev("night_1"), ev("call_2"), ev("call_3")]
    for pn, lv in (("drone_cb", 0.2), ("drone_vc", 0.16)):
        pts = [(n1, lv * 0.8), (n1 + 3, lv)]
        for c in calls:
            pts += [(c + 4, lv * 1.06), (c + 8, lv * 0.96)]
        pts += [(y2 - 2, lv * 0.85), (y2, lv * 0.9)]
        S.P(pn).d(*pts)
    for k, (c, v) in enumerate(zip(calls, (0.35, 0.38, 0.3))):
        her(S, K.call, "D3", c, rhythm=(1, 1, 2.4), sync_first=True)
        her_dyn(S, (c - 0.05, v), (c + 1, v), (c + 2.2, v * 1.06), (c + 3.9, v * 0.8), (c + 4.6, 0.03))
        S.sync.append((c * 60 / 72, f"first night: call {k + 1}", "voice+voice_core", 0.2, "arrive:50"))


def vigil(S, bm, ev):
    prev = None
    for e in bm.find(kind="night"):
        g = K.tetrachord(e["ground"], e.get("dir", -1), e.get("mode", "minor"), "D")
        pads = ("vla_q", "vln2_q", "vln1_q")[:e.get("pad", 2)]
        prev = night(S, e["beat"], e.get("len_beats", 8), g, e.get("mode", "minor"), e.get("level", 0.25),
                     call_at=e.get("call_at", 1.0), call_rh=tuple(e.get("call_rh", (1, 1, 1.5))),
                     answers=e.get("answers", []), calls=e.get("calls", []), pads=pads, prev=prev,
                     major_v=e.get("major_v", False))
        if e.get("first_answer"):
            a = e["answers"][0]
            S.sync.append(((e["beat"] + a["at"]) * 60 / 72, "THE ANSWER, first heard (one far horn)", a["horn"],
                           0.25, "arrive:62"))
    # B8 THE DEAD OF NIGHT (in her frame; the far-peak cutaway is cut): the ground holds its breath (F, then
    # nothing); one high harmonic; and her call falters on the variation's beat 2 (D3, A3, and no octave: the
    # call not even finished), trailing off before the pinprick moment (33 b1)
    f1, pt = ev("far_peak_1"), ev("child_points")
    for pn, p, lv in (("cb_q", "F1", 0.2), ("vc_q", "F2", 0.18)):
        S.P(pn).n(p, f1, 4.6, legato=True)
        S.P(pn).d((f1, lv), (f1 + 2, lv * 0.7), (f1 + 4.5, 0.03))
    S.P("harmonic").n("A5", f1 + 0.5, 7.2, atk=1.6, rel=1.4)
    S.P("harmonic").d((f1 + 0.5, 0.22), (pt, 0.3), (pt + 2.5, 0.32), (f1 + 7.7, 0.12))
    S.P("harm_str").n("A5", f1 + 1.0, 6.6)
    S.P("harm_str").d((f1 + 1.0, 0.08), (pt, 0.12), (f1 + 7.6, 0.05))
    fc = f1 + 1.0
    her(S, K._line, ["D3", "A3"], [1.0, 2.6], fc, sync_first=True)
    # (the B8 ride is -10 dB: 0.40 here plays as ~0.19, a little under the hours' calls)
    her_dyn(S, (fc - 0.05, 0.4), (fc + 1.0, 0.41), (fc + 1.6, 0.36), (fc + 3.0, 0.16), (fc + 3.7, 0.03))
    S.sync.append((fc * 60 / 72, "the dead of night: her call falters (D, A, no octave)", "voice+voice_core", 0.2,
                   "arrive:50"))
    # B10 HER FLARE (in her frame; the far-peak cutaway is cut): she builds the fire up and it flares; her call,
    # full, over a D pedal; then a nearer summit lights its own beacon (41 b1): the ANSWER, nearer, in major for
    # the first time, over G (the Lydian #11 is its C#)
    f2, fl, hf = ev("far_peak_2"), ev("m.flare"), ev("his_fire")
    pad(S, f2, hf + 0.1, {"cb_q": ["D2"], "vc_q": ["D3"], "vla_q": ["F3"], "vln2_q": ["A3"]},
        [(f2, 0.12), (f2 + 2, 0.2), (hf - 0.2, 0.22)], legato=True)
    her(S, K.call, "D3", fl, rhythm=(1, 1, 1.6), sync_first=True)
    her_dyn(S, (fl - 0.05, 0.34), (fl + 1, 0.35), (fl + 2.5, 0.35), (fl + 4.0, 0.03))
    S.sync.append((fl * 60 / 72, "her fire flares: her call, near", "voice+voice_core", 0.2, "arrive:50"))
    pad(S, hf, hf + 4.1, {"cb_q": ["G1"], "vc_q": ["G2"], "vla_q": ["B3"], "vln2_q": ["F#4"]},
        [(hf, 0.2), (hf + 1.5, 0.26), (hf + 4.0, 0.24)], legato=True)
    K.answer(S, "hn", "D4", hf + 0.5, mode="major", rhythm=(1, .5, .5, 1.8))
    S.P("hn").d((hf + 0.45, 0.3), (hf + 1.2, 0.36), (hf + 3.2, 0.32), (hf + 4.4, 0.04))
    S.sync.append(((hf + 0.5) * 60 / 72, "a nearer beacon: the answer, nearer (major)", "hn", 0.25, "arrive:62"))
    # B12 THE CHILD (hour 10): the bass arrives on D, softly; a traveller's child stays; she feeds the fire (47 b1,
    # effects); she sits and calls (48 b1), gently, the child falling asleep against her; the far horns answer,
    # overlapping, far and soft (the world outside); the pad settles down into the crane's pp
    y60, st, yc = ev("year_60"), ev("y60_strikes"), ev("m.y60_call")
    crane = ev("crane")
    pad(S, y60, st, {"cb_q": ["D2"], "vc_q": ["D3", "A3"], "vla_q": ["F#3"], "vln2_q": ["D4"], "vln1_q": ["A4"]},
        [(y60, 0.18), (y60 + 2, 0.25), (st - 0.3, 0.25)], sync=True)
    pad(S, st, yc, {"cb_q": ["D2"], "vc_q": ["D3", "B3"], "vla_q": ["G3"], "vln2_q": ["D4"], "vln1_q": ["B4"]},
        [(st, 0.24), (yc - 0.3, 0.25)], legato=True)
    pad(S, yc, crane + 0.1, {"cb_q": ["D2"], "vc_q": ["D3", "A3"], "vla_q": ["F#3"], "vln2_q": ["E4"], "vln1_q": ["A4"]},
        [(yc, 0.25), (yc + 2.5, 0.24), (crane - 0.5, 0.19)], legato=True)
    her(S, K.call, "D3", yc, rhythm=(1, 1, 2.5), sync_first=True)
    her_dyn(S, (yc - 0.05, 0.32), (yc + 1, 0.33), (yc + 2.5, 0.33), (yc + 4.2, 0.08))
    for off, hn, v in ((1.5, "hn_far", 0.28), (2.25, "hn_farther", 0.27), (3.0, "hn5", 0.26), (3.5, "hn6", 0.26)):
        t = yc + off
        K.answer(S, hn, "D4", t, mode="major", rhythm=(1, .5, .5, 1.5))
        S.P(hn).d((t - 0.05, v * 0.8), (t + 0.6, v), (t + 3.0, v * 0.85), (t + 3.9, 0.04))
    S.sync.append((yc * 60 / 72, "hour 10: her call as she sits, the child asleep", "voice+voice_core", 0.2,
                   "arrive:50"))


# the crane: D major, the strings rising from pp to mf, bar by bar (cb, vc, vla, vln2, vln1)
CRANE = [(0, ["D2"], ["D3", "A3"], ["F#3"], ["A4"], ["D5"]),              # 49: D, pp
         (4, ["D2"], ["D3", "B3"], ["G3"], ["B4"], ["D5"]),               # 50: G/D
         (8, ["B1"], ["B2", "F#3"], ["D4"], ["A4"], ["F#5"]),             # 51: Bm7
         (12, ["D2"], ["D3", "A3"], ["F#4"], ["D5"], ["A5"])]             # 52: the range alight, D, mf


def handback(S, bm, ev):
    crane, alight, grey, sun = ev("crane"), ev("range_alight"), ev("east_greys"), ev("sunrise")
    call, home, wake, birds, title = ev("call_late"), ev("home_phrase"), ev("child_wakes"), ev("birdsong"), ev("title")
    hers, hands = ev("her_light"), ev("hands_open")
    pales = [e["beat"] for e in bm.find(tag="pale")]
    end = bm.bars * 4
    # ---- the crane: strings pp -> mf (the range alight, bar 52), then the east greys (bar 53)
    for i, (off, cb, vc, vla, v2, v1) in enumerate(CRANE):
        t = crane + off
        t1 = crane + CRANE[i + 1][0] if i + 1 < len(CRANE) else grey
        for pn, ps in (("cb", cb), ("vc", vc), ("vla", vla), ("vln2", v2), ("vln1", v1)):
            for p in ps:
                S.P(pn).n(p, t, t1 - t + 0.08, legato=i > 0, sync=(i == 0))
    for pn in ("cb", "vc", "vla", "vln2", "vln1"):
        S.P(pn).gain_db += CRANE_TRIM_DB
    for pn, pk in (("cb", 0.54), ("vc", 0.52), ("vla", 0.5), ("vln2", 0.52), ("vln1", 0.55)):
        S.P(pn).d((crane, 0.16), (crane + 4, 0.26), (crane + 8, 0.38), (alight, pk), (alight + 2.5, pk * 0.95),
                  (grey - 0.5, pk * 0.7), (grey + 0.3, 0.04))
    horns = [(crane + 1, "hn", "call", 0.32), (crane + 4, "hn_far", "ans", 0.3), (crane + 6, "hn2", "call", 0.35),
             (crane + 9, "hn_farther", "ans", 0.32), (crane + 10, "hn", "call", 0.39), (crane + 12, "hn5", "ans", 0.33),
             (alight - 1, "hn2", "call", 0.42), (alight, "hn_far", "ans", 0.4)]
    for t, hn, what, v in horns:
        if what == "call":
            K.call(S, hn, "D3", t, rhythm=(1, 1, 2.5))
            S.P(hn).d((t - 0.05, v * 0.85), (t + 1, v), (t + 3.2, v * 0.9), (t + 4.8, 0.04))
        else:
            K.answer(S, hn, "D4", t, mode="major", rhythm=(1, .5, .5, 2))
            S.P(hn).d((t - 0.05, v * 0.85), (t + 0.8, v), (t + 3.4, v * 0.85), (t + 4.4, 0.04))
    # the east greys: A7sus4 -> A, subito piano; then 275 ms of silence (the cue sheet's breath)
    br = min(b[0] for b in S.breaths if b[0] > grey)
    pad(S, grey, grey + 2, {"cb_q": ["A1"], "vc_q": ["A2", "E3"], "vla_q": ["D4"], "vln2_q": ["G4"], "vln1_q": ["A4"]},
        [(grey, 0.2), (grey + 1.5, 0.26)])
    pad(S, grey + 2, br + 0.02, {"cb_q": ["A1"], "vc_q": ["A2", "E3"], "vla_q": ["C#4"], "vln2_q": ["G4"],
                                 "vln1_q": ["A4"]}, [(grey + 2, 0.26), (br - 0.2, 0.3)], legato=True)
    S.sync.append((grey * 60 / 72, "the east greys: the held chord", "vla_q", 0.2, "bloom"))
    # ---- the sunrise: a bloom out of the silence on D add9, warm; the harmony follows the answer's notes
    #      D add9 (sun, pale 1) | G/D (pale 2: C# its #11) | Bm/D (pale 3) | D add9 (pales 4, 5) | G/D (pale 6)
    #      | D add9 (her light, her hands) | the D pedal alone under HOME | G/D (birdsong) | D major (title)
    DADD9 = {"cb_q": ["D2"], "vc_q": ["A2", "D3"], "vla_q": ["F#3"], "vln2_q": ["E4"], "vln1_q": ["A4"]}
    GD = {"cb_q": ["D2"], "vc_q": ["G2", "D3"], "vla_q": ["B3"], "vln2_q": ["D4"], "vln1_q": ["F#4"]}
    BMD = {"cb_q": ["D2"], "vc_q": ["B2", "D3"], "vla_q": ["F#3"], "vln2_q": ["D4"], "vln1_q": ["B4"]}
    H = [(sun, DADD9), (pales[1], GD), (pales[2], BMD), (pales[3], DADD9), (pales[5], GD), (hers, DADD9)]
    for i, (t, ch) in enumerate(H):
        t1 = H[i + 1][0] if i + 1 < len(H) else home + 0.1
        for pn, ps in ch.items():
            for p in ps:
                S.P(pn).n(p, t, t1 - t + 0.08, legato=i > 0, sync=(i == 0))
    for pn, top in (("cb_q", 0.25), ("vc_q", 0.24), ("vla_q", 0.23), ("vln2_q", 0.23), ("vln1_q", 0.22)):
        S.P(pn).d((sun, top * 0.55), (sun + 1.4, top * 1.07), (sun + 5, top), (pales[0], top * 0.93),
                  (hers, top * 0.95), (hands, top * 1.0), (home - 0.5, top * 0.7), (home + 0.1, 0.03))
    S.sync.append((sun * 60 / 72, "the bloom out of the silence (D add9), score stem", "score", 0.15, "bloom"))
    # the CALL, late, on her cello (broad), over the bloom
    her(S, K.call, "D3", call, rhythm=(2, 2, 4.5), sync_first=True)
    her_dyn(S, (call - 0.05, 0.36), (call + 2, 0.365), (call + 4, 0.375), (call + 6.5, 0.36), (call + 8.4, 0.05))
    S.sync.append((call * 60 / 72, "THE CALL, late, on her cello", "voice+voice_core", 0.2, "arrive:50"))
    # the ANSWER passed from horn to farther horn, one note per beacon as it pales, each horn then silent
    notes = ["D4", "C#4", "B3", "F#3", "A3", "D4"]
    seats = ["hn", "hn2", "hn_far", "hn_farther", "hn5", "hn6"]
    vels = [0.32, 0.32, 0.31, 0.29, 0.27, 0.24]
    for k, t in enumerate(pales[:6]):
        hn, p, v = seats[k], notes[k], vels[k]
        S.P(hn).n(p, t, 3.3, sync=True, antic=0.07)
        S.P(hn).d((t - 0.05, v * 0.75), (t + 0.7, v), (t + 2.4, v * 0.9), (t + 3.3, v * 0.5), (t + 3.8, 0.03))
        S.sync.append((t * 60 / 72, f"beacon {k + 1} pales: {p} on a farther horn", hn, 0.25, f"arrive:{K.m(p)}"))
    # HOME: her cello alone over the low D, arriving on D3 as the child wakes
    S.P("drone_cb").n("D2", home - 0.2, birds - home + 0.4)
    S.P("drone_cb").d((home - 0.2, 0.12), (home + 4, 0.14), (birds, 0.12))
    her_note(S, "B3", home, 2.0)
    her_note(S, "A3", home + 2, 2.0, legato=True)
    her_note(S, "F#3", home + 4, wake - (home + 4) + 0.05, legato=True)
    her_note(S, "D3", wake, birds + 3 - wake, legato=True)
    her_dyn(S, (home - 0.05, 0.33), (home + 1.5, 0.34), (home + 4, 0.33), (home + 6, 0.35), (wake, 0.33),
            (wake + 3, 0.3), (birds, 0.2), (birds + 3, 0.03))
    S.sync.append((home * 60 / 72, "HOME: her cello alone (the lead layer)", "voice", 0.2, "arrive:59"))
    S.sync.append((home * 60 / 72, "HOME: her cello alone (the core layer)", "voice_core", 0.2, "arrive:59"))
    # bar 63 b3: she presses her fire-steel into the child's palm.  The dead of night's high harmonic returns (A5,
    # ppp, over her held F#: D major), swelling in to arrive with the press (no stroke), held through the wake,
    # gone under the birdsong.  Her cello stays the only line.
    touch = ev("shoulder")
    h0 = touch - 1.4
    S.P("harmonic").n("A5", h0, birds + 2.2 - h0, atk=1.3, rel=1.8)
    S.P("harmonic").d((h0, 0.05), (touch, 0.14), (wake, 0.15), (birds, 0.11), (birds + 2.2, 0.05))
    S.P("harm_str").n("A5", h0 + 0.4, birds + 2.0 - (h0 + 0.4))
    S.P("harm_str").d((h0 + 0.4, 0.04), (touch, 0.08), (wake, 0.085), (birds + 2.0, 0.03))
    S.sync.append((wake * 60 / 72, "HOME arrives on D3 as the child wakes", "voice", 0.2, "pitch:50"))
    # the birdsong over G/D (plagal), then the last chord: D major on the title's downbeat, ringing out
    pad(S, birds, title + 0.1, {"cb_q": ["D2"], "vc_q": ["G2", "D3"], "vla_q": ["B3"], "vln2_q": ["D4"],
                                "vln1_q": ["G4"]}, [(birds, 0.08), (birds + 2, 0.13), (title - 0.3, 0.16)])
    pad(S, title, end - 0.3, {"cb_q": ["D2"], "vc_q": ["A2", "D3"], "vla_q": ["F#3"], "vln2_q": ["A3", "D4"],
                              "vln1_q": ["F#4"]}, [(title, 0.2), (title + 1.2, 0.24), (title + 5, 0.2), (end - 3, 0.07),
                                                   (end - 0.4, 0.02)], legato=True)
    S.sync.append((title * 60 / 72, "the last chord: D major (violins I, G4 -> F#4)", "vln1_q", 0.2, "pitch:66"))
