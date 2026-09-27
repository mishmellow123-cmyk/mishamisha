"""THE LONG DAWN v3 - the FALLBACK MASTER of any cut, from its bar map alone (REVISION 1, red team 5.4):
the effects, the drone, the silence piano and the CALL on horns.  A sparse score can't be cringe.

    python render_v3.py B --fallback          # -> out/v3/fallback_B.wav (+ _score, _sfx stems)

Rules, read from the bar map + cue sheet (section tags, section "level", events, breaths):
  drone      a low D in the strings (cb D2 + vc D3, and the viola's A3 when the level is p or more),
             re-bowed, in every section except those tagged black / silence; at the section's level
             (pp default).  Sections tagged race/edge/brink darken it (the viola's Ab3: the tritone)
             and swell toward mp.  Sections tagged major/dawn/title use D add9 (A2, F#3, E4, A4) instead;
             lydian sections the IGNITION chord (Bbmaj9#11/D).  A race section cut to silence or black
             (not the brink, which has its suck) falls away over its last two beats instead (C11's slip).
  breaths    every breath of the cue sheet cuts the drone: its notes end as the breath begins and new
             ones start on the breath's downbeat (a bloom out of silence, never a stroke).  Before a breath
             inside a section the previous section's voicing holds (C22: the tritone until the Ring goes).
  piano      sections tagged silence: the silence piano, D ... A ... D, then the low fifth, entering
             after at least two seconds of true silence.
  CALL       events tagged call (a fire that is lit): a near horn, D3 A3 D4, p-mp, entering one beat
             after the event (the effects own the fire itself); B's first-night calls (tag night1_call) on
             the event itself, each followed by the bar of silence the bar map leaves.  Close fires pass the call between two
             near horns, and a call shortens (half notes -> quarters) when the next one comes too soon.
  ANSWER     events tagged answer: a far horn, D4 C(#)4 B(b)3 F(#)3 (major in major sections); close
             answers alternate between two far horns, each softer and farther.
  dawn       sections tagged dawn: the bloom on D add9 happens on the breath inside the section (the
             sunrise), or on the section's first beat; one horn CALL two bars after it.
  title      sections tagged title: D add9 resolving to D, dying away by the end.
  effects    the bar map's "ambience" + "sfx" lists (sfx_v3).
"""
import kit_v3 as K

LEVEL = {"niente": 0.05, "ppp": 0.13, "pp": 0.2, "p": 0.28, "mp": 0.36, "mf": 0.55}   # mf = the brink, alone
REL_BAND = {"black": (-45, -12), "niente": (-45, -12), "ppp": (-40, -10), "pp": (-30, -6), "p": (-24, -3),
            "mp": (-18, -1), "mf": (-14, 0)}   # LU under the film's loudest 3 s (analysis relative level map)
CALL_RH = ((2, 2, 4), (1, 1, 2))              # half notes, or quarters when the next call is near
# the conductor's ride per cut (dB on the players' dynamics, section by section), from the battery: the brink is
# limited at the master's ceiling, so every other section is set against it (analysis/v3/fallback_<cut>/report.txt)
FALLBACK_RIDE = {
    "A": {"A10": -2.4, "A13": -1.7, "A14": -1.7, "A15": -2.7, "A19": -1.4},
    "B": {"B1": -2.0, "B2": -1.0, "B5": -3.0, "B6": -8.0, "B7": -2.5, "B9": -1.6, "B10": -2.1, "B11": -1.2},
    "C": {"C2": -6.5, "C12": -4.5, "C15": -1.2, "C18": -1.2, "C26": -0.8},    # renders 1-2 (C12's peak was C11's tail)
}
ANS_RH = ((2, 1, 1, 4), (1, .5, .5, 2))


def _voices(tags, lvl):
    if "lydian" in tags:                      # the IGNITION chord, Bbmaj9(#11)/D
        return [("fb_cb_q", "D2"), ("fb_vc_q", "Bb2"), ("fb_vla_q", "F3"), ("fb_vln2_q", "E4"), ("fb_vln1_q", "A4")]
    if tags & {"major", "dawn", "title"}:
        return [("fb_cb_q", "D2"), ("fb_vc_q", "A2"), ("fb_vla_q", "F#3"), ("fb_vln2_q", "E4"), ("fb_vln1_q", "A4")]
    v = [("fb_cb_q", "D2"), ("fb_vc_q", "D3")]
    if "brink" in tags:                       # the cut's loudest: the tritone doubled up into the violins
        return v + [("fb_vla_q", "Ab3"), ("fb_vln2_q", "D4"), ("fb_vln1_q", "Ab4")]
    if tags & {"race", "edge"}:
        v.append(("fb_vla_q", "Ab3"))
    elif lvl >= 0.3:
        v.append(("fb_vla_q", "A3"))
    return v


def _passes(times, seats, rhythms, end):
    """[(beat, seat, rhythm)]: close events (under 8 beats apart) pass between the seats in turn; each takes the
    long rhythm when its seat's next turn is at least its length away, else the short one"""
    out = []
    run = []
    for t in times + [None]:
        if run and (t is None or t - run[-1] >= 8):
            for k, b in enumerate(run):
                seat = seats[k % len(seats)] if len(run) > 1 else seats[0]
                nxt = run[k + len(seats)] if k + len(seats) < len(run) else (t if t is not None else end)
                rh = next((r for r in rhythms if b + sum(r) <= min(nxt, end - 0.5) + 1e-6), None)
                if rh:
                    out.append((b, seat, rh))
            run = []
        if t is not None:
            run.append(t)
    return out


def build(bm):
    S = K.Score(bm.cut, bm)
    for pn in ("cb_q", "vc_q", "vla_q", "vln2_q", "vln1_q"):
        S.add("fb_" + pn, pn)
    S.add("fb_pno", "piano")
    S.add("fb_hn", "hn")
    S.add("fb_hn2", "hn2")
    S.add("fb_hnfar", "hn_far")
    S.add("fb_hnfarther", "hn_farther")
    end = bm.bars * 4
    secs = bm.sections
    brs = sorted((b0, b1) for b0, b1, _ in S.breaths)
    blooms = []                                   # beats where a dawn blooms (for the dawn CALL)
    # ---- the drone, section by section (legato across section joins), cut by every breath
    prev = None
    for i, s in enumerate(secs):
        tags = set(s.get("tags", []))
        lvl = LEVEL.get(s.get("level", "pp"), 0.2)
        b0, b1 = s["b0"], s["b1"]
        if tags & {"black", "silence"}:
            prev = None
            continue
        race = bool(tags & {"race", "edge", "brink"})
        # a race section cut to silence or black (not the brink, which has its suck) falls away over its last two
        # beats instead of swelling into the black (C11: "at the slip one glass tone falls away into silence")
        falls_away = i + 1 < len(secs) and bool(set(secs[i + 1].get("tags", [])) & {"silence", "black"})
        dawn = "dawn" in tags
        voices = _voices(tags, lvl)
        ptags = set(prev.get("tags", [])) if prev else set()
        pvoices = _voices(ptags, LEVEL.get(prev.get("level", "pp"), 0.2)) if prev else voices
        # pieces of the section between breaths: [(start, end, after_breath)]
        cuts = [(x0, x1) for x0, x1 in brs if b0 - 1e-6 <= x1 <= b1 + 1e-6]
        pieces, t, fresh = [], b0 + (0.0 if prev else 0.25), prev is None
        for x0, x1 in cuts:
            if x1 <= t + 1e-6:                    # a breath ending on the section's first beat
                fresh = True
                t = max(t, x1)
                continue
            pieces.append((t, x0 + 0.02, fresh))
            t, fresh = x1, True
        pieces.append((t, b1 + (0.06 if i + 1 < len(secs) else -0.5), fresh))
        lvl_s = lvl
        for k, (p0, p1, fr) in enumerate(pieces):
            if p1 - p0 < 0.5:
                continue
            bloom = dawn and fr and (k > 0 or any(abs(x1 - p0) < 1e-3 for _, x1 in brs) or not prev)
            # before an interior breath the previous section's voicing holds (C22: the tritone until the Ring goes)
            vs = pvoices if (k == 0 and len(pieces) > 1 and not fr) else voices
            for pn, p in vs:
                K.held(S, pn, p, p0, p1, bow=8.0, sync_first=bloom)
            lvl = lvl_s * (0.84 if (len(vs) >= 5 and "brink" not in tags) else 1.0)   # 5 voices sound louder than 3
            for pn, _ in vs:
                if dawn and k == 0 and len(pieces) > 1:
                    # a dawn with its sunrise inside it (B's hand-back): the whole range first, pp -> mf, then the
                    # east greys (subito p, a bar before the breath), then the breath and the bloom
                    S.P(pn).d((p0, lvl * 0.5), (p1 - 4.6, min(0.55, lvl * 1.4)), (p1 - 3.7, lvl * 0.72),
                              (p1 - 0.3, lvl * 0.8))
                elif bloom:                       # warm, never loud: the bloom swells a little, then settles
                    S.P(pn).d((p0, lvl * 0.7), (p0 + 1.4, lvl * 1.1), (p0 + 8, lvl), (p1 - 2, lvl))
                elif "title" in tags:
                    S.P(pn).d((p0, lvl), (p1 - 3, lvl * 0.4), (p1 - 0.5, 0.03))
                elif "brink" in tags:             # the cut's loudest
                    S.P(pn).d((p0, lvl), (p1 - 1.6, min(0.8, lvl * 1.5)), (p1 - 0.3, lvl * 0.8))   # then the suck
                elif race and falls_away:         # C11: the slip (two beats before the black) falls away into
                    S.P(pn).d((p0, lvl), (p1 - 2.0, min(0.5, lvl * 1.4)), (p1 - 0.15, 0.03))   # the silence
                elif race:
                    S.P(pn).d((p0, lvl), (p1 - 0.5, min(0.5, lvl * 1.4)))
                else:
                    S.P(pn).d((p0, lvl * (0.5 if fr else 1.0)), (p0 + 2, lvl), (p1 - 1, lvl))
            if bloom:
                blooms.append(p0)
        if dawn and not blooms:
            blooms.append(b0)
        prev = s
    # ---- the brink: the corrupted call (D Ab D, the race's own figure) on both near horns, swelling into the suck
    for s in secs:
        if "brink" in set(s.get("tags", [])):
            for k, (seat, root) in enumerate((("fb_hn", "D3"), ("fb_hn2", "D3"))):
                b = s["b0"] + 0.5 + 2 * k
                n = int((s["b1"] - 1.0 - b) // 4)
                for j in range(max(0, n)):
                    K.corrupted_call(S, seat, root, b + 4 * j, rhythm=(1, 1, 2), vel=0.5, sync_first=True)
    # ---- the silence piano, after two seconds of true silence
    for s in bm.with_tag("silence"):
        if s["b1"] - s["b0"] >= 6:
            K.silence_piano(S, "fb_pno", s["b0"] + 3.0, damp=s["b1"] - 0.25, vels=(0.18, 0.16, 0.17))
    # ---- the CALL on a near horn at every lit fire (passed between two horns); the ANSWER on far horns
    dawn_calls = [round(x + 8, 6) for x in blooms]
    calls = sorted(set([round(e["beat"] + 1.0, 6) for e in bm.find(tag="call")] + dawn_calls +
                       [round(e["beat"], 6) for e in bm.find(tag="night1_call")]))     # B: on the event itself
    night = {round(e["beat"], 6) for e in bm.find(tag="night1_call")}
    for b, seat, rh in _passes(calls, ("fb_hn", "fb_hn2"), CALL_RH, end):
        if b in night:
            rh = CALL_RH[1]                       # one bar, then the bar of silence where an answer should be
        sec = bm.sec_at(b)
        # the velocity is the player's dynamic (and picks the sample layer: kept >= 0.36 so the soft layer's slow
        # attack never smears the entry); loudness under a crowd of passed calls comes off the note gain instead
        v = 0.36 if sec.get("level", "pp") in ("pp", "ppp", "p") else 0.42
        crowd = sum(1 for c in calls if c != b and abs(c - b) < 8)     # calls passed peak to peak overlap
        K.call(S, seat, "D3", b, rhythm=rh, vel=v, sync_first=True,
               gain_db=(-3.0 if crowd >= 2 else -1.5 if crowd else 0.0) - (2.0 if b in dawn_calls else 0.0))
        S.sync.append((b * 60 / 72, f"CALL in {sec['id']} (bar {int(b // 4) + 1} b{b % 4 + 1:.2f})", seat, 0.25,
                       "arrive:50"))
    answers = sorted(set(round(e["beat"] + 0.5, 6) for e in bm.find(tag="answer")))
    for j, (b, seat, rh) in enumerate(_passes(answers, ("fb_hnfar", "fb_hnfarther"), ANS_RH, end)):
        sec = bm.sec_at(b)
        mode = "major" if set(sec.get("tags", [])) & {"major", "dawn", "title"} else "minor"
        K.answer(S, seat, "D4", b, mode=mode, rhythm=rh, vel=0.36, sync_first=True, gain_db=-3.0 - 1.0 * min(j, 5))
    for pn in ("fb_hn", "fb_hn2", "fb_hnfar", "fb_hnfarther"):
        K.breathe(S, pn, 0, end, depth=0.15, min_dur=1.5)
    # the ride: the strings and the piano play softer; the horns keep their dynamic (and so their sample layer and
    # its measured attack) and take the section's ride on the note gain instead
    ride = FALLBACK_RIDE.get(bm.cut, {})
    horns = ("fb_hn", "fb_hn2", "fb_hnfar", "fb_hnfarther")
    K.ride(S, bm, ride, skip=horns)
    for pn in horns:
        for n in S.P(pn).notes:
            n.gain_db += ride.get(bm.sec_at(n.start)["id"], 0.0)
        K.anticipate(S, pn)
    # ---- the analysis contract
    for s in secs:
        tags = set(s.get("tags", []))
        key = "black" if "black" in tags else s.get("level", "pp")
        S.levels.append((s["id"], *(s.get("rel") or REL_BAND.get(key, (-30, 0)))))
    return S
