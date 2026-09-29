"""THE LONG DAWN v3 - stale-artefact guards for the music/sound pipeline (SOUND-SCORE-C, 29 Sep 2026).

Three silent failures this module turns into loud ones:

  1. A SAVED pre-master score of another length mastered as this cut. sound_v3.write() loaded
     cache/v3/premaster_score_final_<cut>.npy and sliced score[:n]: a 7,200-frame premaster (14,400,000 samples)
     under a 5,920-frame bar map (11,840,000) was truncated without a word. check_premaster() requires EXACTLY
     the cut's length.
  2. A pre-master rendered from another bar map or score module mastered as if current (a stale file with the right
     length: pass 1 and pass 2 of C5 are both 5,920 frames). render_v3.render() now writes an identity sidecar
     (premaster_score_<name>.json: cut, frames, samples, the sha256 of the bar map, cue sheet and score module that
     made it); check_premaster() refuses a sidecar that disagrees with the current files. Cuts in STRICT must have
     the sidecar; the older cuts only warn when it is absent (their premasters predate it).
  3. The retired 7,200-frame C's files picked up for the 5,920-frame v5.2 film because their names are plausible
     (final_C.wav, sound_C.wav, premaster_score_final_C.npy, C_master*.mp4). check_cut_allowed() refuses the retired
     cut unless the caller says so explicitly; check_wav() and check_video_frames() refuse a file of the wrong length.

The adopted behaviour of A and B is unchanged: their files have their cut's exact length, and no flag is needed.
"""
import datetime
import hashlib
import json
import os

SR = 48000
FR = SR // 24                      # samples per frame
RETIRED = {"C": "the 7,200-frame C (v1) is retired: its final_C / sound_C / premaster_score_final_C files belong "
                "to the old cut. The v5.2 film is C5 (score pass 1) or C5P2 (pass 2)"}
STRICT = {"C5P2"}                  # cuts born with the identity sidecar: a premaster without one is refused


class StaleArtefact(RuntimeError):
    """a file whose name says one cut and whose content (or provenance) says another"""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sidecar(npy_path):
    return os.path.splitext(npy_path)[0] + ".json"


def write_premaster_identity(npy_path, name, cut, frames, samples, barmap_path, cues_path=None, score_path=None):
    """record which cut, bar map, cue sheet and score module a saved pre-master came from (additive: the .npy is
    untouched)"""
    d = dict(schema="long-dawn/premaster-identity/1", name=name, cut=cut, frames=int(frames), samples=int(samples),
             barmap=os.path.basename(barmap_path), barmap_sha256=sha256_file(barmap_path),
             cues=os.path.basename(cues_path) if cues_path and os.path.exists(cues_path) else None,
             cues_sha256=sha256_file(cues_path) if cues_path and os.path.exists(cues_path) else None,
             score=os.path.basename(score_path) if score_path else None,
             score_sha256=sha256_file(score_path) if score_path and os.path.exists(score_path) else None,
             utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    with open(sidecar(npy_path), "w") as fh:
        json.dump(d, fh, indent=1)
    return d


def npy_length(npy_path):
    import numpy as np
    return int(np.load(npy_path, mmap_mode="r").shape[0])


def check_premaster(npy_path, cut, frames, barmap_path, cues_path=None, allow_unverified=False):
    """-> [warnings]; raises StaleArtefact unless the saved pre-master is provably this cut's current one"""
    n = frames * FR
    got = npy_length(npy_path)
    base = os.path.basename(npy_path)
    if got != n:
        raise StaleArtefact(f"{base} holds {got} samples ({got / FR:.1f} frames); cut {cut} is {frames} frames = {n} "
                            f"samples. A premaster of another length is another cut (the retired 7,200-frame C is "
                            f"14,400,000): re-render the score for {cut}, never truncate or pad it")
    sc = sidecar(npy_path)
    if not os.path.exists(sc):
        msg = (f"{base} has no identity sidecar ({os.path.basename(sc)}): its bar map and score module cannot be "
               f"checked (it predates the guard, or was copied without it)")
        if cut in STRICT and not allow_unverified:
            raise StaleArtefact(msg + f". {cut} requires it: re-render with render_v3.py {cut}, or pass "
                                "--premaster-unverified after checking the file by other means")
        return ["WARNING: " + msg]
    d = json.load(open(sc))
    bad = []
    if d.get("cut") != cut:
        bad.append(f"it was rendered for cut {d.get('cut')!r}")
    if d.get("samples") != n or d.get("frames") != frames:
        bad.append(f"its sidecar says {d.get('frames')} frames / {d.get('samples')} samples")
    if d.get("barmap_sha256") != sha256_file(barmap_path):
        bad.append(f"its bar map ({d.get('barmap')}) has changed since (sha256 differs from "
                   f"{os.path.basename(barmap_path)} now)")
    if cues_path and os.path.exists(cues_path) and d.get("cues_sha256") not in (None, sha256_file(cues_path)):
        bad.append(f"its cue sheet ({d.get('cues')}) has changed since")
    if bad:
        raise StaleArtefact(f"{base} is stale for {cut}: " + "; ".join(bad) + f". Re-render: render_v3.py {cut}")
    return []


def check_cut_allowed(cut, allow_retired=False):
    if cut in RETIRED and not allow_retired:
        raise StaleArtefact(f"cut {cut}: {RETIRED[cut]}. Re-running it rewrites the retired files the edit may pick "
                            f"up by name; pass --retired-c-7200 only to reproduce the old cut on purpose")


def wav_frames(path):
    import soundfile as sf
    info = sf.info(path)
    if info.samplerate != SR:
        raise StaleArtefact(f"{os.path.basename(path)}: {info.samplerate} Hz, not {SR}")
    return info.frames / FR


def check_wav(path, frames):
    """a master (or stem) must be exactly the cut's length: 5,920 frames = 11,840,000 samples for the v5.2 C"""
    got = wav_frames(path)
    if abs(got - frames) > 1e-9:
        raise StaleArtefact(f"{os.path.basename(path)} is {got:.2f} frames; this cut is {frames}: it belongs to "
                            f"another cut (the retired C is 7,200). Adopt an explicit C5 file instead")
    return True


def check_video_frames(n, cut, expected):
    """sync_audit_v3: the delivery master it picked (newest *_master*.mp4) must have the frames its table expects"""
    if expected is not None and n != expected:
        raise StaleArtefact(f"the {cut} master decodes to {n} frames, but this tool's table says {cut} is {expected}: "
                            f"either it picked another cut's master by name, or the table predates a re-edit of the "
                            f"cut (EDIT-C5 makes C the 5,920-frame v5.2 film; this tool's C tables are the retired "
                            f"7,200-frame cut's). Measure the v5.2 picture with measure_c5_events.py")
