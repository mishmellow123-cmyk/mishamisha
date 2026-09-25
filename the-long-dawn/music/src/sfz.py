"""Minimal SFZ parser for the VSCO-2-CE mappings (SFZ branch).

Parses <control>/<global>/<group>/<region> with opcode inheritance and
returns a flat list of Region dicts with absolute sample paths.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC = os.path.dirname(HERE)
SAMPLES = os.path.join(MUSIC, "samples")
VSCO = os.path.join(SAMPLES, "VSCO-2-CE")
SFZDIR = os.path.join(SAMPLES, "sfz")

_OPC = re.compile(r"([A-Za-z0-9_]+)=(.*?)(?=\s+[A-Za-z0-9_]+=|$)")
NOTE_NAMES = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def _keyval(v):
    """SFZ key values may be MIDI numbers or note names (c4 = 60)."""
    v = v.strip()
    try:
        return int(v)
    except ValueError:
        m = re.match(r"([a-gA-G])([#b]?)(-?\d+)", v)
        if not m:
            raise
        n = NOTE_NAMES[m.group(1).lower()]
        if m.group(2) == "#":
            n += 1
        elif m.group(2) == "b":
            n -= 1
        return n + 12 * (int(m.group(3)) + 1)


def parse_sfz(name):
    path = os.path.join(SFZDIR, name if name.endswith(".sfz") else name + ".sfz")
    text = open(path, encoding="utf-8", errors="replace").read()
    control, glob, group = {}, {}, {}
    regions = []
    cur = None
    section = None
    for raw in text.splitlines():
        line = raw.split("//")[0].strip()
        if not line:
            continue
        # headers may share a line with opcodes
        while line.startswith("<"):
            end = line.index(">")
            hdr = line[1:end]
            line = line[end + 1:].strip()
            if cur is not None:
                regions.append(cur)
                cur = None
            section = hdr
            if hdr == "group":
                group = {}
            elif hdr == "global":
                glob = {}
            elif hdr == "region":
                cur = {}
        if not line:
            continue
        for k, v in _OPC.findall(line):
            v = v.strip()
            if section == "control":
                control[k] = v
            elif section == "global":
                glob[k] = v
            elif section == "group":
                group[k] = v
            elif section == "region":
                cur[k] = v
        if section == "region" and cur is not None:
            # inherit at close time; store snapshot of group/global refs
            cur.setdefault("__group", dict(group))
            cur.setdefault("__global", dict(glob))
    if cur is not None:
        regions.append(cur)

    base = control.get("default_path", "").replace("\\", "/")
    out = []
    for r in regions:
        d = {}
        d.update(r.pop("__global", {}))
        d.update(r.pop("__group", {}))
        d.update(r)
        if "sample" not in d:
            continue
        sp = os.path.join(VSCO, base, d["sample"].replace("\\", "/"))
        if not os.path.exists(sp):
            # some maps have case differences
            alt = _find_ci(sp)
            if alt is None:
                continue
            sp = alt
        key = _keyval(d["key"]) if "key" in d else None
        reg = dict(
            path=sp,
            lokey=_keyval(d.get("lokey", key if key is not None else 0)),
            hikey=_keyval(d.get("hikey", key if key is not None else 127)),
            center=_keyval(d.get("pitch_keycenter", key if key is not None else 60)),
            lovel=int(d.get("lovel", 0)),
            hivel=int(d.get("hivel", 127)),
            volume=float(d.get("volume", 0.0)),
            tune=float(d.get("tune", 0.0)),
            transpose=int(d.get("transpose", 0)),
            seq_length=int(d.get("seq_length", 1)),
            seq_position=int(d.get("seq_position", 1)),
            release=float(d.get("ampeg_release", 0.5)),
            offset=int(d.get("offset", 0)),
            pitch_keytrack=float(d.get("pitch_keytrack", 100)),
        )
        out.append(reg)
    return out


def _find_ci(path):
    d, f = os.path.split(path)
    if not os.path.isdir(d):
        return None
    fl = f.lower()
    for g in os.listdir(d):
        if g.lower() == fl:
            return os.path.join(d, g)
    return None


if __name__ == "__main__":
    import sys
    import soundfile as sf
    for name in sys.argv[1:]:
        regs = parse_sfz(name)
        keys = sorted(set((r["lokey"], r["hikey"], r["center"]) for r in regs))
        vels = sorted(set((r["lovel"], r["hivel"]) for r in regs))
        rr = sorted(set(r["seq_position"] for r in regs))
        durs = [sf.info(r["path"]).duration for r in regs]
        lo = min(r["lokey"] for r in regs)
        hi = max(r["hikey"] for r in regs)
        print(f"{name}: {len(regs)} regions keys {lo}-{hi} centers "
              f"{sorted(set(r['center'] for r in regs))} vel {vels} rr {rr} "
              f"dur {min(durs):.2f}-{max(durs):.2f}s rel {regs[0]['release']}")
