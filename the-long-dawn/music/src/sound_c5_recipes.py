"""THE LONG DAWN v3 - SOUND for C5: the C5 sound table's rows -> sound_v3.build's recipes, events and beds.
Shared by sound_recipes_C5P2 (score pass 2) and sound_recipes_C5 (pass 1), so that importing one never runs the
other's table.
"""
import copy
import os

import sound_c5_table as T
import sound_recipes_C as RC


# C5P2's new effects use recordings already in cache/sound. These two Lokomo anvil takes start
# on their attack (2 ms energy-envelope onset at sample 0); the four-hit take is cropped
# before its second blow. The pair strikes together, with no delay or breathing wet send.
INSTEP_RECIPES = {
    "C5.hammer.cached": dict(src=("fs:386116", 0.0), pre=0.0, post=1.4, hp=90,
                             fi=0.001, fo=0.4, width=0.4, dist=0.5),
    "C5.hammer.instep": dict(layers=[
        dict(src=("fs:386116", 0.0), pre=0.0, post=0.65, fi=0.001, fo=0.20,
             hp=90, lp=6000, width=0.55, g=0.0),
        dict(src=("fs:386115", 0.0), pre=0.0, post=0.65, fi=0.001, fo=0.20,
             hp=90, lp=6000, width=0.9, g=-4.0)], send=0.0, no_breath=True),
    "C5.forge.steady": dict(layers=[dict(src=RC.FIRE, seg=(10, 18), xf=2.5, g=0.0),
                                    dict(src=RC.LAVA, seg=(8, 14), xf=2.0, g=-12.0)],
                            hp=50, lp=6000, width=0.65, crest=10.0, send=0.0, no_breath=True),
}


def _recipe(r):
    if r["recipe"] == "pen":
        rc = RC.pen(r["src_start"], r["dur_f"] / T.FPS)
    else:
        rc = copy.deepcopy((INSTEP_RECIPES if r["recipe"] in INSTEP_RECIPES else RC.RECIPES)[r["recipe"]])
    for k in ("trim", "level_from", "skip", "why"):          # the level below already includes SOUND-C's trim
        rc.pop(k, None)
    rc["level"] = r["level"] + r.get("trim_db", 0.0)
    if r.get("pan") is not None:
        rc["pan"] = r["pan"]
    if r.get("post_max_f") is not None:                       # cut mid-stroke on the shutdown
        rc["post"] = min(rc.get("post", 1.0), r["post_max_f"] / T.FPS)
        rc["fo"] = min(rc.get("fo", 0.08), 0.02)
    if r.get("env_after"):                                    # a shaped tail over every layer (sound_v3, 29 Sep)
        rc["env_after"] = r["env_after"]
    if r.get("no_breath"):                                    # heard through the score's breath (sound_v3, 29 Sep)
        rc["no_breath"] = True
    for k in ("send", "dist", "width"):
        if k in r:
            rc[k] = r[k]
    return rc


def assemble(table, label):
    """-> (RECIPES, EXTRA_EVENTS, EXTRA_BEDS, SKIPPED, DESIGN) for sound_v3.build; raises SystemExit on a table that
    breaks its contract, or (unless LD_SOUND_ALLOW_UNRESOLVED is set) on any row that is not ready"""
    bad = T.problems(table)
    if bad:
        raise SystemExit(f"REFUSED: the {label} sound table breaks its own contract:\n  " + "\n  ".join(bad))
    recipes, events, beds, skipped, design = {}, [], [], [], []
    for r in table["events"]:
        if r["kind"] == "silence":
            continue
        if not r["status"].startswith("ready"):
            skipped.append(r)
            continue
        if r.get("design"):
            design.append(r["id"])
        recipes[r["id"]] = _recipe(r)
        if r["kind"] == "bed":
            b = dict(id=r["id"], t0=r["f0"] / T.FPS, t1=r["f1"] / T.FPS, fade_in=r.get("fade_in", 1.5),
                     fade_out=r.get("fade_out", 1.5))
            if r.get("env_f"):
                b["env"] = [[(f - r["f0"]) / T.FPS, db] for f, db in r["env_f"]]
            beds.append(b)
        else:
            events.append(dict(id=r["id"], t=r["hit_f"] / T.FPS))
    if skipped:
        lines = [f"{r['id']}: {r['status']}" for r in skipped]
        if not os.environ.get("LD_SOUND_ALLOW_UNRESOLVED"):
            raise SystemExit(f"REFUSED: {label} sound rows not ready (resolve them, or set LD_SOUND_ALLOW_UNRESOLVED=1 "
                             "to render without them):\n  " + "\n  ".join(lines))
        print(f"SOUND {label}: SKIPPING rows that are not ready:\n  " + "\n  ".join(lines), flush=True)
    if design:
        print(f"SOUND {label}: design choices rendered unheard (listen): " + ", ".join(design), flush=True)
    return recipes, events, beds, skipped, design
