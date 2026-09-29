"""THE LONG DAWN v3 - SOUND for C5: the C5 sound table's rows -> sound_v3.build's recipes, events and beds.
Shared by sound_recipes_C5P2 (score pass 2) and sound_recipes_C5 (pass 1), so that importing one never runs the
other's table.
"""
import copy
import os

import sound_c5_table as T
import sound_recipes_C as RC


def _recipe(r):
    if r["recipe"] == "pen":
        rc = RC.pen(r["src_start"], r["dur_f"] / T.FPS)
    else:
        rc = copy.deepcopy(RC.RECIPES[r["recipe"]])
    for k in ("trim", "level_from", "skip", "why"):          # the level below already includes SOUND-C's trim
        rc.pop(k, None)
    rc["level"] = r["level"] + r.get("trim_db", 0.0)
    if r.get("pan") is not None:
        rc["pan"] = r["pan"]
    if r.get("post_max_f") is not None:                       # cut mid-stroke on the shutdown
        rc["post"] = min(rc.get("post", 1.0), r["post_max_f"] / T.FPS)
        rc["fo"] = min(rc.get("fo", 0.08), 0.02)
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
