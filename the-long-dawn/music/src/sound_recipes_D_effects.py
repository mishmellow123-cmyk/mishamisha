"""Opt-in D effects recipe assembly, Phase 1 (no render or shared-cache writes).

Use assemble(allow_partial=True) to inspect the reusable recipes. Normal assembly
refuses unbound new pictures/edit selections. The result is a render plan, NOT a
sound_v3 module: BED_CROPS preserve the full donor rendering context and PCM_COPIES
must bypass D mastering. Neither can honestly be represented as EXTRA_BEDS.

The score lane owns sound_recipes_D.py; this module imports no D scaffold.
"""
from copy import deepcopy

import sound_d_table as T


def assemble(table=None, *, allow_partial=False):
    table = T.build() if table is None else table
    bad = T.problems(table)
    if bad:
        raise ValueError("REFUSED: " + "; ".join(bad))
    pending = [r["id"] for r in table["requests"]]
    pending += [r["id"] for r in table["reuse"] if r["status"] == "awaiting_edit"]
    if pending and not allow_partial:
        raise ValueError("REFUSED: D effects are Phase 1; unbound rows: " + ", ".join(pending))
    import sound_recipes_A as A
    import sound_recipes_C as C
    from sound_c5_recipes import _recipe as c_recipe
    from sound_d_designs import design

    recipes, events, beds, pcm = {}, [], [], []
    for r in table["reuse"]:
        if r["kind"] == "pcm_copy":
            pcm.append(deepcopy(r))
            continue
        if r["status"] == "awaiting_edit":
            continue
        donor = r["donor"]
        if r["donor_cut"] == "C5P2":
            recipe = c_recipe(donor)
            space = C.SPACE
        else:
            recipe = deepcopy(A.RECIPES[donor["id"]])
            space = A.SPACE
        recipes[r["id"]] = recipe
        common = dict(id=r["id"], seed_id=r["seed_id"], source=r["source"],
                      source_sync=r["source_sync"], space=deepcopy(space), review=r["review"])
        if r["kind"] == "event":
            events.append(dict(common, t=r["hit_f"] / T.FPS, hit_f=r["hit_f"],
                               donor_hit_f=donor["hit_f"], delta_f=r["delta_f"]))
        else:
            # Preserve original level matching, fades and envelope before clipping.
            # Also preserve filters/reverb history; a new short D bed is not equivalent.
            beds.append(dict(common, f0=r["f0"], f1=r["f1"], donor_crop=deepcopy(r["donor_crop"]),
                             donor_row=deepcopy(donor), reference_cut="A" if r["donor_cut"] == "AP2" else "C5P2",
                             render_mode=r["render_mode"],
                             crop_stage="after full donor bed processing, including its space/history"))
    return dict(RECIPES=recipes, EXTRA_EVENTS=events, BED_CROPS=beds, PCM_COPIES=pcm,
                NEW_DESIGNS={r["id"]: dict(request=deepcopy(r), design=design(r["design"]))
                             for r in table["requests"]},
                PENDING=pending, CONSTRAINTS=deepcopy(table["constraints"]),
                phase=1, deliverable=False,
                note="Recipe data only; no audio rendered, new D sync measured, or mix verified")
