"""Dump the world's timed events (for the score) as JSON."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import world as Wd  # noqa: E402

C.reset(samples=1, width=32, height=16)
w = Wd.World()
ev = {
    "t_drop": Wd.T_DROP, "t_tree": Wd.T_TREE, "t_answer": Wd.T_ANSWER, "t_coda": Wd.T_CODA,
    "t_black": Wd.T_BLACK, "t_sphere": w.t_sphere,
    "births": sorted(float(x) for x in (w.T_arr[:, 0])),
    "drift": [{"t": d["t0"], "word": d["ob"].data.body, "pan": math.cos(d["th"])} for d in w.drift],
    "branches": [], "answer": [],
}
for wd in w.tree_words:
    b = wd["b"]
    pan = max(-1.0, min(1.0, float(wd["anchor"][0]) / 4.0))
    if b.chosen:
        ev["answer"].append({"t": w.word_times.get(b, Wd.T_ANSWER), "word": b.word, "pan": pan})
    else:
        ev["branches"].append({"t": float(wd["t_grow"]), "word": b.word, "pan": pan, "depth": b.depth})
ev["answer"].sort(key=lambda e: e["t"])
json.dump(ev, open(sys.argv[1], "w"), indent=1)
print("answer:", [(round(e["t"], 2), e["word"]) for e in ev["answer"]])
print("sphere", round(w.t_sphere, 2), "branch words", len(ev["branches"]), "drift", len(ev["drift"]))
