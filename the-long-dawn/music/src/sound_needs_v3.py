"""THE LONG DAWN v3 - SOUND lane: what each film needs from the real world, and the Freesound queries that look for it.

    python sound_needs_v3.py B            # run every B need's queries (cached), print the candidates
    python sound_needs_v3.py B flint      # one need
Candidates are licence-checked (CC0 / CC-BY only) and listed with Freesound's own descriptors
(loudness, centroid, pitch salience = speech/music bleed risk, flatness, dynamic range, boominess).
"""
import sys

import freesound_v3 as F

# need -> (what it must be, [queries], extra filter)
NEEDS = {
    "B": {
        "flint": ("three flint-on-steel strikes: a sharp dry chink with a short ring, no scrape (not a ferro rod)",
                  ["flint steel", "flint striker", "strike a light flint", "firesteel flint spark"], ""),
        "blow": ("a human blowing gently on embers / tinder: breath only, no voice",
                 [("blowing", "tag:fire"), ("blow", "tag:breath"), ("exhale", "tag:blow"), "blowing embers"], ""),
        "tinder": ("tinder / char cloth taking the spark and catching: tiny crackle, a soft small ignition",
                   ["tinder", "char cloth", ("lighting", "tag:campfire"), ("fire starting", ""), ("ignite", "tag:wood")], ""),
        "roar": ("a fire flaring up into a steady blaze: natural, no gas, no cartoon whoosh",
                 [("flare", "tag:fire"), ("bonfire", "tag:ignite"), ("catching", "tag:fire"), ("burst", "tag:fire"),
                  ("whoosh", "tag:bonfire")], ""),
        "fire_bed": ("a small wood fire, close, steady crackle; no voices, birds, insects or water",
                     [("campfire", "tag:crackle"), ("campfire", "tag:winter"), ("fire", "tag:snow"),
                      ("wood fire", "tag:close-up")], ""),
        "feed": ("putting wood / a log on a fire: wood knock, settle, a crackle burst",
                 [("log", "tag:fire"), ("wood", "tag:stoking"), ("adding", "tag:fire"), ("stoke", "")], ""),
        "lid": ("a small clay / ceramic lid lifted and knocked: one soft earthen knock",
                [("lid", "tag:ceramic"), ("lid", "tag:clay"), ("terracotta", ""), ("lid", "tag:pottery")], ""),
        "ember_hiss": ("a dying ember's last faint hiss / tick", [("embers", ""), ("hiss", "tag:ember"),
                       ("coals", "tag:crackle"), ("glowing", "tag:embers")], ""),
        "wind": ("wind on a high snow ridge / summit: broadband, no structures whistling, no mic buffeting",
                 ["mountain wind", "summit wind", "alpine wind ridge", "glacier wind", "arctic wind",
                  "antarctic wind"], ""),
        "spindrift": ("wind-driven snow hissing over a snow surface", ["blowing snow", "spindrift", "snow drift wind",
                      "snow blowing surface"], ""),
        "storm": ("a snow squall: strong gusty wind with driven snow", ["blizzard", "snowstorm wind",
                  "snow storm mountain"], ""),
        "dawn_air": ("a still high mountain dawn: near silence, air; NO birds, bells, aircraft, voices",
                     ["mountain silence", "quiet mountain air", "high altitude ambience", "calm snow ambience"], ""),
        "bird": ("the first bird at dawn on a mountain: one clear, simple song, no other birds",
                 ["ring ouzel", "alpine accentor", "black redstart song", "blackbird song dawn", "snow bunting song",
                  "water pipit"], ""),
        "torch": ("a hand torch taking flame from a fire: a soft catch and flutter", ["torch ignite",
                  "torch fire flame", "torch lighting"], ""),
    },
    "C": {
        "page": ("an old heavy book's page turned slowly by a hearth: parchment, no voices",
                 [("page turn", "tag:book"), ("page", "tag:old"), ("parchment", ""), ("book", "tag:page-turn")], ""),
        "pen": ("a dip pen / quill scratching on paper, close", [("quill", ""), ("dip pen", ""), ("pen", "tag:writing"),
                ("nib", "tag:paper")], ""),
        "burn": ("paper catching fire and crackling / curling as it burns", [("paper", "tag:burning"),
                 ("burning paper", ""), ("paper", "tag:fire")], ""),
        "anvil": ("an anvil ring / hammer on anvil, a smithy (C8's tick)", [("anvil", ""), ("anvil", "tag:blacksmith")], ""),
        "hiss": ("hot metal into snow or water: a sharp hiss and steam", [("quench", ""), ("hot metal", "tag:water"),
                 ("sizzle", "tag:steam"), ("hiss", "tag:quench")], ""),
        "cock": ("a rooster crowing far away, soft, at dawn", [("rooster", "tag:distant"), ("cock crow", ""),
                 ("rooster", "tag:dawn")], ""),
        "sea": ("gentle waves on a shore at dawn, no gulls, no people", [("waves", "tag:gentle"), ("sea", "tag:calm"),
                ("waves", "tag:shore")], ""),
        "murmur": ("a small crowd murmuring low outdoors at night, no intelligible words", [("murmur", "tag:crowd"),
                   ("walla", "tag:murmur"), ("crowd", "tag:murmur")], ""),
        "torch": ("hand torches burning / flaring, fire flutter", [("torch", "tag:burning"), ("torches", "")], ""),
        "hearth": ("a fireplace / hearth crackling indoors, quiet room", [("fireplace", "tag:crackling"),
                   ("hearth", "")], ""),
        "coldtick": ("cooling metal ticking / pinging", [("metal", "tag:cooling"), ("ticking", "tag:cooling"),
                     ("expansion", "tag:metal")], ""),
        "drop": ("a single drop into still water, close", [("drop", "tag:water-drop"), ("water drop", "tag:single"),
                 ("drip", "tag:single")], ""),
        "seethe": ("molten metal / something white-hot seething, bubbling hiss", [("molten", ""), ("lava", "tag:bubbling"),
                   ("sizzle", "tag:metal")], ""),
    },
}


def line(s):
    lic = F.licence_short(s.get("license")).replace("CC-BY ", "BY")
    return (f"{s['id']:>7} {lic:5s} {s.get('duration', 0):6.1f}s {s.get('samplerate', 0) / 1000:.0f}k/{s.get('bitdepth', 0)}"
            f" {s.get('type', ''):4s} {(s.get('filesize') or 0) / 1e6:5.1f}M r{s.get('avg_rating', 0):.1f}/{s.get('num_ratings', 0)}"
            f" dl{s.get('num_downloads', 0):<5} ps{(s.get('pitch_salience') or 0):.2f} {s.get('username', '')[:10]:10s} "
            f"{s.get('name', '')[:44]}")


def run(cut, only=None, n=8):
    for need, (what, queries, flt) in NEEDS[cut].items():
        if only and need != only:
            continue
        print(f"\n=== {cut}.{need}: {what}")
        seen = set()
        for q in queries:
            q, f2 = q if isinstance(q, tuple) else (q, "")
            res, count = F.search(q, n=n, extra_filter=" ".join(x for x in (flt, f2) if x), fields=F.FIELDS + "," + F.DESC)
            print(f"--- '{q}': {count} hits")
            for s in res:
                if s["id"] in seen:
                    continue
                seen.add(s["id"])
                print(line(s))


if __name__ == "__main__":
    for need in (sys.argv[2].split(",") if len(sys.argv) > 2 else [None]):
        run(sys.argv[1], need)
    print(f"\n# requests today {F.used_today()}")
