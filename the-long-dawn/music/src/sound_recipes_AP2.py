"""A's recorded effects under score pass 2 (AP2): the effects table IS A's, read at lookup time.

    python sound_v3.py AP2      # A's effects mastered under the AP2 premaster (cache/v3/premaster_score_final_AP2.npy)

AP2's crossing pass changes the score (score_v3_AP2.py); its bar map adds two score anchors and moves none of A's
(barmap_ap2.py), so every effect of A resolves to the frame it has in A. The asound and syncA lanes re-time A's
effects in A's own files (sound_recipes_A.py, picture_sync_A.json, cues_A.json): every attribute here is forwarded to
sound_recipes_A when sound_v3 reads it, so their merged recipes, PICTURE overrides, extra beds/events, SPACE and any
optional hook (SILENCE, extra_cues, ...) reach AP2 with no copy to go stale; after a merge that touches cues_A.json,
regenerate AP2's cue sheet (python barmap_ap2.py) before rendering.

These attributes are AP2's own (sound_v3's optional hooks, inert for every other cut):
  REF_LEVEL_CUT = "A"    each cue is matched to A's cached reference level for the same cue (the same design, the same
                         numbers A's master was matched to), not recomputed under a new "AP2:" cache key
  CACHE_READ_ONLY        a cache miss is computed in memory and never written: this cut never rewrites the shared
                         caches in music/cache/sound (ref_levels_v2.json, src_loudness.json)
  PREMASTER_POLISH       A6/A7 breath recovery and dynamics in sound_polish_edge; score notes and effect recipes
                         remain the same. This bounded edit runs before peak protection and mastering.
"""
import sound_recipes_A as _A

REF_LEVEL_CUT = "A"
CACHE_READ_ONLY = True

# polishedge: recover the race breath from its cached performances and shape the
# edge's dynamics before the shared mastering chain. A and other cuts stay inert.
from sound_polish_edge import apply as PREMASTER_POLISH


def __getattr__(name):
    """everything else follows A at lookup time; an optional hook A lacks stays absent (getattr raises), so
    sound_v3's hasattr() tests see exactly what A has"""
    return getattr(_A, name)
