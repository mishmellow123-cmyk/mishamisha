"""THE LONG DAWN v3 - SOUND for C5 under score pass 2 (the v5.2 film): every effect from the MEASURED sound table
(sound_c5_table.build(), snapshot in music/sound/c5_sound_events.json), each played by the recording SOUND-C chose and
approved for the same kind of event (sound_recipes_C), at SOUND-C's approved level. Pass 1's twin is sound_recipes_C5.

    python sound_v3.py C5P2                                  # the whole effects stem, in sync, mastered on the saved
                                                             # premaster_score_final_C5P2.npy (refused if stale)
    LD_SOUND_ALLOW_UNRESOLVED=1 python sound_v3.py C5P2      # render what is ready; print every row it skips

It refuses to import while any row lacks a recording or a frame, unless told to skip them.
The owner's in-step revision keeps the forge sounding across 3848 and through C17; its
bed and synchronized hammers bypass the unchanged score's shutdown breath.

The sound master's three edges (lane polishcsound, 30 Sep: the riffle 13.7 s, the shutdown 160.3 s, the sunrise's
breath 196.4 s) are polish_c5_sound's: the score releases on 3848 and through the dawn's dissolve, re-mixed only
inside its windows, and the master outside them is the delivered one, sample for sample (pinned references in
music/cache/v3, polish_c5_reference.json).
"""
import sound_c5_table as T
import sound_recipes_C as RC
from sound_c5_recipes import assemble

TABLE = T.build("C5P2")
RECIPES, EXTRA_EVENTS, EXTRA_BEDS, SKIPPED, DESIGN = assemble(TABLE, "C5P2")
SPACE = dict(RC.SPACE)
SILENCE = []

# the bounded premaster polish and the exact-exterior master (sound_v3.write's optional hooks; C5P2 only)
from polish_c5_sound import prepare_master, bound_master, refresh_breath_probes  # noqa: E402,F401
