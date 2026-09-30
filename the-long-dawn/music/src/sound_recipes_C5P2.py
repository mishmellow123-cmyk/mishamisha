"""THE LONG DAWN v3 - SOUND for C5 under score pass 2 (the v5.2 film): every effect from the MEASURED sound table
(sound_c5_table.build(), snapshot in music/sound/c5_sound_events.json), each played by the recording SOUND-C chose and
approved for the same kind of event (sound_recipes_C), at SOUND-C's approved level. Pass 1's twin is sound_recipes_C5.

    python sound_v3.py C5P2                                  # the whole effects stem, in sync, mastered on the saved
                                                             # premaster_score_final_C5P2.npy (refused if stale)
    LD_SOUND_ALLOW_UNRESOLVED=1 python sound_v3.py C5P2      # render what is ready; print every row it skips

It refuses to import while any row lacks a recording or a frame, unless told to skip them.
The owner's in-step revision keeps the forge sounding across 3848 and through C17; its
bed and synchronized hammers bypass the unchanged score's shutdown breath.
"""
import sound_c5_table as T
import sound_recipes_C as RC
from sound_c5_recipes import assemble

TABLE = T.build("C5P2")
RECIPES, EXTRA_EVENTS, EXTRA_BEDS, SKIPPED, DESIGN = assemble(TABLE, "C5P2")
SPACE = dict(RC.SPACE)
SILENCE = []
