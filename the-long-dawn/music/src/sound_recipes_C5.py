"""THE LONG DAWN v3 - SOUND for C5 under score pass 1 (score_v3_C5, barmap_C5).

The effects follow the picture, and pass 1 and pass 2 share one picture, so every row is sound_recipes_C5P2's except
the hammers: they are the music's (no strike is drawn), and pass 1's trap ostinato and lone hammer sit on barmap_C5's
frames, so here they follow barmap_C5's pulse (sound_c5_table.build("C5")). `python sound_v3.py C5` masters the stem
on pass 1's saved premaster (premaster_score_final_C5.npy; checked for length, and for its identity sidecar when it
has one).
"""
import sound_c5_table as T
import sound_recipes_C as RC
from sound_c5_recipes import assemble

TABLE = T.build("C5")
RECIPES, EXTRA_EVENTS, EXTRA_BEDS, SKIPPED, DESIGN = assemble(TABLE, "C5")
SPACE = dict(RC.SPACE)
SILENCE = [(T.SHUTDOWN / T.FPS, T.RING_CUT / T.FPS)]

__all__ = ["DESIGN", "EXTRA_BEDS", "EXTRA_EVENTS", "RECIPES", "SILENCE", "SKIPPED", "SPACE", "TABLE"]
