"""Cut D score-only draft recipe; opt in by rendering cut D explicitly.

No stand-in effects, recorded speech, or modifications to A/C's recipes.  The
first minute is the delivered AP2 SCORE stem, including its ignition polish.
Paths below are relative to the repository, never workstation-specific.
"""
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
OPENING_PCM = dict(
    path="music/out/v3/sound_AP2_score.wav",
    source="music/out/v3/sound_AP2_score.wav", f0=0, f1=1440, protected=True,
    source_f0=0, source_f1=1440, destination_f0=0,
    mode="replace_after_master", status="measured_source",
    provenance="AP2 delivered score stem; ignition polish included; no effects stem",
)
RECIPES = {}
SCORE_ONLY = True
EXTRA_EVENTS = []
EXTRA_BEDS = []
SPACE = {}
SILENCE = []
CACHE_READ_ONLY = True


def opening_path():
    return REPOSITORY / OPENING_PCM["path"]


def opening_region():
    """Return a fresh recipe; callers cannot mutate the module's contract."""
    return dict(OPENING_PCM)
