"""Candidate-only 24-frame lit lead-in for C5 Cold; shutdown stays C3848.

The director's study borrows C3816..3839 from the Map hold. This module only
renders the proposed additional material; the edit decision is external.
Accepted frames use the original driver, without any schedule substitution.
"""
from contextlib import contextmanager
import threading

import c_v5 as base

FIRST = 3816
VARIANTS = ('accepted', 'lead24')
_LOCK = threading.Lock()


@contextmanager
def _extended_cold():
    # The inherited frame method uses both shot_at() and SHOTS to bound its
    # shutter. Limit the substitution to a single call in this candidate.
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('Cold lead-in renderer is not reentrant')
    original = base.SHOTS
    try:
        base.SHOTS = {**original, 'cold': (FIRST, original['cold'][1])}
        yield
    finally:
        base.SHOTS = original
        _LOCK.release()


class Scene(base.Scene):
    def __init__(self, variant='accepted'):
        if variant not in VARIANTS:
            raise ValueError(f'Unknown Cold study: {variant}')
        self.variant = variant
        super().__init__()

    def frame(self, frame, scale=.5):
        if self.variant == 'lead24' and FIRST <= frame < 3840:
            with _extended_cold():
                return super().frame(frame, scale)
        return super().frame(frame, scale)
