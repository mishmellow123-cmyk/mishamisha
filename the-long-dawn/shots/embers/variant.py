"""Which cut of THE LONG DAWN v2 is being rendered (render.py sets CUT before the scene is built).

    A  THE ALLEGORY  towers of embers + the vortex fix, text-band calm in the text windows  -> renders/embers_v2
    B  THE LEGEND    as A, but wordless: no text-band attenuation anywhere                 -> renders/embers_B
    C  TOLKIEN       as A (with the text band) + the Ring, the Eye, the grasp on the Ring   -> renders/embers_C
"""
CUT = 'A'


def set_cut(c):
    global CUT
    CUT = c.upper()
    assert CUT in ('A', 'B', 'C'), CUT


def text_band():
    return CUT != 'B'


def tolkien():
    return CUT == 'C'
