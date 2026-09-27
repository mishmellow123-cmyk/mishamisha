"""Which cut of THE LONG DAWN v2 is being rendered (render.py sets CUT before the scene is built).

    A  THE ALLEGORY  towers of embers + the vortex fix, text-band calm in the text windows  -> renders/embers_v2
    B  THE LEGEND    as A, but wordless: no text-band attenuation anywhere                 -> renders/embers_B
    C  TOLKIEN       as A (with the text band) + the Ring, the Eye, the grasp on the Ring   -> renders/embers_C
"""
CUT = 'A'


def set_cut(c):
    global CUT
    CUT = c.upper()
    assert CUT in ('A', 'B', 'C', 'A3', 'C3'), CUT


def text_band():
    return CUT != 'B'


def tolkien():
    return CUT in ('C', 'C3')


# v3 (director's switch, A only): MAIN = "two giants, uncoded"; ALT = "coded pair" (the giants dressed as a pagoda
# and an obelisk; everything else identical) -> renders/embers_v2_alt_codedtowers
TOWERS_ALT = False


def giants():
    """the two towers that outgrow the rest (A only; C keeps many equal towers)"""
    return (2, 6) if CUT in ('A', 'A3') else ()


def fire_side_only():
    """A: every tower is lit only on the face turned to the fire, black toward the others (BIBLE_V3 A7 / E5).
    C3 (EMBERS-C, director 27 Sep: "use A's approved forge family: charcoal crust, banded stacks, glowing throats")
    takes the same look"""
    return CUT in ('A', 'A3', 'C3')
