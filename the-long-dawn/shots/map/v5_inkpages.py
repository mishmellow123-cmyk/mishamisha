"""Opt-in C v5.2 plates. Existing page artists/materials remain unchanged."""
import numpy as np
import pages as PG
import pen
from pen import Strokes, INK, GILT


def contour(S, points, width, seed, smooth=8):
    pen.line(S, np.asarray(points), width, seed, smooth=smooth,
             lift=(2.1, 4.5), dens=.94, thin_end=.35)


def shade(S, polygon, box, seed, angle=65, spacing=.065, tone=.8):
    pen.hatch(S, PG._poly_inside(np.asarray(polygon)), lambda x, y: np.full_like(x, tone),
              box, angle, spacing, .12, .010, seed, dens=.85,
              seg=(.45, 1.8), wob=.003, gap=.05)


def refusal():
    """A gloved offering palm; the other figure's hood and shoulder turn right.

    Space below y=18 is untouched for EDIT's provisional caption. No face or
    lettering is encoded here. Stroke timing is absolute shot time, not chunk time.
    """
    S = Strokes()
    # The open hand is horizontal, its four fingers supported together rather
    # than fanned as a greeting. The ring rests above their curved upper surface.
    sleeve = [(1.8,11.7),(4.7,11.2),(5.2,12.8),(2,13.8)]
    contour(S, sleeve, .033, 701, smooth=0)
    for q,points in enumerate([
        [(1.96,12.18),(3.0,11.96),(4.05,11.40)],
        [(2.14,13.33),(3.33,12.74),(4.68,12.28)],
        [(3.32,11.65),(3.92,11.74),(4.42,11.55)],
        [(3.19,13.20),(4.13,12.84),(4.71,12.95)],
    ]):contour(S,points,.009,702+q)
    shade(S,[(1.85,13.50),(3.83,12.6),(4.67,12.32),(5.03,12.78),(2.01,13.75)],
          (1.8,12.2,5.1,13.8),716,angle=-27,spacing=.063)
    cuff = [(4.5,11.1),(5.05,10.95),(5.65,12.55),(5.15,12.83)]
    contour(S,cuff+[cuff[0]],.021,731,0)
    hand = [(5.05,11.12),(6.10,10.74),(6.68,10.40),(7.23,10.48),
            (7.96,10.78),(9.55,10.82),(9.84,10.99),(9.54,11.16),
            (8.37,11.28),(9.54,11.24),(9.72,11.42),(9.45,11.58),
            (8.12,11.67),(9.26,11.68),(9.44,11.85),(9.19,12.03),
            (7.76,12.14),(8.83,12.11),(9.01,12.30),(8.79,12.49),
            (7.45,12.69),(6.29,12.39),(5.58,12.55)]
    contour(S,hand,.025,740,3)
    contour(S,[(5.44,11.38),(6.4,11.03),(7.1,11.18),(7.68,11.43)],.016,741)
    contour(S,[(6.23,12.2),(6.7,11.87),(7.62,11.85)],.012,742)
    contour(S,[(5.43,11.65),(5.76,12.24)],.010,743)
    for q in range(14):
        x=6.2+q*.17
        contour(S,[(x,12.28+.10*np.sin(q*.2)),(x+.25,12.46+.07*np.sin(q*.2))],.008,750+q,0)
    S.retime(.35,2.4,overlap=.28)
    k=len(S)
    theta=np.linspace(0,2*np.pi,140)
    for radius,width,layer in ((.52,.022,INK),(.43,.018,INK),(.475,.032,GILT)):
        S.add(np.column_stack([8.17+radius*np.cos(theta),10.35+radius*.9*np.sin(theta)]),
              width,layer=layer)
    S.retime(2.4,3.15,order=range(k,len(S)),overlap=.1)
    k=len(S)
    # Far shoulder and the swept back edge make the turning gesture legible.
    cloak=[(13.20,9.55),(14.79,10.12),(15.39,10.65),(15.77,11.88),(16.53,16.55),
           (15.66,16.39),(15.12,17.02),(14.55,16.71),(13.83,17.11),
           (13.04,16.81),(12.32,17.08),(11.77,16.57),
           (12.33,13.67),(11.62,12.63),(11.1,11.13),(11.77,10.78),(12.72,12.04)]
    contour(S,cloak,.025,790,smooth=3)
    hood=[(12.65,9.53),(12.12,8.96),(12.16,8.17),(12.40,7.37),(13.25,6.77),
          (14.12,7.10),(14.55,7.81),(14.76,8.52),(14.38,9.22),
          (14.63,10.16),(13.58,9.83),(12.65,9.53)]
    contour(S,hood,.024,791,smooth=3)
    # Hood opening faces away, dark cloth only: deliberately no facial marks.
    opening=[(14.10,7.80),(14.55,8.0),(14.74,8.51),(14.36,9.22),(14.20,9.63),(13.91,8.96)]
    shade(S,opening,(13.85,7.75,14.8,9.7),794,angle=78,spacing=.022,tone=1)
    shade(S,opening,(13.85,7.75,14.8,9.7),793,angle=5,spacing=.027,tone=1)
    contour(S,[(12.29,8.60),(12.74,9.05),(13.37,9.28),(13.74,9.61)],.012,795)
    contour(S,[(12.71,7.43),(12.74,8.02),(13.18,8.39)],.010,796)
    # Raised glove, wrist connected visibly to sleeve, palm toward the offer.
    raised=[(11.16,11.13),(10.70,10.45),(10.34,9.83),(10.39,9.56),
            (10.62,9.57),(10.99,10.01),(10.84,9.0),(10.68,8.35),(10.78,8.15),
            (10.97,8.22),(11.28,9.1),(11.12,8.12),(11.24,7.91),(11.43,8.02),
            (11.66,9.03),(11.60,8.13),(11.77,7.97),(11.94,8.13),(12.02,9.21),
            (12.13,8.43),(12.30,8.36),(12.43,8.51),(12.31,9.79),(11.77,10.78)]
    contour(S,raised,.020,797,2)
    contour(S,[(11.19,9.8),(11.63,9.54),(11.99,9.61)],.011,798)
    contour(S,[(11.30,10.40),(11.73,10.04)],.010,799)
    contour(S,[(11.02,11.01),(11.75,10.58)],.020,800,0)
    # Form-following drapery. Leave broad paper planes between the dark folds.
    for i,points in enumerate([
        [(13.39,10.18),(13.11,12.3),(12.47,16.65)],
        [(14.20,10.50),(13.91,13.23),(14.36,16.75)],
        [(14.65,10.77),(15.06,13.28),(15.75,16.78)],
        [(12.50,12.38),(12.97,13.01),(13.47,13.09)],
        [(11.92,11.45),(12.72,12.60),(13.42,12.68)],
    ]): contour(S,points,.019,810+i)
    S.retime(3.15,5.4,order=range(k,len(S)),overlap=.23)
    k=len(S)
    shade(S,[(13.34,10.43),(13.08,12.8),(12.45,16.72),(12.92,16.82),(13.70,12.61)],
          (12.4,10.4,13.8,16.9),825,angle=78,spacing=.057)
    shade(S,[(14.69,11),(15.23,12.03),(16.51,16.46),(15.87,16.81),(15.23,13.4)],
          (14.6,11,16.6,16.9),826,angle=72,spacing=.075)
    shade(S,[(12.44,7.69),(12.26,8.48),(12.64,9.33),(13.56,9.82),(13.13,8.86)],
          (12.2,7.5,13.6,10),827,angle=38,spacing=.052)
    # Broken secondary hatching follows cloth instead of filling an icon with
    # three flat patches. The right edge sits in the hood's and shoulder's shade.
    shade(S,[(14.28,10.4),(15.36,11.23),(16.58,16.48),(15.99,16.9),
             (15.43,15.05),(14.93,12.55)],(14.2,10.3,16.7,17),828,angle=83,spacing=.047)
    shade(S,[(12.14,13.1),(12.70,13.53),(12.26,15.97),(11.84,16.54)],
          (11.8,13.0,12.8,16.6),829,angle=77,spacing=.042)
    shade(S,[(12.27,8.13),(12.17,8.93),(12.8,9.59),(14.27,10.00),
             (13.79,9.63),(12.74,8.78)],(12.1,8.1,14.3,10.1),830,angle=29,spacing=.043)
    shade(S,[(13.45,7.04),(14.01,7.39),(14.34,7.79),(13.89,7.56)],
          (13.4,7.0,14.4,7.9),831,angle=42,spacing=.065)
    for i in range(17):
        q=i/16
        contour(S,[(14.54+.47*q,12.2+q*2.7),(14.68+.68*q,12.9+q*2.65)],.0075,850+i)
    for i in range(9):
        q=i/8
        contour(S,[(11.62+.48*q,11.4),(12.26+.5*q,12.45),(12.64+.43*q,12.66)],.007,880+i)
    # Short broken ground pulls seat the hem without adding a setting or symbol.
    for i in range(18):
        x=10.6+i*.40;y=17.43+.12*np.sin(i*1.7)
        contour(S,[(x,y),(x+.24+.11*np.sin(i),y-.025)],.007,900+i,0)
    S.retime(5.4,7.05,order=range(k,len(S)),overlap=.7)
    return S


class AbandonedDeep(PG.Deep):
    """Discard only the miners; consume their RNG so all later original strokes stay exact."""
    def _miner(self, S, x, yfoot, h, rng, lay):
        super()._miner(Strokes(), x, yfoot, h, rng, lay)

    def additions(self):
        S=Strokes()
        gx=float(self.vein[0,0]); gy=float(self.surface(gx))
        # Resting lantern outside the mouth; a handle, iron cage and closed base.
        lx,ly=gx-1.35,gy-.06
        lantern_start=len(S)
        for q,p in enumerate([
            [(lx-.33,ly),(lx-.31,ly-.62),(lx+.31,ly-.62),(lx+.33,ly),(lx-.33,ly)],
            [(lx-.41,ly-.63),(lx,ly-.91),(lx+.41,ly-.63)],
            [(lx-.18,ly-.86),(lx-.18,ly-1.17),(lx+.18,ly-1.17),(lx+.18,ly-.86)],
            [(lx-.31,ly-.1),(lx+.31,ly-.1)],
            [(lx-.14,ly-.58),(lx-.14,ly-.14)],[(lx+.14,ly-.58),(lx+.14,ly-.14)],
        ]): contour(S,p,.020,950+q,0)
        for i in range(lantern_start,len(S)):
            S.P[i]=np.array([lx,ly])+(S.P[i]-[lx,ly])*1.65
            S.R[i]*=2.05
        # Empty ladders descend alongside the retained stair system.
        for k,h in enumerate(self.halls):
            x=h['x0']+.38 if k%2 else h['x1']-.38
            ya=h['yc']+.48; yb=h['y1']-.04
            for dx in (-.12,.12):contour(S,[(x+dx,ya),(x+dx+.07,yb)],.029,970+k*3+(dx>0),0)
            for j,y in enumerate(np.arange(ya+.1,yb,.17)):
                dx=.07*(y-ya)/(yb-ya)
                contour(S,[(x-.12+dx,y),(x+.12+dx,y)],.022,995+k*40+j,0)
        return S
