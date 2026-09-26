"""DAWN lens: the sun as a clean glare (no diffraction starburst) and a restrained anamorphic
streak. HDR, additive, sized for a 1920-wide frame and scaled to W.

v3: everything but the disc's own white-hot core is gold (the look's dawn_gold / accord_gold),
so the flash as the sun breaks the limb lands gold, not as a grey veil over black space."""
import numpy as np

HOT = (1.0, 0.95, 0.86)          # the disc: white-gold
GOLD = (1.0, 0.74, 0.42)         # the halo round it (dawn_gold, a touch deeper)
AMBER = (1.0, 0.60, 0.27)        # the wide veil: deeper still, so it reads gold, never grey


def sun_glare_clean(W, H, sx, sy, core, flash=0.0, streak=0.0, streak_len=240.0):
    """core: brightness of the disc glare (0 while the disc is hidden); flash: extra round bloom
    for the moment the sun breaks the limb; streak: peak of a thin horizontal anamorphic line."""
    img = np.zeros((H, W, 3), np.float32)
    k = W / 1920.0
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = (x + 0.5 - sx) / k
    dy = (y + 0.5 - sy) / k
    r = np.sqrt(dx * dx + dy * dy) + 1e-3
    hot, gold, amber = (np.asarray(c, np.float32) for c in (HOT, GOLD, AMBER))
    if core > 0:
        g_hot = core * (1.4 * np.exp(-r / 6.0) + 0.55 * np.exp(-r / 22.0))
        g_warm = core * 0.12 / (1.0 + (r / 60.0) ** 2)
        g_far = core * 0.016 / (1.0 + (r / 210.0) ** 2) ** 1.5
        img += g_hot[..., None] * hot + g_warm[..., None] * gold + g_far[..., None] * amber
    if flash > 0:
        # the burst: a white-gold heart, a gold bloom, and a wide amber veil that falls away
        # toward the frame's edges (so space round the sun goes gold, not milky)
        g1 = flash * 2.4 * np.exp(-r / 24.0)
        g2 = flash * 0.6 / (1.0 + (r / 110.0) ** 2) ** 2
        g3 = flash * 0.012 / (1.0 + (r / 300.0) ** 2) ** 2
        img += g1[..., None] * hot + g2[..., None] * (0.5 * gold + 0.5 * amber) + g3[..., None] * amber
    if streak > 0:
        ax = np.abs(dx)
        prof = 0.55 * np.exp(-ax / streak_len) + 0.45 * np.exp(-ax / (0.3 * streak_len))
        thin = np.exp(-0.5 * (dy / 1.25) ** 2) + 0.16 * np.exp(-0.5 * (dy / 5.0) ** 2)
        img += (streak * prof * thin)[..., None] * gold
    return img
