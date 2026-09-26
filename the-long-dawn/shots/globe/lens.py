"""DAWN v2 lens: the sun as a clean glare (no diffraction starburst) and a restrained anamorphic
streak. HDR, additive, sized for a 1920-wide frame and scaled to W."""
import numpy as np


def sun_glare_clean(W, H, sx, sy, core, flash=0.0, streak=0.0, tint=(1.0, 0.88, 0.70),
                    streak_tint=(1.0, 0.86, 0.68), streak_len=240.0):
    """core: brightness of the disc glare (0 while the disc is hidden); flash: extra round bloom
    for the moment the sun breaks the limb; streak: peak of a thin horizontal anamorphic line."""
    img = np.zeros((H, W, 3), np.float32)
    k = W / 1920.0
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = (x + 0.5 - sx) / k
    dy = (y + 0.5 - sy) / k
    r = np.sqrt(dx * dx + dy * dy) + 1e-3
    tint = np.asarray(tint, np.float32)
    if core > 0:
        g_hot = core * (1.4 * np.exp(-r / 6.0) + 0.55 * np.exp(-r / 22.0))
        g_warm = core * 0.12 / (1.0 + (r / 60.0) ** 2)
        g_far = core * 0.018 / (1.0 + (r / 210.0) ** 2) ** 1.5
        img += (g_hot[..., None] * np.asarray((1.0, 0.96, 0.9), np.float32) + g_warm[..., None] * tint
                + g_far[..., None] * np.asarray((0.9, 0.9, 0.95), np.float32))
    if flash > 0:
        g = flash * (2.4 * np.exp(-r / 26.0) + 0.5 / (1.0 + (r / 90.0) ** 2) + 0.03 / (1.0 + (r / 300.0) ** 2))
        img += g[..., None] * np.asarray((1.0, 0.93, 0.82), np.float32)
    if streak > 0:
        ax = np.abs(dx)
        prof = 0.55 * np.exp(-ax / streak_len) + 0.45 * np.exp(-ax / (0.3 * streak_len))
        thin = np.exp(-0.5 * (dy / 1.25) ** 2) + 0.16 * np.exp(-0.5 * (dy / 5.0) ** 2)
        # fade in over the core so the line never shows as a hard bar through the disc
        img += (streak * prof * thin)[..., None] * np.asarray(streak_tint, np.float32)
    return img
