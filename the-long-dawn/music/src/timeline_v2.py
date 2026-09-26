"""THE LONG DAWN v2 - the time grid (BIBLE_V2.md section 3).

24 fps, 72 BPM, 1 beat = 20 frames = 40 000 samples, 1 bar = 80 frames.
The v2 film is 2968 frames = 123.667 s = 5 936 000 samples @ 48 kHz.
Frames 0-1199 (bars 1-15) are the v1 first half, unchanged.  THE BEACON RUN
(1520-1679) is new; everything after it sits 160 frames later than in v1.
"""
SR = 48000
FPS = 24
FR = SR // FPS                 # 2000 samples per frame
BEAT_N = 40000
TOTAL_FRAMES = 2968
TOTAL_N = TOTAL_FRAMES * FR    # 5 936 000
TOTAL_S = TOTAL_N / SR         # 123.667
RENDER_N = TOTAL_N + SR        # 1 s of tail margin while rendering (trimmed by the master)
SHIFT = 160                    # v1 -> v2 frame offset after the Beacon Run


def fb(frame):
    """global beat of a (v2) frame"""
    return frame / 20.0


def gb(bar, beat=1.0):
    return (bar - 1) * 4 + (beat - 1)


def bar_of_frame(frame):
    return int(frame // 80) + 1


# picture sections (v2 frames) for the analysis reports
SECTIONS = [
    ("intro", 0, 320), ("kindling", 320, 640), ("race", 640, 960), ("grasp", 960, 1040),
    ("silence", 1040, 1200), ("first_beacon", 1200, 1440), ("far_peak", 1440, 1520),
    ("beacon_run", 1520, 1680), ("montage", 1680, 1920), ("world_answers", 1920, 2080),
    ("accord", 2080, 2400), ("dawn", 2400, 2624), ("coda", 2624, 2968),
]

# sync points shared by every cut (frame, label)
SYNC_COMMON = [
    (0, "wind + drone from silence"), (320, "KINDLING shimmer"), (480, "IGNITION"),
    (640, "RACE drums enter (after the breath)"), (800, "storm swell"),
    (1040, "IMPACT (after the suck)"),
    (1236, "flint strike 1"), (1262, "flint strike 2"), (1290, "flint strike 3"),
    (1318, "kindling catches"), (1360, "the beacon ROARS"), (1480, "the shepherd's beacon"),
    (1520, "cut to THE BEACON RUN"),
    (1540, "run beacon 1"), (1560, "run beacon 2"), (1580, "run beacon 3"), (1600, "run beacon 4"),
    (1620, "run beacon 5"), (1640, "run beacon 6"), (1660, "run beacon 7"),
    (1700, "desert ignition"), (1760, "ice ignition"), (1810, "karst ignition"),
    (1850, "city ignition"), (1890, "sea ignition"), (1920, "the world answers"),
    (2160, "flames merge"), (2320, "carved ring sweep begins"), (2385, "hearth flare"),
    (2400, "CLIMAX - the sun breaks"), (2624, "dissolve to the hill"),
    (2740, "torch handover"), (2800, "the child's beacon catches"),
    (2880, "final chord"),
]

RUN_BEACONS = [1540, 1560, 1580, 1600, 1620, 1640, 1660]
MONTAGE_IGN = [1700, 1760, 1810, 1850, 1890]
OATHS = [2160, 2200, 2240, 2280]          # cut A/C carvings - the music must NOT hit these one by one
