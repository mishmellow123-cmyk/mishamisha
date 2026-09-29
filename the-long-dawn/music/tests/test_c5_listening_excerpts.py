"""listening_excerpts_c5.py: its windows come from the measured bar map, missing picture becomes a slate, and the
excerpt's audio starts on its first frame to the sample (checked by decoding the written .mov)."""
import os
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

import audio_guard_v3 as AG
import listening_excerpts_c5 as L

FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
needs_ffmpeg = pytest.mark.skipif(not (os.path.exists(L.FFMPEG) and os.path.exists(FFPROBE)), reason="no ffmpeg")


def test_windows_sit_in_the_film_and_quote_the_measured_frames():
    ws = L.windows()
    assert [w[0] for w in ws] == ["refusal", "trap", "first_fires", "map", "silence_ring"]
    for wid, f0, f1, qs in ws:
        assert 0 <= f0 < f1 < L.FRAMES and qs, wid
    text = " ".join(q for w in ws for q in w[3])
    for f in ("2546-2588", "2880", "3523", "3584", "3642", "3791", "3786", "4181"):
        assert f in text, f
    moved = dict(L.barmap_frames(), all_lit=3795)
    assert "3795" in " ".join(L.windows(moved)[3][3])                       # the questions follow the bar map


def test_missing_picture_is_named():
    assert "FLINT" in L.missing_label(2700) and "BEACON RUN" in L.missing_label(3200)
    assert L.frame_source("/nonexistent", 3850) is None


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """a 20-frame film: a delivered shot 'fake' on frames 0-11 (frame 7 lost), a beep on frame 5 of the audio"""
    from PIL import Image
    monkeypatch.setattr(L.M, "SHOTS", {"fake": ("fake", 0, 11, "CX")})
    (tmp_path / "frames" / "fake").mkdir(parents=True)
    for f in range(12):
        if f != 7:
            Image.new("RGB", (1920, 804), (20 * f, 40, 90)).save(tmp_path / "frames" / "fake" / f"f_{f:05d}.jpg")
    y = np.zeros((20 * L.FR, 2), np.float32)
    y[5 * L.FR:5 * L.FR + 480] = 0.25                  # a block, so its very first sample is non-zero
    wav = tmp_path / "final_X.wav"
    sf.write(wav, y, L.SR, subtype="PCM_24")
    out = tmp_path / "out"
    out.mkdir()
    return tmp_path, str(wav), str(out)


@needs_ffmpeg
def test_excerpt_is_in_sync_to_the_sample(fake):
    tmp, wav, out = fake
    mov, slates = L.excerpt(str(tmp / "frames"), wav, "t", 0, 11, ["a question"], out, frames_total=20)
    assert slates == [7]
    n = subprocess.run([FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", mov], capture_output=True, text=True).stdout
    assert int(n) == 12
    raw = subprocess.run([L.FFMPEG, "-v", "error", "-i", mov, "-map", "0:a", "-f", "f32le", "-ac", "1", "-"],
                         capture_output=True).stdout
    a = np.frombuffer(raw, np.float32)
    assert len(a) == 12 * L.FR
    assert int(np.flatnonzero(np.abs(a) > 1e-3)[0]) == 5 * L.FR                 # the beep, on frame 5, to the sample
    txt = open(mov[:-4] + ".txt").read()
    assert "SLATES" in txt and "a question" in txt and "UNHEARD" in txt


def test_excerpt_refuses_audio_of_another_length(fake):
    tmp, wav, out = fake
    with pytest.raises(AG.StaleArtefact):
        L.excerpt(str(tmp / "frames"), wav, "t", 0, 11, [], out, frames_total=21)
