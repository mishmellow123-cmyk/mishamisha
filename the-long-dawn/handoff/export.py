"""Export rendered frames as high-quality clips (global frame ranges in filenames) so a fresh
container can rebuild renders/<dept>/f_%05d.png with handoff/import.py."""
import os, re, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for dept in ['hills', 'embers', 'montage', 'globe', 'accord']:
    d = os.path.join(ROOT, 'renders', dept)
    fs = sorted(int(m.group(1)) for f in os.listdir(d) if (m := re.match(r'f_(\d{5})\.png$', f)))
    runs, s = [], None
    for i, f in enumerate(fs):
        if s is None: s = f
        if i == len(fs) - 1 or fs[i + 1] != f + 1: runs.append((s, f)); s = None
    for a, b in runs:
        for c0 in range(a, b + 1, 300):           # chunks keep files well under 100 MB
            c1 = min(b, c0 + 299)
            out = os.path.join(ROOT, 'handoff', 'clips', f'{dept}_{c0:05d}-{c1:05d}.mp4')
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '24', '-start_number', str(c0),
                            '-i', os.path.join(d, 'f_%05d.png'), '-frames:v', str(c1 - c0 + 1),
                            '-c:v', 'libx264', '-preset', 'medium', '-crf', '15', '-pix_fmt', 'yuv420p', out], check=True)
            print(out, os.path.getsize(out) // 1e6, 'MB', flush=True)
