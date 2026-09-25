"""Rebuild renders/<dept>/f_%05d.png (global numbering) from handoff/clips/*.mp4."""
import os, re, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for f in sorted(os.listdir(os.path.join(ROOT, 'handoff', 'clips'))):
    m = re.match(r'(\w+)_(\d{5})-(\d{5})\.mp4$', f)
    if not m: continue
    dept, a = m.group(1), int(m.group(2))
    d = os.path.join(ROOT, 'renders', dept); os.makedirs(d, exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', os.path.join(ROOT, 'handoff', 'clips', f),
                    '-start_number', str(a), os.path.join(d, 'f_%05d.png')], check=True)
    print('restored', f)
