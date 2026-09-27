"""RUN-C's flicker (boil) gate as ONE farm unit: AOV caches -> the ink pass on consecutive frames (anchored, plus the
re-dealt boiling control) -> ink_check -> a report card written as a frame (renders/run_c_tests/gate_report/f_00000.png,
1920x804), because the farm streams back frames, not logs. Run from shots/run/:

    NUMBA_NUM_THREADS=8 python3 runc_gate.py            # scroll 232-239 and reveal 108-115 at half scale
PASS = anchored/re-dealt well under 0.5 on both shots (pre-H5 0.25; after H5 0.38-0.40)."""
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'renders', 'run_c_tests'))      # = render_ink.OUT
SETS = (('scroll', 232, 239), ('reveal', 108, 115))


def run(args):
    r = subprocess.run([sys.executable] + args, cwd=HERE, capture_output=True, text=True)
    txt = (r.stdout or '') + (r.stderr or '')
    if r.returncode != 0:
        print(txt[-3000:], flush=True)
        raise SystemExit(f'failed: {" ".join(args)}')
    return txt


def main():
    th = os.environ.get('NUMBA_NUM_THREADS', '8')
    lines = [f'RUN-C FLICKER GATE  {time.strftime("%Y-%m-%d %H:%MZ", time.gmtime())}  (half scale, land only)']
    for shot, s, e in SETS:
        t0 = time.time()
        run(['render_ink.py', '--shot', shot, '--range', f'{s}-{e}', '--scale', '0.5', '--threads', th, '--skip'])
        run(['inkpass.py', '--shot', shot, '--range', f'{s}-{e}', '--scale', '0.5', '--tag', '_g'])
        run(['inkpass.py', '--shot', shot, '--range', f'{s}-{e - 1}', '--scale', '0.5', '--salt', '1000', '--tag', '_reseed'])
        txt = run(['ink_check.py', '--shot', shot, '--range', f'{s}-{e}', '--scale', '0.5', '--tag', '_g', '--surf', 'land'])
        with open(os.path.join(OUT, f'boil_check_{shot}_h5.txt'), 'w') as fh:
            fh.write(txt)
        print(txt, flush=True)
        mean = [ln for ln in txt.splitlines() if ln.startswith('MEAN')]
        lines.append(f'{shot} {s}-{e}: ' + (mean[0] if mean else 'NO RESULT') + f'   ({time.time() - t0:.0f}s)')
        lines += ['   ' + ln for ln in txt.splitlines() if '->' in ln]
    img = np.full((804, 1920, 3), 245, np.uint8)
    for i, ln in enumerate(lines):
        cv2.putText(img, ln[:150], (24, 48 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    d = os.path.join(OUT, 'gate_report')
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, 'tmp_report.png')
    cv2.imwrite(tmp, img)
    os.replace(tmp, os.path.join(d, 'f_00000.png'))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
