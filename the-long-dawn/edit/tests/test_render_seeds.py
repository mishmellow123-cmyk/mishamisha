"""Render noise must not depend on the process. Python salts str hashes per process (PYTHONHASHSEED), and
assemble.py draws captions in spawned Pool workers that each rebuild them, so a seed taken from hash() gave every
worker its own flicker, crumble order and sparks for the same caption (found 29 Sep, 32 of 32 caption fields).
The same seed made book_C's baked ink lines (inkline.py) differ from one render process to the next.

Run: python -B -m unittest discover -s the-long-dawn/edit/tests -p 'test_*.py'
"""
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tokenize
import unittest

ROOT = Path(__file__).resolve().parents[2]

CHILD = r'''
import json, sys, warnings, zlib
warnings.simplefilter('ignore')
sys.path.insert(0, sys.argv[1] + '/edit'); sys.path.insert(0, sys.argv[1] + '/shots/map')
import titles, inkline
out = {}
for ln in titles.lines_v3('C', 0.25):
    out[ln.id] = zlib.crc32(ln.noise.tobytes())
    if ln.kind == 'fire':
        a, heat, u, te = ln._fire(ln.f_out - 6)
        sp = ln._sparks_at(u, te)[0]
        out[ln.id + ' sparks'] = zlib.crc32(sp['x'].tobytes() + sp['vx'].tobytes() + sp['life'].tobytes())
for key in inkline.LINES:
    out['ink ' + key] = zlib.crc32(inkline.InkLine(key, 20).noise.tobytes())
print(json.dumps(out))
'''


def fields(hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    r = subprocess.run([sys.executable, '-c', CHILD, str(ROOT)], env=env, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise AssertionError(r.stderr[-2000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


class RenderSeedTests(unittest.TestCase):
    def test_caption_and_ink_noise_are_the_same_in_every_process(self):
        a, b = fields(1), fields(2)
        self.assertEqual(set(a), set(b))
        self.assertGreaterEqual(len(a), 17 + 9 + 2)            # C5: 17 captions, 9 of them fire; ink T1 and T14
        self.assertEqual([k for k in a if a[k] != b[k]], [])

    def test_no_render_seed_is_taken_from_hash(self):
        """The class, by pattern: a call of builtin hash() on a line whose code seeds or draws noise, anywhere in
        edit/ or shots/. Read as tokens, so comments and docstrings that mention hash() are not calls."""
        bad, scanned = [], 0
        for base in ('edit', 'shots'):
            for p in (ROOT / base).rglob('*.py'):
                if 'tests' in p.parts:
                    continue
                scanned += 1
                try:
                    toks = list(tokenize.generate_tokens(io.StringIO(p.read_text(errors='replace')).readline))
                except (tokenize.TokenError, SyntaxError):
                    continue
                names = {}
                for t in toks:
                    if t.type == tokenize.NAME:
                        names.setdefault(t.start[0], []).append(t.string)
                code = [t for t in toks if t.type in (tokenize.NAME, tokenize.OP)]
                for prev, t, nxt in zip([None] + code, code, code[1:] + [None]):
                    if (t.string == 'hash' and nxt is not None and nxt.string == '('
                            and (prev is None or prev.string != '.')
                            and re.search(r'rng|seed|noise|random', ' '.join(names.get(t.start[0], ())))):
                        bad.append(f'{p.relative_to(ROOT)}:{t.start[0]}')
        self.assertGreater(scanned, 50)                         # the scan saw the tree
        self.assertEqual(bad, [], 'seed with zlib.crc32 (titles._seed), not hash(): hash() is salted per process')


if __name__ == '__main__':
    unittest.main()
