"""Cut D's consolidated edit owns its timeline without changing the A/B/C alternatives.

The accepted hashes were captured before D was added (30 Sep 2026). Intentional edits to those alternative cuts
must revise their baselines explicitly; adding or replacing D's story must not.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

EDIT = Path(__file__).resolve().parents[1]
ROOT = EDIT.parent
sys.path.insert(0, str(EDIT))
sys.path.insert(0, str(ROOT / 'music' / 'src'))
import assemble as AS  # noqa: E402
import edl_v3 as EDL  # noqa: E402
import titles  # noqa: E402
from timeline_v3 import BarMap  # noqa: E402


ACCEPTED = {
    'A': ('2702427b27fc9dfb7c262702a62777867c570c8bd90026d805615e5734b31371',
          '22bacc8b776394081adfc297037ff9af9fecc5f78f46d99bc4a9cc4e7987be83',
          '421b483638b6b8b6fb708758e9ff23ab0113ec608ff200e77f6253bd6fb8fdf1',
          'cf6010510a0934beec55403b4b12587b2ab5457f107a3b3c2c20701f1467256c'),
    'B': ('32656271eec4c84884415287b81cdb563ee682d8b78ad697d892cd1e4b915c82',
          '5cdfd76d43ef17b5bcb37ce6d9d166a1610c056170907fb6a6629990f0a38a9c',
          '8bb00d658e77d921e6de85425d0dc63debd6f9782303bd674bcf779b1e9cbbcc',
          '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'),
    'C': ('e17b1d8920ba19a1bf94e5d84a762db9b38f79079800389f52dff33c9c1bdf48',
          'aad2d322c1852327b6a617287ae42f9b48ac4a67ba1c177938ccf24320226e66',
          'c3a236953ce32a021f31e9be73afbe0a1d5ef2d2a58df26017806fd2b8826d8d',
          '84d9b9b6babaaf136e83ef584cd66d86fd79593e9b59e7fbb7b992b4d8b5a823'),
}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def assert_accepted(data, expected):
    assert hashlib.sha256(data).hexdigest() == expected


@pytest.mark.parametrize('cut', 'ABC')
def test_alternative_edls_exports_captions_and_transitions_unchanged(cut):
    payloads = ((EDIT / 'edl' / f'edl_{cut}.json').read_bytes(), encoded(AS.edl_doc(cut)),
                encoded(titles.text_table(cut)), encoded(EDL.TRANS[cut]))
    for data, expected in zip(payloads, ACCEPTED[cut]):
        assert_accepted(data, expected)
        # Negative control for each guard, including the raw export-byte check.
        with pytest.raises(AssertionError):
            assert_accepted(data + b' ', expected)


def mutable_ids(value):
    if isinstance(value, dict):
        return {id(value)}.union(*(mutable_ids(v) for v in value.values()))
    if isinstance(value, list):
        return {id(value)}.union(*(mutable_ids(v) for v in value))
    if isinstance(value, tuple):
        return set().union(*(mutable_ids(v) for v in value))
    return set()


def assert_independent(left, right):
    assert not mutable_ids(left) & mutable_ids(right)


@pytest.mark.parametrize('original,consolidated', [(EDL.C, EDL.D), (EDL.TRANS['C'], EDL.TRANS['D']),
                                                   (titles.C5_TEXT, titles.D_TEXT)])
def test_d_is_distinct_and_deeply_independent(original, consolidated):
    assert consolidated != original
    assert_independent(original, consolidated)
    with pytest.raises(AssertionError):
        assert_independent(original, list(original))  # a shallow list copy leaks ownership through its rows


def test_d_has_its_own_barmap_sections_and_caption_table():
    d, c = BarMap('D'), BarMap('C5')
    assert d.cut == d.d['cut'] == 'D'
    assert d.frames == EDL.TOTAL['D']
    assert EDL.BARMAP['D'] != EDL.BARMAP['C']
    assert [s['id'] for s in d.sections] == list(dict.fromkeys(s['sec'] for s in EDL.D))
    assert_independent(d.sections, c.sections)
    assert titles.text_table('D') != titles.text_table('C')
    disk = json.loads((ROOT / 'music/v3/barmap_D.json').read_text())
    assert disk['status'] == 'treatment; new picture pending'
    assert disk['text'] == json.loads(encoded(titles.text_table('D')))
    assert EDL.check(ROOT / 'music/v3')


def assert_d_status(bm):
    assert bm.d['status'] == 'treatment; new picture pending + cues'


def test_d_status_survives_barmap_loading_and_c_keeps_historical_status():
    assert_d_status(BarMap('D'))
    c = BarMap('C5')
    assert c.d['status'] == 'locked (SHOWRUNNER-REV) + cues'
    with pytest.raises(AssertionError):
        assert_d_status(c)


def test_d_can_extend_to120_bars_without_extending_c(tmp_path, monkeypatch):
    # This synthetic extension tests the timeline contract; it is not a proposed D story edit.
    for name in EDL.BARMAP.values():
        (tmp_path / name).write_bytes((ROOT / 'music/v3' / name).read_bytes())
    path = tmp_path / EDL.BARMAP['D']
    bm = json.loads(path.read_text())
    bm.update(frames=9600, bars=120)
    bm['sections'][-1].update(f1=9600, bar_end=120, t1=400.0)
    path.write_text(json.dumps(bm))
    rows = deepcopy(EDL.D)
    rows[-1]['f1'] = 9600
    monkeypatch.setitem(EDL.EDL, 'D', rows)
    monkeypatch.setitem(EDL.TOTAL, 'D', 9600)
    assert EDL.check(tmp_path)
    assert EDL.d_export_extra()['bars'] == 120
    assert BarMap('D', path=path).frames == 9600
    assert EDL.TOTAL['C'] == EDL.C[-1]['f1'] == 5920
    # A hidden assumption that D still has C's length must fail against D's own map.
    monkeypatch.setitem(EDL.TOTAL, 'D', EDL.TOTAL['C'])
    with pytest.raises(AssertionError):
        EDL.check(tmp_path)


@pytest.mark.parametrize('kind', ['ink', 'fire'])
def test_d_uses_c_caption_style_and_backing(kind, monkeypatch):
    row = dict(id='D_STYLE_TEST', line='The words on the page.', f_in=0, f_out=140, set=kind,
               x=960, y=402, backing=0.3)
    # Match the procedural seed so this check isolates style routing from the intentional per-cut seed.
    seed = titles._seed
    monkeypatch.setattr(titles, '_seed', lambda *parts: seed('C', *parts[1:]))
    picture = np.full((201, 480, 3), 0.5, np.float32)
    c = titles.TextV3('C', row, 0.25).draw(picture.copy(), 40)
    d = titles.TextV3('D', row, 0.25).draw(picture.copy(), 40)
    np.testing.assert_array_equal(d, c)
    # Without the D route this falls into A/B's fade style, font settings and omitted backing.
    legacy = titles.TextV3('A', row, 0.25).draw(picture.copy(), 40)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(legacy, c)
