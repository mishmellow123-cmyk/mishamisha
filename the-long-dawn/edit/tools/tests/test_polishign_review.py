"""Negative controls for the bounded review pipeline; no source frames are loaded."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

SPEC = importlib.util.spec_from_file_location('polishign_review', Path(__file__).parents[1] / 'polishign_review.py')
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def test_thirteenth_frame_is_refused_before_worker_launch(monkeypatch):
    calls = []
    monkeypatch.setattr(R, 'run_worker', lambda args: calls.append(args))
    with pytest.raises(SystemExit, match='At most 12'):
        R.main(['batch', '--version', 'before', '--frames', ','.join(map(str, range(13)))])
    assert calls == []
    R.main(['batch', '--version', 'before', '--frames', ','.join(map(str, range(12)))])
    assert len(calls) == 1


def test_historical_modules_keep_worktree_resource_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'ROOT', tmp_path)
    source = {
        'afix_comp': b'VALUE = "historical"',
        'edl_v3': b'import afix_comp\nVALUE = afix_comp.VALUE',
        'assemble': b'import edl_v3\nVALUE = edl_v3.VALUE',
    }
    monkeypatch.setattr(R, 'baseline_source', lambda name, ref: source[name])
    for name in R.BASE_MODULES:
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(VALUE='current'))
    module = R.load_assembler('before', 'test-ref')
    assert module.VALUE == 'historical'
    assert module.__file__ == str(tmp_path / 'edit/assemble.py')
    assert sys.modules['edl_v3'].__file__ == str(tmp_path / 'edit/edl_v3.py')


def test_missing_sources_and_slates_fail_instead_of_creating_evidence():
    ctx = types.SimpleNamespace(read=lambda ref: None, take_frame=lambda take, f: None,
                                slate=lambda i, status: 'slate pixels', shots=[{'sec': 'A4'}])
    assembler = types.SimpleNamespace(_CTX=ctx, _init=lambda *args: None)
    R.checked_context(assembler)
    with pytest.raises(OSError, match='Cannot decode'):
        ctx.read('missing.png')
    with pytest.raises(OSError, match='Missing source'):
        ctx.take_frame({'stem': 'missing'}, 1020)
    with pytest.raises(OSError, match='Refusing SLATE'):
        ctx.slate(0, 'gap')


def test_stale_png_and_changed_source_cannot_be_reused(tmp_path):
    png, metadata = R.paths(tmp_path, 'before', 1020)
    png.parent.mkdir()
    png.write_bytes(b'png evidence')
    identity = {'source_sha256': 'original-source'}
    # A stale PNG without a validation marker is never evidence.
    assert not R.reusable(tmp_path, 'before', 1020, identity)
    metadata.write_text(json.dumps(dict(identity, png_sha256=hashlib.sha256(png.read_bytes()).hexdigest())))
    assert R.reusable(tmp_path, 'before', 1020, identity)
    assert not R.reusable(tmp_path, 'before', 1020, {'source_sha256': 'changed-source'})
    png.write_bytes(b'replaced evidence')
    assert not R.reusable(tmp_path, 'before', 1020, identity)


def test_controls_compare_all_pixels_at_output_resolution(tmp_path):
    import numpy as np
    from PIL import Image

    for version in ('before', 'after'):
        png, metadata = R.paths(tmp_path, version, 959)
        png.parent.mkdir()
        rgb = np.zeros((268, 640, 3), np.uint8)
        if version == 'after':
            rgb[0, 0, 0] = 1  # vanishes when downsampled to gray; must still fail equality
        Image.fromarray(rgb).save(png)
        metadata.write_text(json.dumps({'source_sha256': version,
                                        'png_sha256': hashlib.sha256(png.read_bytes()).hexdigest()}))
    args = types.SimpleNamespace(out=tmp_path, frames=[959], controls=[959], name='test')
    with pytest.raises(SystemExit, match='Negative-control frames changed'):
        R.measure(args)
    record = json.loads((tmp_path / 'test_measurements.json').read_text())['frames'][0]
    assert record['changed_pixels'] == 1
    assert record['max_channel_delta'] == 1
    assert record['before']['step_mad_gray_320x134'] is None


def test_master_with_a_different_frame_grid_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(R.subprocess, 'check_output', lambda *args, **kwargs:
                        json.dumps({'streams': [{'r_frame_rate': '30/1'}]}))
    args = types.SimpleNamespace(master=tmp_path / 'master.mov', out=tmp_path, frames=[1000])
    with pytest.raises(ValueError, match='24 fps grid'):
        R.master(args)
