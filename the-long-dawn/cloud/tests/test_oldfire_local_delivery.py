"""Exercise the actual farm JPEG encoder and local assembly's refusal paths."""
from pathlib import Path
import sys
from unittest import mock

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import farm_node
import render_oldfire_local as delivery
import assemble_oldfire_local as assembly

SMALL = (12, 20)


def encoded(tmp_path, frame=5840, value=97):
    folders = {key: tmp_path/key for key in ('rgb', 'matte')}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
        pixels = np.full((*SMALL, 3), value, np.uint8)
        pixels[:, ::3] = (40, 110, 220)
        assert cv2.imwrite(str(folder/f'f_{frame:05d}.png'), pixels)
    return delivery.encode_pair(folders, tmp_path/'ready', frame, SMALL)


def test_actual_farm_scan_encodes_both_outputs_with_required_tables(tmp_path):
    with mock.patch.object(farm_node, 'scan', wraps=farm_node.scan) as scan:
        result = encoded(tmp_path)
    scan.assert_called_once()
    assert scan.call_args.kwargs == {'final': True}
    for path, metadata in result.values():
        assert path.suffix == '.jpg'
        assert metadata['decode_checked'] and metadata['sampling'] == '4:4:4'
        assert metadata['quantisation'] == delivery.Q95_TABLES
        assert metadata['sha256'] == delivery.sha256(path)


@pytest.mark.parametrize('options,reason', [
    ([cv2.IMWRITE_JPEG_QUALITY, 94, cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
      cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444], 'q95'),
    ([cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
      cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420], '4:4:4')])
def test_wrong_encoding_is_rejected(tmp_path, options, reason):
    path = tmp_path/'wrong.jpg'
    assert cv2.imwrite(str(path), np.zeros((*SMALL, 3), np.uint8), options)
    with pytest.raises(ValueError, match=reason):
        delivery.validate_jpeg(path, SMALL)


def test_decode_and_dimensions_are_checked(tmp_path):
    path = encoded(tmp_path)['rgb'][0]
    with pytest.raises(ValueError, match='dimensions'):
        delivery.validate_jpeg(path, (11, 20))
    path.write_bytes(path.read_bytes()[:40])
    with pytest.raises(Exception):
        delivery.validate_jpeg(path, SMALL)


def test_comparison_reports_changes_without_tolerance_suppression(tmp_path):
    first = encoded(tmp_path/'a')['rgb'][0]
    same = delivery.compare_jpegs(first, first, SMALL)
    assert same['maximum_channel_difference'] == 0 and same['byte_equal']
    second = encoded(tmp_path/'b', value=98)['rgb'][0]
    changed = delivery.compare_jpegs(second, first, SMALL)
    assert changed['maximum_channel_difference'] > 0 and not changed['byte_equal']
    assert changed['mean_absolute_channel_difference'] > 0


def test_renders_is_read_only_even_through_alias(tmp_path, monkeypatch):
    root, actual = tmp_path/'project', tmp_path/'accepted'
    root.mkdir(); actual.mkdir()
    (root/'renders').symlink_to(actual, target_is_directory=True)
    alias = tmp_path/'alias'
    alias.symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr(delivery, 'ROOT', root)
    for destination in (actual/'x', alias/'x', root/'renders'/'x'):
        with pytest.raises(ValueError, match='read-only'):
            delivery.safe_output(destination)


def sources(tmp_path, monkeypatch):
    monkeypatch.setattr(assembly, 'END', 5843)
    monkeypatch.setattr(assembly, 'validate_jpeg',
                        lambda path: delivery.validate_jpeg(path, SMALL))
    farm, patch = tmp_path/'farm', tmp_path/'patch'
    for base in (farm, patch):
        base.mkdir(); base.with_name(base.name+'_matte').mkdir()
    for frame, base in ((5840, farm), (5841, patch), (5842, patch)):
        pair = encoded(tmp_path/f'stage-{frame}', frame, 97+frame%3)
        for key, (path, _) in pair.items():
            folder = base if key == 'rgb' else base.with_name(base.name+'_matte')
            (folder/path.name).write_bytes(path.read_bytes())
    return farm, patch


def test_assembly_copies_exact_source_bytes_and_provenance(tmp_path, monkeypatch):
    farm, patch = sources(tmp_path, monkeypatch)
    result = assembly.assemble(farm, patch, 5841, tmp_path/'owner')
    assert result['frames'] == 3 and len(result['files']) == 6
    assert [r['origin'] for r in result['files']] == ['farm']*2+['local']*4
    for row in result['files']:
        assert Path(row['source']).read_bytes() == Path(row['target']).read_bytes()
    assert (tmp_path/'owner_assembly.json').is_file()


def test_missing_patch_never_falls_back_to_farm_and_leaves_no_output(tmp_path, monkeypatch):
    farm, patch = sources(tmp_path, monkeypatch)
    missing = patch/'f_05842.jpg'
    (farm/missing.name).write_bytes(missing.read_bytes())
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        assembly.assemble(farm, patch, 5841, tmp_path/'owner')
    assert not (tmp_path/'owner').exists()


def test_existing_destination_is_refused_before_writes(tmp_path, monkeypatch):
    farm, patch = sources(tmp_path, monkeypatch)
    output = tmp_path/'owner'
    output.mkdir(); (output/'retain.txt').write_text('retained')
    with pytest.raises(ValueError, match='fresh'):
        assembly.assemble(farm, patch, 5841, output)
    assert (output/'retain.txt').read_text() == 'retained'
    assert not (tmp_path/'owner_matte').exists()


@pytest.mark.parametrize('bad', ['5839', '6080', '5841-5840', '5840-5841-5842'])
def test_local_cli_rejects_bad_frame_ranges(bad):
    with pytest.raises(ValueError):
        delivery.frames_of(bad)


def test_receipt_symlink_into_renders_is_rejected_before_read_or_write(tmp_path, monkeypatch):
    root = tmp_path/'project'
    accepted = root/'renders'
    accepted.mkdir(parents=True)
    protected = accepted/'delivery.jsonl'
    protected.write_text('protected: deliberately invalid receipt JSON')
    out = tmp_path/'owner'
    out.with_name(out.name+'_delivery.jsonl').symlink_to(protected)
    monkeypatch.setattr(delivery, 'ROOT', root)
    monkeypatch.setattr(delivery, 'build_renderer', lambda *_: pytest.fail('must refuse before render'))
    with pytest.raises(SystemExit) as error:
        delivery.main(['--out', str(out), '--frames', '5840', '--caption-hold'])
    assert error.value.code == 2
    assert protected.read_text() == 'protected: deliberately invalid receipt JSON'
    assert not out.exists() and not out.with_name(out.name+'_matte').exists()


def test_local_cli_resumes_only_matching_verified_receipts(tmp_path, monkeypatch, capsys):
    import json
    pair = encoded(tmp_path/'stage')
    out = tmp_path/'owner'
    outputs = {}
    for key, (path, metadata) in pair.items():
        folder = out if key == 'rgb' else out.with_name(out.name+'_matte')
        folder.mkdir(); (folder/path.name).write_bytes(path.read_bytes())
        outputs[key] = metadata
    out.with_name(out.name+'_delivery.jsonl').write_text(json.dumps(dict(
        frame=5840, caption_hold=True, outputs=outputs))+'\n')
    original = delivery.validate_jpeg
    monkeypatch.setattr(delivery, 'validate_jpeg', lambda path: original(path, SMALL))
    monkeypatch.setattr(delivery, 'build_renderer', lambda *_: pytest.fail('must skip render'))
    delivery.main(['--out', str(out), '--frames', '5840', '--caption-hold', '--resume'])
    assert json.loads(capsys.readouterr().out)['verified']
    (out/'f_05840.jpg').write_bytes(pair['matte'][0].read_bytes()+b'changed')
    with pytest.raises(SystemExit):
        delivery.main(['--out', str(out), '--frames', '5840', '--caption-hold', '--resume'])
