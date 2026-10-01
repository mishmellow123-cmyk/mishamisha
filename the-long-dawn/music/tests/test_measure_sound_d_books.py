from copy import deepcopy

import numpy as np
import pytest

import measure_sound_d_books as M


def rows():
    result=[dict(frame=f,letters=dict(red=0,light=0),metal_control=dict(red=0,light=0))
            for f in range(5880,5925)]
    result[17]['letters']['red']=346
    for row in result[18:]: row['letters']['light']=800
    return result


def test_pixel_predicate_distinguishes_orange_entry_from_bright_glow():
    image=np.array([[[230,100,40],[250,190,120],[150,100,60]]],np.uint8)
    assert M.counts(image)==dict(red=1,light=1)
    with pytest.raises(ValueError): M.counts(np.full((1,1,3),np.nan))


def test_complete_dark_prefix_onset_and_threshold_controls():
    evidence=rows()
    assert all(M.detect_flare(evidence,red_floor=n)==5897 for n in (50,100,200))
    absent=deepcopy(evidence);absent[17]['letters']['red']=0
    assert M.detect_flare(absent) is None


def test_round5_late_melt_control_failure_is_not_hidden_by_relaxed_predicates():
    evidence=rows()
    for row in evidence:
        if row['frame'] >= M.ROUND5_FLARE_STOP:
            row['metal_control']['red']=500
    with pytest.raises(ValueError,match='adjacent non-letter metal'):
        M.detect_flare(evidence)
    settled=[row for row in evidence if row['frame'] < M.ROUND5_FLARE_STOP]
    assert M.detect_flare(settled,stop=M.ROUND5_FLARE_STOP)==5897
    settled[17]['metal_control']['red']=500
    with pytest.raises(ValueError,match='adjacent non-letter metal'):
        M.detect_flare(settled,stop=M.ROUND5_FLARE_STOP)


@pytest.mark.parametrize('change',['missing','duplicate','order','prior_glow','metal','unconfirmed','early'])
def test_negative_picture_evidence_fails_closed(change):
    evidence=rows()
    if change=='missing': evidence.pop(4)
    elif change=='duplicate': evidence[4]=deepcopy(evidence[3])
    elif change=='order': evidence[1],evidence[2]=evidence[2],evidence[1]
    elif change=='prior_glow': evidence[0]['letters']['light']=100
    elif change=='metal': evidence[17]['metal_control']['red']=50
    elif change=='unconfirmed': evidence[18]['letters']['light']=0
    elif change=='early': evidence[1]['letters']['red']=300
    with pytest.raises(ValueError): M.detect_flare(evidence)


def test_caption_onset_requires_complete_clean_prefix_and_sustained_ink():
    trace=[dict(frame=f,dark_pixels=0 if f<6017 else 94) for f in range(6011,6020)]
    assert M.caption_onset(trace)==6017
    for mode in ('missing','border','blink','absent','early'):
        bad=deepcopy(trace)
        if mode=='missing': bad.pop(2)
        if mode=='border': bad[0]['dark_pixels']=489
        if mode=='blink': bad[-1]['dark_pixels']=0
        if mode=='absent':
            for row in bad: row['dark_pixels']=0
        if mode=='early': bad[1]['dark_pixels']=40
        with pytest.raises(ValueError): M.caption_onset(bad)


def test_round5_dark_predicate_uses_all_three_rgb_channels():
    pixels=np.array([[[100,70,50],[121,70,50],[100,86,50],[100,70,61]]],np.uint8)
    assert M.dark_count(pixels,(120,85,60)) == 1
    with pytest.raises(ValueError): M.dark_count(np.full((1,1,3),np.nan),(120,85,60))


def test_completion_needs_final_mark_and_entire_later_hold():
    rows=[dict(frame=f,period=0 if f<5983 else 20) for f in range(5963,6080)]
    assert M.persistent_entry(rows,5963,6080,'period',floor=5)==5983
    for mode in ('missing','duplicate','earlier_mark','late_absence','none','too_late'):
        bad=deepcopy(rows)
        if mode=='missing': bad.pop(40)
        elif mode=='duplicate': bad[10]=deepcopy(bad[9])
        elif mode=='earlier_mark': bad[0]['period']=10
        elif mode=='late_absence': bad[-1]['period']=0
        elif mode=='none':
            for r in bad:r['period']=0
        elif mode=='too_late':
            for r in bad:r['period']=20 if r['frame']==6079 else 0
        with pytest.raises(ValueError): M.persistent_entry(bad,5963,6080,'period',floor=5)


def test_native_frame_rejects_wrong_dimensions_and_read_time_replacement(tmp_path,monkeypatch):
    from PIL import Image
    folder=tmp_path/'renders/synthetic_test_fixture';folder.mkdir(parents=True)
    path=folder/'f_05960.jpg'
    Image.new('RGB',(100,100)).save(path)
    with pytest.raises(ValueError,match='dimensions'):M.native_frame(tmp_path,'synthetic_test_fixture',5960)
    Image.new('RGB',(1920,804)).save(path)
    hashes=iter(('0'*64,'1'*64))
    monkeypatch.setattr(M,'sha',lambda p:next(hashes))
    with pytest.raises(ValueError,match='changed during'):M.native_frame(tmp_path,'synthetic_test_fixture',5960)


def test_round5_overlay_keeps_completion_scope_and_exclusive_offset_separate(tmp_path):
    for name in M.INPUTS:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}\n')
    data=dict(frames=[dict(frame=f,path=f'renders/synthetic_fixture/f_{f:05d}.jpg',sha256='0'*64)
                      for f in range(5840,6080)],source_sha256={},limits='Synthetic schema test only',
              flare=dict(frame=5897,thresholds={'50':5897,'100':5897,'200':5897}),
              caption_onset=dict(frame=5960,thresholds={'strict':5960,'primary':5960,'loose':5960}),
              caption_completion=dict(frame=5983,thresholds={'strict':5983,'primary':5983,'loose':5983}))
    overlay=M.round5_overlay(data,tmp_path)
    assert overlay['events']['D.new.oldfire.quill']['frame']==5960
    end=overlay['bounds']['D.new.oldfire.quill']['end']
    assert end['frame']==5983 and end['offset_f']==1
    assert end['measurement']['claim']=='completion'
    assert end['measurement_scope']=='native_delivered_plate'
    assert len(end['measurement']['frames'])==117
    assert end['measurement']['frames'][0]['frame']==5963
    assert end['measurement']['frames'][-1]['frame']==6079
    assert not overlay['suppressions']  # Never silently mint new leaf source pins.
