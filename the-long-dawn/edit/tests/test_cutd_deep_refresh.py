"""Versioned refresh gates and byte reuse; tiny isolated fixtures, no farm or native rendering."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

EDIT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(EDIT),str(EDIT/'tools')]
import assemble as AS
import rebake_cutd_deep as B


@pytest.fixture
def refresh(monkeypatch,tmp_path):
    root=tmp_path/'repo';(root/'edit').mkdir(parents=True)
    (root/'edit/edl_v3.py').write_text('immutable owner edit\n')
    native=tmp_path/'native.png';native.write_bytes(b'contract-native-fixture')
    contract=tmp_path/'contract.json'
    native_rgb=np.full((2,3,3),70,np.uint8)
    contract.write_text(json.dumps(dict(schema='openring.race-ending.r5',frame=2719,
        farm_plate='renders/embers_D_race/f_02719.jpg',native_plate=native.name,
        native_dimensions=[1920,804],render_sha256=hashlib.sha256(native_rgb.tobytes()).hexdigest())))
    (tmp_path/'delivery_consistency.json').write_text(json.dumps(dict(pass_all=True,
        artifact_sha256={native.name:B.digest(native),contract.name:B.digest(contract)})))
    state=tmp_path/'state.json'
    state.write_text(json.dumps(dict(owner_edl_sha256=B.digest(root/'edit/edl_v3.py'))))
    args=SimpleNamespace(out=tmp_path/'r5/local_renders',reuse_from=tmp_path/'r4/local_renders',
                         contract=contract,owner_state=state,race_after='2026-09-30T16:46:00-04:00')
    folder=root/'renders/embers_D_race';folder.mkdir(parents=True)
    for f in range(2400,2720):
        (folder/f'f_{f:05d}.jpg').write_bytes(b'synthetic-jpeg-index-fixture')
    farm=folder/'f_02719.jpg'
    floor=int(datetime.fromisoformat(args.race_after).timestamp()*1_000_000_000)
    os.utime(farm,ns=(floor+1_000_000_000,floor+1_000_000_000))
    monkeypatch.setattr(B,'ROOT',root)
    monkeypatch.setattr(AS,'RENDERS',str(root/'renders'))
    monkeypatch.setattr(AS,'_INDEX',{})
    take=AS.EDL.T('embers_D_race',0,'exact',need=(2400,2719))
    monkeypatch.setattr(AS.EDL,'D_DEEP_RACE_SOURCE',take)
    monkeypatch.setattr(AS.EDL,'D_DEEP_RACE_UNDER',('take',take,2719,'D'))
    monkeypatch.setattr(B,'NATIVE_SHAPE',(2,3,3))
    monkeypatch.setattr(B.cv2,'imread',lambda *args:native_rgb[...,::-1] if native.read_bytes()==b'contract-native-fixture'
                        else np.zeros_like(native_rgb))
    return args,root,farm,floor


CONTRACT_VERSIONS=(('openring.race-ending.r5','render_sha256','float_rgb_sha256'),
                   ('openring.race-ending.r6','float_rgb_sha256','render_sha256'))


def write_contract(args,contract):
    """A fixture producer attests the exact contract file, independent of its schema validity."""
    args.contract.write_text(json.dumps(contract))
    receipt_path=args.contract.parent/'delivery_consistency.json'
    receipt=json.loads(receipt_path.read_text())
    receipt['artifact_sha256'][args.contract.name]=B.digest(args.contract)
    receipt_path.write_text(json.dumps(receipt))


@pytest.mark.parametrize('schema,required,other',CONTRACT_VERSIONS)
def test_each_contract_version_uses_its_required_hash_and_preserves_owner_pin(refresh,schema,required,other):
    args,root,farm,floor=refresh
    contract=json.loads(args.contract.read_text())
    contract.update(schema=schema,**{required:'a'*64,other:'b'*64})
    write_contract(args,contract)
    result=B.refresh_gate(args)
    assert result['status']=='ready' and result['declared_render_buffer_sha256']=='a'*64
    assert result['farm_sha256']==B.digest(farm) and not args.out.exists()
    owner=root/'edit/edl_v3.py'
    owner.write_text(owner.read_text()+'unauthorized owner change\n')
    with pytest.raises(ValueError,match='Owner EDL'):
        B.refresh_race(args)
    assert not args.out.exists()


@pytest.mark.parametrize('schema',['openring.race-ending.r4','openring.race-ending.r6.extra',None])
def test_unknown_contract_schema_is_rejected_before_asset_copy(refresh,monkeypatch,schema):
    args,root,farm,floor=refresh
    contract=json.loads(args.contract.read_text())
    contract.update(schema=schema,float_rgb_sha256='c'*64)
    write_contract(args,contract)
    monkeypatch.setattr(B,'reuse_assets',lambda *a:pytest.fail('Assets copied for an unsupported schema'))
    with pytest.raises(ValueError,match='race-ending contract schema'):
        B.refresh_race(args)
    assert not args.out.exists()


@pytest.mark.parametrize('schema,required,other',CONTRACT_VERSIONS)
@pytest.mark.parametrize('wrong_key_only',[False,True],ids=['missing_key','wrong_key_only'])
def test_version_specific_hash_is_required_without_fallback(refresh,monkeypatch,schema,required,other,wrong_key_only):
    args,root,farm,floor=refresh
    contract=json.loads(args.contract.read_text())
    contract['schema']=schema
    contract.pop('render_sha256',None)
    contract.pop('float_rgb_sha256',None)
    if wrong_key_only:
        contract[other]='d'*64
    write_contract(args,contract)
    monkeypatch.setattr(B,'reuse_assets',lambda *a:pytest.fail('Assets copied without the required hash'))
    with pytest.raises(ValueError,match=required):
        B.refresh_race(args)
    assert not args.out.exists()


@pytest.mark.parametrize('schema,required,other',CONTRACT_VERSIONS)
@pytest.mark.parametrize('mismatch',['native','contract','pass_all'])
def test_both_contract_versions_require_matching_delivery_receipt(refresh,monkeypatch,schema,required,other,mismatch):
    args,root,farm,floor=refresh
    contract=json.loads(args.contract.read_text())
    value=contract.pop('render_sha256')
    contract.update(schema=schema,**{required:value})
    write_contract(args,contract)
    receipt_path=args.contract.parent/'delivery_consistency.json'
    receipt=json.loads(receipt_path.read_text())
    if mismatch=='pass_all':
        receipt['pass_all']=False
    else:
        name=contract['native_plate'] if mismatch=='native' else args.contract.name
        receipt['artifact_sha256'][name]='0'*64
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(B,'reuse_assets',lambda *a:pytest.fail('Assets copied with an invalid delivery receipt'))
    with pytest.raises(ValueError,match='contract file hashes'):
        B.refresh_race(args)
    assert not args.out.exists()


def test_complete_fresh_selected_delivery_passes_but_one_missing_frame_does_not(refresh):
    args,root,farm,floor=refresh
    result=B.refresh_gate(args)
    assert result['status']=='ready' and result['inventory']['available']==320
    assert result['farm_sha256']==B.digest(farm)
    assert result['farm_dimensions']==[3,2]  # synthetic fixture; the production NATIVE_SHAPE is804x1920
    (farm.parent/'f_02401.jpg').unlink()
    (farm.parent/'f_02401.jpeg').touch()  # unsupported extension cannot rescue the actual frame
    assert B.refresh_gate(args)['status']=='pending'
    result=B.refresh_race(args)
    assert result['inventory']['available']==319 and not args.out.exists()


def test_freshness_is_strict_and_current_selected_jpeg_is_required(refresh,monkeypatch):
    args,root,farm,floor=refresh
    os.utime(farm,ns=(floor,floor))
    assert B.refresh_gate(args)['status']=='pending'
    os.utime(farm,ns=(floor+1_000_000_000,floor+1_000_000_000))
    assert B.refresh_gate(args)['status']=='ready'
    (farm.parent/'f_02719.png').write_bytes(b'older preferred PNG')
    assert B.refresh_gate(args)['status']=='pending'
    args.race_after='2026-09-30T16:46:00'
    with pytest.raises(ValueError,match='UTC offset'):
        B.refresh_gate(args)


@pytest.mark.parametrize('bad_image',[None,np.zeros((1,3,3),np.uint8)],ids=['unreadable','wrong_native_shape'])
def test_raw_farm_decode_rejects_bad_delivery_before_copy_but_preserves_pending_gate(refresh,monkeypatch,bad_image):
    args,root,farm,floor=refresh
    original=B.cv2.imread
    farm_reads=[]
    def read(path,flag):
        if Path(path)==farm:
            farm_reads.append(path)
            return bad_image
        return original(path,flag)
    monkeypatch.setattr(B.cv2,'imread',read)
    monkeypatch.setattr(B,'reuse_assets',lambda *a:pytest.fail('Assets copied before raw farm validation'))
    os.utime(farm,ns=(floor,floor))
    assert B.refresh_race(args)['status']=='pending' and not farm_reads
    os.utime(farm,ns=(floor+1_000_000_000,floor+1_000_000_000))
    missing=farm.parent/'f_02401.jpg';missing.unlink()
    assert B.refresh_race(args)['status']=='pending' and not farm_reads
    missing.write_bytes(b'synthetic-jpeg-index-fixture')
    with pytest.raises(ValueError,match='must decode natively'):
        B.refresh_race(args)
    assert farm_reads==[str(farm)] and not args.out.exists()


def test_owner_and_native_contract_pins_and_immutable_output_are_required(refresh):
    args,root,farm,floor=refresh
    owner=root/'edit/edl_v3.py'
    original=owner.read_bytes();owner.write_bytes(original+b'changed')
    with pytest.raises(ValueError,match='Owner EDL'):
        B.refresh_gate(args)
    owner.write_bytes(original)
    native=args.contract.parent/'native.png';native.write_bytes(b'wrong native')
    with pytest.raises(ValueError,match='contract file hashes'):
        B.refresh_gate(args)
    args.out=args.reuse_from/'nested'
    with pytest.raises(ValueError,match='immutable'):
        B.refresh_gate(args)


def make_reusable(root):
    for stem,first,last,suffix in (('cutd_deep_clean',2720,2757,'png'),
                                   ('cutd_deep_sweep_coeff',2720,2757,'npz'),
                                   ('cutd_deep_exit',2945,2980,'png'),
                                   ('cutd_deep_exit_matte',2945,2980,'png')):
        (root/stem).mkdir(parents=True)
        for f in range(first,last+1):
            (root/stem/f'f_{f:05d}.{suffix}').write_bytes(f'{stem}/{f}'.encode())


def test_reuse_is_exact_and_never_overwrites_different_or_redirected_assets(tmp_path):
    source,out=tmp_path/'old',tmp_path/'new'
    make_reusable(source)
    rows=B.reuse_assets(source,out)
    assert len(rows)==148
    assert all(B.digest(source/r['file'])==B.digest(out/r['file'])==r['sha256'] for r in rows)
    assert B.reuse_assets(source,out)==rows  # idempotent byte verification
    changed=out/rows[0]['file'];changed.write_bytes(b'changed')
    with pytest.raises(ValueError,match='overwrite'):
        B.reuse_assets(source,out)
    assert B.digest(source/rows[0]['file'])==rows[0]['sha256']
    linked=tmp_path/'linked';linked.mkdir()
    (linked/'cutd_deep_clean').symlink_to(source/'cutd_deep_clean',target_is_directory=True)
    with pytest.raises(ValueError,match='outside'):
        B.reuse_assets(source,linked)


def test_runtime_composes_all38_frames_and_detects_glow_drift(monkeypatch,tmp_path):
    out=tmp_path/'new';out.mkdir()
    reused=tmp_path/'old'
    farm=tmp_path/'fresh.jpg'
    monkeypatch.setattr(B,'NATIVE_SHAPE',(2,3,3))
    monkeypatch.setattr(B,'save',lambda *a:None)
    monkeypatch.setattr(AS,'_INDEX',{})
    monkeypatch.setattr(AS,'_CTX',None)  # mocked _init assigns this global; restore its prior value after the test
    race=np.full((2,3,3),.25,np.float32)
    corrupt={'glow':False,'first':False}
    calls=[]
    class Context:
        shots=[dict(code='D11a'),dict(code='D11b')]
        plans=[dict(take=AS.EDL.D_DEEP_ENTRY)]*2
        def read(self,path):
            assert path==str(farm)
            return race
        def take_frame(self,take,f):
            return race+float(corrupt['first'])
        def picture(self,f):
            calls.append(f)
            is_entry=2720<=f<2758
            delta=float(corrupt['glow'] and os.environ['CUTD_LOCAL_RENDERS']==str(out) and not is_entry)
            return race+delta,None,'entry' if is_entry else 'cutd_deep_exit + deep_reveal',AS.EDL.D_DEEP_ENTRY
    monkeypatch.setattr(AS,'_init',lambda *a:setattr(AS,'_CTX',Context()))
    monkeypatch.setenv('CUTD_LOCAL_RENDERS','previous')
    result=B.refresh_runtime(out,reused,farm)
    assert [r['frame'] for r in result['frames']]==list(range(2720,2758))
    assert [f for f in calls if 2720<=f<2758]==list(range(2720,2758))
    assert all(r['before_sha256']==r['after_sha256'] for r in result['glow_identity'])
    assert os.environ['CUTD_LOCAL_RENDERS']=='previous'
    corrupt['glow']=True
    with pytest.raises(ValueError,match='changed glow'):
        B.refresh_runtime(out,reused,farm)
    corrupt['glow']=False;corrupt['first']=True
    with pytest.raises(ValueError,match='fresh held race'):
        B.refresh_runtime(out,reused,farm)


@pytest.mark.parametrize('pending',[False,True])
def test_refresh_reports_total_cost_and_distinguishes_unrun_native_work(refresh,monkeypatch,pending):
    args,root,farm,floor=refresh
    gate=B.refresh_gate(args)
    gate['status']='pending' if pending else 'ready'
    monkeypatch.setattr(B,'refresh_gate',lambda _:gate)
    called=[]
    def reuse(*unused):
        args.out.mkdir(parents=True)
        return []
    monkeypatch.setattr(B,'reuse_assets',reuse)
    monkeypatch.setattr(B,'refresh_runtime',lambda *a:called.append(True) or dict(frames=[]))
    before=time.perf_counter()
    report=B.refresh_race(args)
    after=time.perf_counter()
    assert 0 <= report['total_seconds'] <= after-before
    assert isinstance(report['peak_rss_bytes'],int) and report['peak_rss_bytes']>0
    assert report['native_runtime_ran'] is (not pending)
    assert bool(called) is (not pending)
    assert (report['runtime'] is None) is pending
