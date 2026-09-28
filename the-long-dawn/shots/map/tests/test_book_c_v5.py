"""Small contracts for opt-in v5 pages; image approval remains a separate gate."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import book_c_v5 as V
import v5_inkpages as P
import v5_pen as PP
import pages as PG
import book as B
import pen


class V5PageContracts(unittest.TestCase):
    def test_absolute_ranges_exclude_existing_first_half(self):
        self.assertEqual(V.SHOTS,{'refusal':(2080,2320),'deep_abandoned':(4240,4480),'pen':(5440,5680)})
        for shot,(a,b) in V.SHOTS.items():
            parsed=V.parse_args(['--shot',shot,'--frames',f'{a},{b-1}','--out','/tmp/test-v5'])
            self.assertEqual(parsed.frame_list,[a,b-1])
            for frame in (a-1,b):
                with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    V.parse_args(['--shot',shot,'--frames',str(frame),'--out','/tmp/test-v5'])

    def test_cli_requires_explicit_absolute_output(self):
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            V.parse_args(['--shot','pen','--frames','5440','--out','renders/book_C'])

    def test_refusal_drawing_is_deterministic_progressive_and_leaves_caption_band(self):
        a=P.refusal().pack();b=P.refusal().pack()
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        self.assertGreater(len(a['T0']),100)
        self.assertTrue(np.isfinite(a['P']).all())
        self.assertLess(a['P'][:,1].max(),18)
        self.assertGreater(a['T0'].min(),0)
        self.assertLessEqual(a['T1'].max(),7.1)
        self.assertGreater(np.sum((a['T0']<2)&(a['T1']>1)),0)
        self.assertGreater(np.sum(a['T0']>5),0)

    def test_mine_retains_every_non_miner_stroke_exactly(self):
        class Recorded(PG.Deep):
            def __init__(self):super().__init__();self.removed=[]
            def _miner(self,S,*args):
                first=len(S);super()._miner(S,*args);self.removed.extend(range(first,len(S)))
        old=Recorded();original=old.build('ink',.6,9.6)
        new=P.AbandonedDeep();abandoned=new.build('ink',.6,9.6)
        self.assertGreater(len(old.removed),0)
        keep=[i for i in range(len(original)) if i not in set(old.removed)]
        self.assertEqual(len(keep),len(abandoned))
        # Timing windows differ after removing workers; geometry, radius,
        # material, density and RNG-driven vein branches must remain identical.
        for key in ('P','R','D','L','G','N'):
            for i,j in enumerate(keep):
                np.testing.assert_array_equal(getattr(abandoned,key)[i],getattr(original,key)[j])
        np.testing.assert_array_equal(new.vein,old.vein)
        self.assertEqual(new.halls,old.halls)
        self.assertGreater(len(new.additions()),40)

    def test_physical_pen_has_volume_and_clears_cockled_page(self):
        bk=B.Book(TL=2.8,TR=1.2,seed=13)
        verts,normals,uv,faces,mats=PP.geometry(bk)
        self.assertTrue(np.isfinite(verts).all())
        self.assertEqual(set(mats),{0,1,2})
        self.assertGreater(verts[:,2].ptp() if hasattr(verts[:,2],'ptp') else np.ptp(verts[:,2]),1.)
        self.assertGreater(len(faces),300)
        np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1,atol=1e-12)
        heights=np.array([B.height(v[0],v[1],bk.params,bk.ck)[0] for v in verts])
        self.assertGreater((verts[:,2]-heights).min(),-.01)

    def test_resting_pen_has_support_on_both_sides_of_mass(self):
        bk=B.Book(TL=2.8,TR=1.2,seed=13)
        vertices,_,uv,_,_=PP.geometry(bk)
        levels=np.unique(uv[:,0])
        centres=np.array([vertices[uv[:,0]==q].mean(0) for q in levels])
        bottom=np.array([vertices[uv[:,0]==q,2].min() for q in levels])
        q=np.linspace(0,1,1001)
        axis=centres[0]+q[:,None]*(centres[-1]-centres[0])
        lower=np.interp(q,levels,bottom)
        ground=np.array([B.height(p[0],p[1],bk.params,bk.ck)[0] for p in axis])
        gap=lower-ground
        self.assertLess(gap[q<.5].min(),.035)
        self.assertLess(gap[q>.7].min(),.035)
        self.assertGreater(gap.min(),-.01)

    def test_projection_depth_is_camera_forward_and_perspective_correct(self):
        cam=B.Cam([0,-8,8],[0,0,0],45,64,32)
        vertices=np.array([[-2,0,0],[2,0,0],[0,1,1.]])
        sl,w,z,m=PP._pixels(cam,vertices,[0,1,2])
        reconstructed=w@vertices
        expected=(reconstructed-cam.pos)@cam.f
        np.testing.assert_allclose(z[m],expected[m],atol=1e-12)

    def test_pen_compositor_hits_visible_volume_and_respects_foreground_depth(self):
        bk=B.Book(TL=2.8,TR=1.2,seed=13)
        cam=B.Cam([0,-8,15],[0,0,3],45,64,32)
        vertices=np.array([[-2,0,6],[2,0,6],[0,1,7.]])
        mesh=(vertices,np.tile([0,0,1.],(3,1)),np.zeros((3,2)),np.array([[0,1,2]]),np.array([0]))
        light=B.Light([-30,20,30])
        for original_depth,expect_hit in ((100.,True),(1.,False)):
            G=np.zeros((32,64,16));G[...,1]=original_depth
            rgb,depth=PP.composite(np.ones((32,64,3),np.float32),G,bk,cam,light,mesh)
            self.assertTrue(np.isfinite(rgb).all())
            self.assertEqual(bool((depth<original_depth).any()),expect_hit)

    def test_jpeg_is_444_and_absolute_numbering_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'f_05440.jpg'
            V.save(p,np.ones((8,12,3),np.float32)*.5,'jpg')
            with Image.open(p) as im:
                self.assertEqual(im.size,(12,8))
                self.assertEqual(im.layer[0][1:3],(1,1))
            self.assertEqual(sorted(x.name for x in p.parent.iterdir()),['f_05440.jpg'])

    def test_job_output_and_range_contracts(self):
        for shot,(a,b) in V.SHOTS.items():
            job=json.loads((HERE.parents[1]/'cloud/jobs'/f'book_C5_{shot}.json').read_text())
            self.assertEqual(job['branch'],'codex/c-v5-ink-pages')
            self.assertEqual(job['shape'],[804,1920])
            self.assertEqual(len(job['render']),1)
            self.assertIn(f'--frames {a}-{b-1}',job['render'][0])
            for out in job['outputs']:self.assertEqual(out['frames'],f'{a}-{b-1}')
            self.assertIn('"$(pwd)/'+job['outputs'][0]['out_dir']+'"',job['render'][0])


if __name__=='__main__':unittest.main()
