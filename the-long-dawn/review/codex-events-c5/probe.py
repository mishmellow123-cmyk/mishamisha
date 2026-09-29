"""Contact sheets of EDIT's composed picture; no captions or review burn-in."""
import sys,os,json
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'edit'))
import assemble as A
ctx=A.Ctx('C',scale=.5,clean=True)
edl=json.load(open(ROOT/'edit/edl/edl_C.json'))
assert json.dumps([(s['f0'],s['f1'],s['takes']) for s in ctx.shots],sort_keys=True)==json.dumps([(s['f0'],s['f1'],[{k:v for k,v in t.items() if k!='folders'} for t in s['takes']]) for s in edl['shots']],sort_keys=True)
for s,p in zip(ctx.shots,ctx.plans):
 print(s['f0'],s['f1'],p['kind'],p['have'],p['take']['stem'] if p['take'] else '',flush=True)
 for f in range(s['f0'],s['f1']):
  t=p['take']
  if not t: continue
  for k in ['add','matte']:
   if t.get(k) and f+t['off'] not in A.index(os.path.join(A.RENDERS,t[k])): raise ValueError((f,k,t[k]))
  if t.get('under') and not A.locate_under(t['under'],f)[0]: raise ValueError((f,'under'))
sets={'first':[80,160,240,319,320,360,400,480,559,560,600,640,680,699,700,720,760,800,840,841,850,870,880,920,1039], 'forge':[1040,1080,1120,1160,1200,1240,1280,1320,1360,1400,1439,1440,1480,1520,1560,1600,1679,1680,1720,1800,1904,1920,1940,1960,1992,2000,2040,2079], 'later':[2640,2649,2650,2652,2669,2670,2697,2698,2701,2724,2780,2835,2836,2857,2879,3120,3160,3200,3240,3280,3320,3360,3400,3439,4720,4800,4880,4960,5040,5120,5199,5200,5320,5439,5680,5700,5720,5760,5840,5919]}
if len(sys.argv)>1: sets={sys.argv[1]:[int(f) for f in sys.argv[2].split(',')]}
for name,fs in sets.items():
 im=Image.new('RGB',(5*384,((len(fs)+4)//5)*181),(20,20,20));d=ImageDraw.Draw(im)
 for i,f in enumerate(fs):
  x,*_=ctx.picture(f)
  tile=Image.fromarray((x*255+.5).astype('uint8')).resize((384,161))
  xx,yy=i%5*384,i//5*181;im.paste(tile,(xx,yy+20));d.text((xx+5,yy+3),str(f),fill='white')
 im.save(ROOT/'review/codex-events-c5'/f'{name}.jpg')
