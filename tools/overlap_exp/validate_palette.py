"""Python port of the dataviz palette validator's six checks (no node on this box).
Thresholds, OKLab math and the Machado 2009 CVD matrices are copied verbatim from
scripts/validate_palette.js so the verdicts match."""
import math, os, sys
BAND={'light':(0.43,0.77),'dark':(0.48,0.67)}
CHROMA_FLOOR=0.10; CVD_TARGET=8.0; CVD_FLOOR=6.0; NORMAL_FLOOR=15.0; CONTRAST_MIN=3.0
SURF={'light':'#fcfcfb','dark':'#1a1a19'}
M={'protan':[[0.152286,1.052583,-0.204868],[0.114503,0.786281,0.099216],[-0.003882,-0.048116,1.051998]],
   'deutan':[[0.367322,0.860646,-0.227968],[0.280085,0.672501,0.047413],[-0.011820,0.042940,0.968881]]}
h2s=lambda h:[int(h.lstrip('#')[i:i+2],16)/255 for i in (0,2,4)]
s2l=lambda c: c/12.92 if c<=0.04045 else ((c+0.055)/1.055)**2.4
lin=lambda h:[s2l(c) for c in h2s(h)]
def oklab(rgb):
    r,g,b=rgb
    l=(0.4122214708*r+0.5363325363*g+0.0514459929*b)**(1/3)
    m=(0.2119034982*r+0.6806995451*g+0.1073969566*b)**(1/3)
    s=(0.0883024619*r+0.2817188376*g+0.6299787005*b)**(1/3)
    return (0.2104542553*l+0.7936177850*m-0.0040720468*s,
            1.9779984951*l-2.4285922050*m+0.4505937099*s,
            0.0259040371*l+0.7827717662*m-0.8086757660*s)
def och(h):
    L,a,b=oklab(lin(h)); return L, math.hypot(a,b), (math.degrees(math.atan2(b,a))%360)
def sim(h,k):
    r,g,b=lin(h); Mk=M[k]
    return [min(1,max(0,Mk[i][0]*r+Mk[i][1]*g+Mk[i][2]*b)) for i in range(3)]
def dE(a,b,k=None):
    x=oklab(sim(a,k) if k else lin(a)); y=oklab(sim(b,k) if k else lin(b))
    return 100*math.dist(x,y)
def rl(h):
    r,g,b=lin(h); return .2126*r+.7152*g+.0722*b
def cr(a,b):
    x,y=sorted((rl(a),rl(b)),reverse=True); return (x+.05)/(y+.05)

# A SEQUENTIAL ramp is judged by different rules than a categorical palette, and running
# the categorical ones on it produces false failures: its dark end legitimately has low
# chroma and its light end legitimately has low contrast, because the ramp encodes
# magnitude by lightness rather than identity by hue. The JS validator has both rule
# sets; this port originally had only the categorical one, and flagged a perfectly good
# amber ramp as FAIL.
#   ordinal:  adjacent OKLCH dL >= 0.06, lightest step >= 2.0:1 against the surface
ORDINAL_MIN_DL = 0.06
ORDINAL_LIGHT_FLOOR = 2.0

pal=sys.argv[1].split(','); mode=sys.argv[2] if len(sys.argv)>2 else 'light'
surf=sys.argv[3] if len(sys.argv)>3 else SURF[mode]
if os.environ.get("DIVERGING"):
    # A diverging map is judged differently again. Its midpoint represents ZERO and is
    # meant to recede, so the ordinal "lightest step >= 2:1" check fails it by design --
    # running that ruleset flagged a correct grey midpoint as a defect. What must hold is
    # that the two POLES separate from each other and from the midpoint, and that the
    # midpoint is genuinely neutral rather than a third hue.
    lo, mid, hi = pal[0], pal[len(pal)//2], pal[-1]
    d_poles = min(dE(lo,hi,'protan'), dE(lo,hi,'deutan'))
    d_lo = min(dE(lo,mid,'protan'), dE(lo,mid,'deutan'))
    d_hi = min(dE(mid,hi,'protan'), dE(mid,hi,'deutan'))
    c_mid = och(mid)[1]
    ok = d_poles>=8.0 and min(d_lo,d_hi)>=8.0 and c_mid<=0.04
    print(f"palette {pal}  mode={mode}   [DIVERGING ruleset]")
    print(f"  [{'PASS' if d_poles>=8 else 'FAIL'}] pole vs pole      dE {d_poles:.1f} (target 8)")
    print(f"  [{'PASS' if d_lo>=8 else 'FAIL'}] pole vs midpoint  dE {d_lo:.1f} / {d_hi:.1f}")
    print(f"  [{'PASS' if c_mid<=0.04 else 'FAIL'}] midpoint neutral  chroma {c_mid:.3f} (max 0.04)")
    print(f"  => {'ALL PASS' if ok else 'HAS FAILURES'}")
    sys.exit(0)

if os.environ.get("ORDINAL"):
    print(f"palette {pal}  mode={mode}  surface={surf}   [SEQUENTIAL ruleset]")
    dl=[abs(och(a)[0]-och(b)[0]) for a,b in zip(pal,pal[1:])]
    lightest=max(pal,key=lambda c:och(c)[0]); crl=cr(lightest,surf)
    hs=[och(c)[2] for c in pal]
    spread=max(hs)-min(hs); spread=min(spread,360-spread)
    ok=min(dl)>=ORDINAL_MIN_DL and crl>=ORDINAL_LIGHT_FLOOR and spread<=40
    print(f"  [{'PASS' if min(dl)>=ORDINAL_MIN_DL else 'FAIL'}] adjacent dL >= {ORDINAL_MIN_DL}"
          f"      {[round(d,3) for d in dl]}")
    print(f"  [{'PASS' if crl>=ORDINAL_LIGHT_FLOOR else 'FAIL'}] lightest step contrast"
          f"     {lightest} at {crl:.2f}:1 (floor {ORDINAL_LIGHT_FLOOR})")
    print(f"  [{'PASS' if spread<=40 else 'FAIL'}] single hue                 spread {spread:.0f} deg")
    print(f"  => {'ALL PASS' if ok else 'HAS FAILURES'}")
    sys.exit(0)
lo,hi=BAND[mode]; ok=True
print(f"palette {pal}  mode={mode}  surface={surf}")
off=[(c,round(och(c)[0],3)) for c in pal if not (lo<=och(c)[0]<=hi)]
ok&=not off; print(f"  [{'PASS' if not off else 'FAIL'}] lightness band L {lo}-{hi}   {off or 'all inside'}")
low=[(c,round(och(c)[1],3)) for c in pal if och(c)[1]<CHROMA_FLOOR]
ok&=not low; print(f"  [{'PASS' if not low else 'FAIL'}] chroma floor >={CHROMA_FLOOR}      {low or 'all above'}")
worst=1e9
for i in range(len(pal)-1):
    a,b=pal[i],pal[i+1]; d=min(dE(a,b,'protan'),dE(a,b,'deutan')); worst=min(worst,d)
    st='PASS' if d>=CVD_TARGET else ('FLOOR' if d>=CVD_FLOOR else 'FAIL')
    if d<CVD_FLOOR: ok=False
    print(f"  [{st:>5}] CVD adjacent {a} vs {b}: dE {d:.1f}  (target {CVD_TARGET}, floor {CVD_FLOOR})")
nf=min(dE(pal[i],pal[i+1]) for i in range(len(pal)-1))
ok&= nf>=NORMAL_FLOOR
print(f"  [{'PASS' if nf>=NORMAL_FLOOR else 'FAIL'}] normal-vision floor >= {NORMAL_FLOOR}: worst adjacent dE {nf:.1f}")
for c in pal:
    v=cr(c,surf); 
    if v<CONTRAST_MIN: ok=False
    print(f"  [{'PASS' if v>=CONTRAST_MIN else 'WARN'}] contrast {c} vs surface: {v:.2f}:1  (min {CONTRAST_MIN})")
print(f"  => {'ALL PASS' if ok else 'HAS FAILURES'}")
