# False-dawn returns in cut D v2

`glowvars.render()` defaults to the accepted clear-high-deck false dawn. A500 is
shot frame 420 on this route. Existing false-dawn code and jobs are unchanged.
The explicit variants use absolute cut-D-v2 frames: `brink` 4080–4239,
`twofires` 4560–5039, `watch` 7040–7359, and `truedawn` 7360–7839.
All share A500's locked camera, terrain, cloud geometry and finish.
The crown match fixes the camera at frame 420; other poses are rejected.

`cloud/glowvars_render.py` renders native 1920×804 frames at 1.5× supersampling.
Choose a fresh output directory; it refuses the owner's linked renders and
existing frames. Farm manifests write only `renders/falsedawn_<variant>` and
use `claude/owner-night-20260929`. They are specifications, not submissions.

```sh
python3 cloud/glowvars_render.py --variant twofires --frames 4560,4980 --out /tmp/glowvars-twofires
```

On a shared production machine, run rendering through its prescribed slot
wrapper. The CLI prints measured wall seconds and process peak RSS per frame.
Its PNG writer uses the existing random output dither; compatibility is pinned
on HDR and finished RGB arrays before that dither, plus the accepted source hash.

## Timing and staging

The bar map uses 24 fps, 72 BPM, 4/4: 20 frames per beat, 80 per bar.
`--beat-offset N` adds N to the D frame for musical phase. Both racing returns
surge every 20 frames throughout. The pair ignites on D4320, before our first
frame D4560. Five approaching catches run D4760–4920 at 40-frame intervals;
the nearest is fully burning by D4980. A nonzero offset moves the first cascade
catch to the next aligned two-beat boundary. `--pair-frame` and `--cascade-frame`
allow editorial changes in absolute D frames.

Watch and dawn have a warm breath each 80 frames. Their fire clocks, clouds,
camera and night lighting are continuous across D7359→7360; the dawn then
interpolates smoothly from that lighting over 480 frames. The remaining watch
fires carry ignition D5040, when EVERY RIDGE establishes the lit range.
No figure is added.

The final native basket-root centres are `(733.4482365330008, 463.0645838537898)`
and `(1386.8711906600877, 455.13424081989217)`, held in `CROWN_LEFT_BASE` and
`CROWN_RIGHT_BASE`. Vision matches these anchors. Their measured physical
terrain seats are fixed constants in `glowvars_sites.py`; its catalogue now
checks summit provenance and never retargets the pair. `glowvars_sites.audit()`
reports the coordinates and provenance. Watch and true dawn retain the chain
and add summit fires across the view, for 40 fires in total.

New approaching ignitions must have rendered bases above native y643.2
(the lower-20% caption boundary). The seating pass checks this after resolving
actual terrain support. The former chain fire at y645.2 is now a watch-only
fire, already burning when the watch begins. Captions 12, 17 and 18 can occupy
the lower third without new catches in the protected band.

## Fire and dawn rendering

The flame and particle recipes come from `nighta`/`fire2`; the under-cloud
patches and shader are the same ones used by A13/A14. Distant beacon emission
uses a shallower haze attenuation curve for legibility. The pair has seven
flame tongues, a tapered iron basket, a warm halo and continuous sparks.
The pair uses a broader tongue table and twice the later fires' emission so
the separate tongues remain bright against the racing glow. Smoke and sparks
share a screen-right wind, projected in a local frame about
the fixed root with the flame's screen magnification. Terrain depth still
occludes them. The first pair's nominal flame height is 60 native pixels;
the visible body is shorter and varies with flicker. The five approaching
fires retain nominal heights 20, 24, 32, 36 and 40 pixels; additional watch fires
use 12–26 according to distance.

Separate inverse-square point lights illuminate the existing rock and snow
around each fire. They have soft tails. Enlarging the pair does not multiply
physical fire size or the cleared-rock platforms. Racing under-cloud patches
use deep red-orange light; the existing cloud shader excludes terrain and
sky. Particle caches include ignition time and remain local to this module.

True dawn uses `dawn.dawn_sky` and the blue-hour palette. The sun rises at the
accepted glow's −30° azimuth; sky luminance continuously veils the existing stars.
Its final lighting, sun elevation and palette retain the approved Round 1
endpoint. No camera move accompanies the dawn. Returned arrays remain linear
HDR; finish with `glowvars.FINISH`.

Fire seats are reconstructed from the accepted marcher's depth buffer near each
catalogued summit. The fixed pair instead retains its measured physical seats
and checks that the raster still supports them. Coarser probes check nearby
support while retaining the same anchors. No terrain changes are made. Missing
support is an error. Fog receives physical coordinates; flame projection,
terrain lights and platforms receive curvature-adjusted y.
