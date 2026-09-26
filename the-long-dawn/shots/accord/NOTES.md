# ACCORD v2 — src frames 1912–2247 (v2 timeline = src + 160) → `renders/accord_A|B|C/f_%05d.png`

Three variants from one renderer (`--variant A|B|C`); v1 (`renders/accord/`, cloud) is untouched.
* **A — Allegory.** The four oaths carved and written in gold at 2000/2040/2080/2120, each rolled upright to the top of frame (as v1), crisp.
* **B — Legend (wordless).** No words. The oath band becomes a carved braid with a fire rising at each emissary's station, lit in the same ember language as the frieze: each station fire catches as its emissary reaches over it with a torch (≈1990–1998), then after the merge the fire runs along the braid from every station until the ring is one (2004–2060) and settles to embers. The stepped roll is replaced by one slow, continuous, eased turn; the council is framed centred (h ≈ 10.5 m) so the whole mandala turns in frame.
* **C — Tolkien.** As A, plus the Ring (see below).

## What changed from v1 (all variants)
* **Emissaries** (`geom.sd_fig`, `scene.py`). Rebuilt as dark, dignified hooded figures seen from above: cloth with crease-profile folds fanning out from the neck, a folded mantle that overhangs every figure (longer on the torch side), hood shells with a real cavity, a brim that overhangs the face and a peak behind; six silhouettes (cowl with liripipe, deep hood, head-wrap with a tail, bare head with the hood thrown back, a slighter veiled figure, a pointed capuchin) at varied heights and widths. All cloth sits in one near-black charcoal-to-umber range; they differ by weave and sheen (`F_WEAVE`, `F_SHEENK`) and cut, not colour. Wool/velvet response (faces turned to us sink dark, edges hold the firelight), a faint warm back-rim from the crowd's torches, a soft fill from the upper flames so the hoods read. After the merge they hold their charred-black spent torches upright before them, hands withdrawn into the sleeves. Near-field depth of field 2000–2140 (`accord.near_dof`: table in focus, heavier toward the frame edges) keeps the edge figures soft. Through the flare their hearth light is compressed (`P_FIGK`), so they stay dark silhouettes against the blaze instead of going pastel.
* **The "together" ring is gone.** In its place the floor band carries an original carved frieze: 48 small three-tongued fires on a running line, the grander one in each emissary's shadow. In the same clockwise sweep (2160–2220) fire runs round it: short tapered flame tongues ride the head of the sweep, blown forward, and die down behind it (`particles.frieze_tongues`), leaving dim, mottled orange embers in the grooves (`shade.frieze_fire`) — firelight in carved stone, never a flat gold fill, far below the hearth. (Rejected on the way: a braid with medallions, which read as a ring of eyes; and a gold fill, which read as a laurel emblem.)
* **No T12 dimming** (no screen band is calmed any more).
* **Temporal fixes** (critics' pass): the ignition glare tapers to zero by 2016 (no pop at 2011); the camera dissolves through the mist layers over 5–14 frames (no strobes at 1946/1955); the light the fire throws on the room breathes at ≤ 1 rad/frame and ≤ ~3.5 % (no 10 Hz dips; the flames themselves still dance). Checked by frame-mean luma: no single-frame dip > 3 % in 1926–1975, 2003–2019, 2176–2231.
* **A/C:** the pull-back to the mandala starts 14 frames later than v1 (keys 2150/2163/2176) so Oath IV holds full size.
* **Crisp text.** Cause of the soft local renders: `accord.py range` inherited argparse's `--scale 0.5` (meant for `still`), so every local "delivery" frame was rendered at half resolution and bilinearly upscaled. A local full-res render of 2010 matches the cloud frame (edge-gradient p99 301 vs 297). `range` now defaults to full resolution, and any reduced scale is written to `renders/accord_<V>/draft/`, never the delivery folder.

## The Ring (C)
`ring.py`. A plain heavy gold band (OD 0.37 m — mythic scale, ~90 px at the oath framing) lying askew (27°) on a bed of grey ash in the hearth. It carries an inscription on both faces in an **invented** broad-nib script generated here from original strokes (arches with curling legs, flame-like stems with coiled heads, U-bowls with rising tails, coils, marks above the line) — no real alphabet, no real text. It glints in the torchlight in the cold hearth as the flames stream in; the letters kindle as the merged fire takes it (2001–2016) and then breathe with a slow pulse (58 frames). The fire is hollowed around it (`FP_HOLLOW`) so it lies visible among the flames; the flare closes over it and the white swallows it.

## Re-render
```
source ~/.venvs/longdawn/env.sh; export NUMBA_NUM_THREADS=1
python shots/accord/accord.py range 1912 2247 --variant A --worker 0/2 &   # + --worker 1/2
python shots/accord/accord.py finish --variant A                          # preview.mp4 + contact.png
python shots/accord/accord.py still 2100 --variant B --scale 0.5          # test still -> renders/accord_B/tests/
python shots/accord/accord.py still 2060 --variant C --window 760,300,400,400   # full-res crop
```
Caches live in `renders/accord_cache/` (carved maps per kind `text`/`orn` as memory-mapped .npy, the plain, the Ring's inscription); delete to rebuild (`python shots/accord/textmaps.py text orn`). Look-dev helpers: `figlook.py` (emissaries from above), `keysheet.py` (key-frame sheet), `textmaps.py preview <kind> out.png`, `ring.py preview out.png`.
Speed: 7–21 s/frame single-threaded uncontended (oath framing is the slowest); the full pass (3 × 336 frames, 2 workers, heavy contention from other departments) took 124 min, i.e. ~15 s/frame/worker on average. Performance work: shadow rays use a cheap silhouette proxy of the figures, the hood math only runs inside a head bounding sphere, and extra AA samples go only to id edges and high-contrast pixels; the fire is ray-marched into its own buffer and softened by a sub-pixel blur (removes v1's cross-hatch jitter).

## Deviations / cheats
* Figure lighting cheats: the crowd back-rim, the upper-flame fill, torches lighting their own bearers at half strength, and the flare compression.
* The oath letters (A, C) stay gold for legibility; every other carving that lights (frieze, B's braid) is ember-light.
* The v1 deviations still hold (merge on the 2000 hit, two-line oaths, lighting cheats for the radial shadows, exposure ride).

Review sheet (v1 | A | B | C at matched frames): `~/mishamisha/_local_logs/review/accord_v2.jpg` (`_local_logs/accord_v2_sheet.py`).
