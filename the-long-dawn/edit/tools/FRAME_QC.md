# Frame QC

Run from `the-long-dawn` with its existing NumPy/OpenCV environment:

```sh
python edit/tools/frame_qc.py renders/crossing_A --range 4880-5839 --film A --json review/crossing_qc.json
```

The range is inclusive. Input frames are read only; the tool writes the requested
JSON report and prints a short summary. It expects 1920x804 by default. Use
`--width` and `--height` for another known delivery size. When both formats exist,
`f_00001.png` takes precedence over `f_00001.jpg`; the report lists the alternative.
A corrupt PNG remains a decode error even when a JPEG alternative exists.

Exit status is 0 when every requested frame and intersecting join endpoint decodes
in the expected dimensions, 1 for missing/undecodable/wrong-size/unsupported input
frames or unavailable intersecting joins, and 2 for invalid arguments or
report/configuration I/O errors. Entirely outside joins do not affect exit status.
Black frames, holds and pops are **review
candidates**: their presence does not make the process fail or establish an
artistic defect. A deliberate freeze, quiet night, flash or undeclared cut can
produce these candidates. No thresholds have been calibrated against final films.

`frame_qc_joins.json` contains the four declared joins. `--film A`, `B` or `C`
selects that film's joins; omitting the film selects none. `--joins file.json`
replaces the defaults; `--extra-joins file.json` appends. Both require a film.
Files use the same `{"joins": [{"film": "A", "from": 4879, "to": 4880,
"label": "heart into lantern"}]}` shape (a bare list is also accepted). Duplicate
film/frame pairs are rejected. Joins may have nonadjacent endpoints. For each join
whose span intersects the requested range, both endpoint files are inspected in
the same directory, including an endpoint outside that range. Thus a scan starting
at 4880 can compare 4879 if it exists, or report that endpoint missing. Extra reads
appear in `join_endpoint_frames`; each join endpoint records `in_requested_range`.
They do not extend requested-frame counts or adjacent-transition detection.
Nonintersecting joins are reported as `outside_range` without reading their files.
These are global cut frame numbers. Supply files with those names, or custom join
values matching the source files; the tool makes no EDL or source-number mappings.

Declared adjacent cuts are measured separately and excluded from pop detection.
Their destination starts a new baseline segment; missing, undecodable and
wrong-size frames also break segments. Adjacent metrics never jump over a bad
frame. Exact duplicate pairs at declared cuts remain recorded, but are excluded
from the summary's held-frame candidate count.

The reported luminance is display-encoded Rec.709-weighted RGB on [0,1], not
physical luminance. Black means every color component is at most `--black-max`
(default .01). Constant means each color channel is exactly spatially uniform.
Other frames are near-constant when both luminance standard deviation and full
range are at most `--near-constant-std` (.002) and `--near-constant-range` (.02).
Flags can overlap. Grayscale and 8/16-bit PNGs are supported. Alpha is excluded
from luminance; fully transparent frames receive a separate flag.

For each valid consecutive pair, luminance change is the absolute difference in
frame means. Structure change is the mean absolute difference between luminance
thumbnails after subtracting each thumbnail's own mean. Thumbnails preserve the
frame aspect ratio, with width at most `--thumbnail-width` (96). Very small or
color-only changes can escape these luminance-based metrics.

Each pop threshold is the larger of its absolute floor and local median +
`--pop-mad-multiplier` (6) times 1.4826 MAD. Neighbors lie within `--pop-window`
(12) frame transitions on each side, in the same uninterrupted segment, excluding
the tested transition and declared joins. Below `--pop-min-neighbors` (4), only
the absolute floor is used. The floors are `--pop-luma-floor` (.08) and
`--pop-structure-floor` (.06). JSON includes neighbor counts, medians, MADs,
thresholds and the fallback method so a flag can be examined and settings changed.
Sustained changes may become the local baseline and need human review.

Holds use exact decoded array equality, including channel layout, bit depth and
alpha; they are independent of PNG/JPEG file-byte equality. JPEG recompression
or PNG dither can prevent an exact match even when a frame looks held. Join
reports additionally compare SHA-256 fingerprints of decoded arrays. Memory
retains the previous image, scalar per-frame/per-transition results, and small
thumbnails for declared join endpoints; it does not load the movie into memory.
OpenCV uses one thread. No input is deleted, repaired or renamed.
