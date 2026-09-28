# Delivered Pen sample audit

No new visual blocker found in the inspected delivery samples. This is a bounded review of the actual final JPEGs, not an approval of the assembled film.

## Visual observations

- **Contact and shadow:** At native resolution in C5440, C5560 and C5679, the pointed nib meets the right page with its shadow converging beneath the tip. The shadow opens into a broader, softer shape beneath the higher shaft. I did not observe the prior suspended-nib pose in these samples. Across C5550–5573 the contact remains visually attached; no sudden shadow displacement or gap appears. These are image observations, not new measurements of surface clearance.
- **Gutter:** The deep central crease stays behind the shaft and remains coherent as the camera approaches. The dark shaft loses some contrast where it crosses the dark gutter, but its diagonal continues across it. No sampled frame shows the shaft cut away, an extra outline or a discontinuous shadow at this crossing.
- **Blank-page space:** Both pages remain unmarked apart from the paper's small texture flecks. The lower parts of the spread retain broad empty space. At 480 pixels wide, the diagonal pen and its soft shadow remain evident; the ferrule and nib split are subdued, so this scale conveys the resting pen principally through its silhouette and book context. Captions were absent and were not tested.
- **Motion:** All 24 consecutive delivered frames C5550–5573 were examined in reduced-composition and unchanged native-pixel nib crops. The position, pen shape, page features and shadow are stable through the slow approach; exposure changes are gradual in the inspected sequence. There is no observed one-frame geometric jump or detached tip. A sheet inspection does not establish how the complete ten-second shot feels in real-time playback.
- **End frame:** C5679 arrived before audit closure and was inspected at native resolution. The complete pen retains margin, the nib remains in contact and the broad blank lower-page space remains. This endpoint adds no new blocker; the frames between the motion window and the end were not visually audited here.

## Source and evidence

The source explicitly caches one pen mesh for the fixed cockled book and moves only the camera and hearth light during this shot: `book_c_v5.py:64–74`, with book/mesh caching in `book_c.py:378–387` and the light motion in `book_c.py:442–447`. `v5_pen.py:20–33` solves the supported pose; its shadow is projected along the light direction onto sampled page height (`:78–94`). The earlier [review handback](../../HANDBACK.md) documents why the initial lifted pose was rejected and how the support regression was tested. I read those tests and notes but did not rerun geometry or render code in this audit.

The four read source files match their SHA256 entries in the renderer receipt. Exact sample filenames, hashes, dimensions and receipt state are in [evidence.json](evidence.json). The audit captures 26 unique RGB JPEGs and 26 matching mattes (C5440, C5550–5573 and C5679); the matte at C5560 was also visually inspected. All sampled images decoded at 1920×804. The final evidence snapshot reads the owner's receipt with status `verified`: all 26/26 RGB and 26/26 matte SHA256 hashes match its recorded entries. The receipt contains 240 RGB and 240 matte entries; this audit independently compares only the 52 selected files.

- [24 reduced compositions](motion-composition-5550-5573.png)
- [24 native-pixel nib/shadow crops](motion-nib-native-5550-5573.png)
- [First/middle/end pen crops at native pixels](key-pen-native.png)
- [First/middle/end composition](key-composition.png)
- [Native gutter crops](key-gutter-native.png)

No renderer or compiled kernel was called. Input JPEGs, mattes, receipts and sources were unchanged. The sheets use area reduction for composition and unchanged pixels for native crops, with labels outside the image. This audit does not duplicate full-shot numeric QC, inspect the other frames, test incoming page-turn/outgoing joins, assess final captions or sound, or claim audience comprehension.

Publication note: this report is a public copy. Private paths are normalized; package-link changes and original file SHA256 are recorded in [PUBLICATION.json](PUBLICATION.json). Original delivery-receipt and sample hashes are preserved. Helper scripts are not included.
