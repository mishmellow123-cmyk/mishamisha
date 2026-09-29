# C5 delivered-picture synchronization audit

Continued the interrupted lane in its existing worktree. Existing contact sheets and candidate detectors were inspected; every shot was re-read from delivered frames. No audio was rendered. All frame numbers below are absolute C5 cut frames at 24 fps.

## Result and limits

72 sync ids: 1 derived, 37 measured, 32 pass 1, 2 verified.

The added sections use `assemble.Ctx.picture` at 960×402, before caption overlay; source selection uses the exported EDL and EDIT’s `plan_shot`/`locate`/`index`/`chain` implementation. Composite layers are checked for presence, and each source file is hashed. The previously measured nine shots retain their native/half-resolution detector conventions. No frames were written under renders/.

FLINT has two measured spark flashes and no black frame. Catch and blowing onset remain unresolved: the candidate detectors also trigger on glowing tinder or the strike flash. A visible flame develops later, but these measurements do not establish its first frame; no substitute onset is invented. The seven run catches agree with pass 1. The sun is already partly above the ridge at the illumination cut, so its onset is left-censored. PLENTY is a continuous camera transition; title lettering first becomes detectable after its section entry.

Thresholded onsets are operational pixel measurements, not claims of the earliest perceptible photon. The page catch is scoped after the flying glyph disappears; earlier glyph sparks also satisfy the same brightness mask. Run fastest/full remain null because camera-registered glow novelty does not measure flame-size growth. Medium confidence reflects these scene-specific masks and downsampling. No motion playback or listening judgment is claimed.

## Every sync id

Delta is the selected C5P2 cue minus pass 1. “moves the music” flags every absolute delta ≥2 frames; the score’s actual use of a cue is separately checked below. Added pass-2 ids have no pass-1 frame or delta. A retained cue’s frame is not a measured frame.

| Sync id | Pass 1 | Measured frame(s) / evidence | C5P2 | Delta | Method / retention reason | Confidence |
|---|---:|---|---:|---|---|---|
| hearth | 0 | — | 0 | +0 | Sound cue over EDIT black; no visible hearth event. | not measured |
| page_turn_0 | 40 | — | 40 | +0 | C0-79 are EDIT black; no visible page turn at C40. | not measured |
| book | 80 | — | 80 | +0 | Section/music entry at C80; the book fades in after this boundary. | not measured |
| blank_sheaf | 240 | — | 240 | +0 | Camera drift across illustrated pages; no isolated blank-sheaf arrival established. | not measured |
| riffle | 320 | {"first":326} | 326 | +6 **moves the music** | opening.riffle: mean absolute luma change from previous frame; threshold separates the rapid leaf sweep from the inspected slow camera drift; first = first sample >= 10; selected first | medium |
| ring_motif | 360 | — | 360 | +0 | Musical motif entry over a mountain illustration; no separate small Ring stroke isolated. | not measured |
| page_turn | 560 | {"first":574} | 574 | +14 **moves the music** | letters.page_turn: mean absolute luma change from previous frame; the event is the rapid sweep, not an inferred physical hand contact; first = first sample >= 10; selected first | medium |
| letters_glow | 700 | {"first":705,"half":707,"full":709} | 705 | +5 **moves the music** | letters.glow: count R>90, R-G>20 and luma minus Gaussian(sigma=3)>8; baseline median C680-699, end median C716-729; first/half/full = 10/50/90%; presence requires span >= 100 and no baseline trigger; selected first | medium |
| letters_lift | 720 | {"first":744,"half":750,"full":755} | 744 | +24 **moves the music** | letters.lift: count the glow mask outside a 3-pixel dilation of the C730 glyph mask; camera drift within that dilation is tolerated; measures displaced area, not a physical lift height; baseline median C730-734, end median C756-765; first/half/full = 10/50/90%; presence requires span >= 100 and no baseline trigger; selected first | medium |
| fire_catches | 800 | {"first":801,"fastest":810,"full":817} | 801 | +1 | letters.catch: count luma>150 and R>180 in the central fire ROI; first = first sample >= 100; fastest = largest core-area increase through C840; full = 90% of median core area C818-830 (the catch flash); selected first | medium |
| burn_through | 840 | {"first":856,"half":884,"full":925} | 856 | +16 **moves the music** | letters.burn: pixels with luma<10 in the page interior; composed book_C_ft plus embers_C3_e15, never the matte alone; baseline median C820-839, end median C940-959; first/half/full = 10/50/90%; presence requires span >= 100 and no baseline trigger; selected first | medium |
| fire_alone | 880 | — | 880 | +0 | Section/chord entry during the continuing page burn; no separate ignition at C880. | not measured |
| towers_rise | 1040 | {"first":1069,"half":1088,"full":1089} | 1069 | +29 **moves the music** | forge.towers: count luma>55 and R-B>25 outside x380-589 (exclude the lone central flame); apparent emergence in the moving camera; baseline median C1040-1059, end median C1110-1120; first/half/full = 10/50/90%; presence requires span >= 100 and no baseline trigger; selected first | medium |
| anvil | 1200 | — | 1200 | +0 | No pictured hammer contact identified. Impulse C1262 shows a rotating Ring in adjacent stills, not a strike. | not measured |
| inscription | 1320 | {"first":1321} | 1321 | +1 | forge.inscription: count luma-Gaussian(sigma=3)>3, luma<235, R>160 in the inspected first-glyph ROI; reject the saturated reflection; first = first sample >= 120; selected first | medium |
| ring_rises | 1360 | {"first":1389,"half":1405,"full":1435} | 1389 | +29 **moves the music** | forge.ring_rises: centroid y of the largest connected gold body (R-B>25,R>60,area>=30) in x380-629; track the body rather than its specular highlights; apparent screen-space rise during the camera pullback; baseline median C1378-1384, end median C1440-1450; first/half/full = 10/50/90%; presence requires span >= 20 and no baseline trigger; selected first | medium |
| hush | 1440 | — | 1440 | +0 | Musical hush at the section cut; no discrete picture onset measured. | not measured |
| breath_race | 1555 | — | 1555 | +0 | Musical breath, not a picture event. | not measured |
| race | 1560 | — | 1560 | +0 | Musical acceleration; continuous tower/camera motion, no accepted discrete onset. | not measured |
| deep | 1680 | — | 1680 | +0 | Section boundary within a continuing transition onto parchment. | not measured |
| tick_8 | 1760 | — | 1760 | +0 | Musical subdivision, not a picture event. | not measured |
| tick_16 | 1840 | — | 1840 | +0 | Musical subdivision, not a picture event. | not measured |
| eye_burn | 1920 | {"first":1933,"half":1939,"full":1942} | 1933 | +13 **moves the music** | eye.burn: count luma<30 in the page interior of EDIT RGB+(1-matte)*embers_C3; the backing storm is not black; baseline median C1905-1920, end median C1955-1970; first/half/full = 10/50/90%; presence requires span >= 100 and no baseline trigger; selected first | medium |
| slit_nothing | 2000 | {"first":2003,"half":2005,"full":2033} | 2003 | +3 **moves the music** | eye.slit: count luma<25 in the iris central strip; exclude the earlier page tear by starting after the composited burn; baseline median C1992-2000, end median C2060-2079; first/half/full = 10/50/90%; presence requires span >= 100 and no baseline trigger; selected first | medium |
| mirror | 2080 | — | 2080 | +0 | Section boundary/legacy alias for refusal. | not measured |
| refusal | 2080 | — | 2080 | +0 | Section cut into the Refusal page; no new within-shot event. | not measured |
| old_story | 2120 | {"first":2114,"last":2135} | 2120 | +0 | ink arrival: pixels whose luma DROPS by > 18 from the previous frame (960x402, frame to frame, so a slow drift never accumulates); a stroke frame has >= 10 such pixels; positions in 1920x804; the stroke segment before the Ring's; the offering hand is being drawn when the clarinet's Ring enters (the Ring itself appears on the palm 2138-2150, under the motif's Ab-D) | high |
| turns_away | 2200 | {"first":2195,"last":2202} | 2200 | +0 | ink arrival: pixels whose luma DROPS by > 18 from the previous frame (960x402, frame to frame, so a slow drift never accumulates); a stroke frame has >= 10 such pixels; positions in 1920x804; more than 20 frames after the figure begins, the first run of stroke frames centred at x < 1190, y < 300 (beside the hood); the refusal gesture, the raised hand, is drawn 2195-2202; pass 1's 2200 lies inside it and keeps the cor anglais' V-i landing on 2280 | high |
| trap | 2320 | — | 2320 | +0 | Section cut; inter-shot burn is not supplied by Ctx.picture. | not measured |
| low_fire | 2400 | {"first":2423,"out":2428} | 2423 | +23 **moves the music** | trap.low_forge_sinks: flame pixels: luma > 150 and R > 180 (the forge flames' yellow-white cores), 960x402 decode; its flame tracked on the base; first = first frame under 90% of its median area over the shot's first 60 frames, out = first frame under 10% (the flame's core leaves the mask; the renderer keeps a 6% flame); selected first | high |
| surge | 2440 | {"first":2483,"fastest":2486,"half":2487,"full":2490} | 2483 | +43 **moves the music** | trap.forges_surge: sum of flame pixels: luma > 150 and R > 180 (the forge flames' yellow-white cores), 960x402 decode (Ring blobs excluded) per frame; first/full = 10%/90% of the way from its median over frames 0-79 to its median over 190-225; fastest = largest one-frame rise; the same ramp on the three largest steady single flames (evidence.each) shows the surge is simultaneous; selected first | high |
| flares_back | 2480 | {"first":2546,"full":2557} | 2546 | +66 **moves the music** | trap.low_forge_returns: flame pixels: luma > 150 and R > 180 (the forge flames' yellow-white cores), 960x402 decode; the longest new track within 20 px (x) of where the low flame went out; full = first frame at >= 90% of its median area 12-30 frames after it reappears; selected first | high |
| pulls_ahead | 2560 | {"first":2588,"half":2607,"full":2626} | 2588 | +28 **moves the music** | trap.leader_left_pulls_ahead: flame pixels: luma > 150 and R > 180 (the forge flames' yellow-white cores), 960x402 decode; per frame the flame with the highest top within x 727-931 (1920 px) of the Ring; first/full = 10%/90% of its top's climb from the median over frames 200-239 to the last 8 frames' median; half = its 50% crossing (a one-frame 'fastest' is not reported: the tallest flame can switch to a neighbour's blob for a frame); selected first; earlier of left/right first frames | high |
| leaders_reach | — | {"first":2588,"half":2607,"full":2626} | 2625 | new id | trap.leader_left_pulls_ahead: flame pixels: luma > 150 and R > 180 (the forge flames' yellow-white cores), 960x402 decode; per frame the flame with the highest top within x 727-931 (1920 px) of the Ring; first/full = 10%/90% of its top's climb from the median over frames 200-239 to the last 8 frames' median; half = its 50% crossing (a one-frame 'fastest' is not reported: the tallest flame can switch to a neighbour's blob for a frame); selected full; rounded mean of left/right full frames (inherited placement) | high |
| flint_black | 2640 | — | 2640 | +0 | No black in 240 examined FLINT frames (maximum-luma threshold <=2); retain musical cut at 2640 without claiming pictured black. | not measured |
| catch | 2836 | — | 2836 | +0 | Flame-area detector fails its pre-catch control on glowing tinder/smoke; onset, fastest and full unresolved. SOUND 2836 remains derived. | not measured |
| reveal | 2880 | {"frame":2880,"near_glow_peak":2880,"far_glow_peak":2880} | 2880 | +0 | reveal.both_fires_ignite: warm wash: R - B > 80 and R > 170 (1920x804) counted in a 220x170 px window around each fire; the flame glyph's ink outline: luma < 120 in the near window; the ignition frame = the frame of each window's largest warm wash (the catch glow flashes, then settles); selected frame | high |
| promise | 2960 | — | 2960 | +0 | Caption timing, excluded from picture-only measurements. | not measured |
| run | 3120 | — | 3120 | +0 | Section cut into Beacon Run; separate catches are measured. | not measured |
| beacon_1 | 3160 | {"first":3160,"fastest":null,"full":null} | 3160 | +0 | run.beacon_1: warm pixels R-B>80,R>170; register preceding mask using phase correlation of landscape rows220-389; warp then dilate by 3px; a new connected warm component >=50px is a catch. First is the arrival of new warm area. Fastest/full are null: registered glow novelty is not flame-size growth.; selected first | medium |
| beacon_2 | 3200 | {"first":3200,"fastest":null,"full":null} | 3200 | +0 | run.beacon_2: warm pixels R-B>80,R>170; register preceding mask using phase correlation of landscape rows220-389; warp then dilate by 3px; a new connected warm component >=50px is a catch. First is the arrival of new warm area. Fastest/full are null: registered glow novelty is not flame-size growth.; selected first | medium |
| beacon_3 | 3240 | {"first":3240,"fastest":null,"full":null} | 3240 | +0 | run.beacon_3: warm pixels R-B>80,R>170; register preceding mask using phase correlation of landscape rows220-389; warp then dilate by 3px; a new connected warm component >=50px is a catch. First is the arrival of new warm area. Fastest/full are null: registered glow novelty is not flame-size growth.; selected first | medium |
| beacon_4 | 3280 | {"first":3280,"fastest":null,"full":null} | 3280 | +0 | run.beacon_4: warm pixels R-B>80,R>170; register preceding mask using phase correlation of landscape rows220-389; warp then dilate by 3px; a new connected warm component >=50px is a catch. First is the arrival of new warm area. Fastest/full are null: registered glow novelty is not flame-size growth.; selected first | medium |
| beacon_5 | 3320 | {"first":3320,"fastest":null,"full":null} | 3320 | +0 | run.beacon_5: warm pixels R-B>80,R>170; register preceding mask using phase correlation of landscape rows220-389; warp then dilate by 3px; a new connected warm component >=50px is a catch. First is the arrival of new warm area. Fastest/full are null: registered glow novelty is not flame-size growth.; selected first | medium |
| beacon_6 | 3360 | {"first":3360,"fastest":null,"full":null} | 3360 | +0 | run.beacon_6: warm pixels R-B>80,R>170; register preceding mask using phase correlation of landscape rows220-389; warp then dilate by 3px; a new connected warm component >=50px is a catch. First is the arrival of new warm area. Fastest/full are null: registered glow novelty is not flame-size growth.; selected first | medium |
| beacon_7 | 3400 | {"first":3400,"fastest":null,"full":null} | 3400 | +0 | run.beacon_7: warm pixels R-B>80,R>170; register preceding mask using phase correlation of landscape rows220-389; warp then dilate by 3px; a new connected warm component >=50px is a catch. First is the arrival of new warm area. Fastest/full are null: registered glow novelty is not flame-size growth.; selected first | medium |
| map | 3440 | — | 3440 | +0 | Section cut to map; separate catches already measured. | not measured |
| map_beacon_2 | — | {"first":3468,"fastest":3471,"full":3472,"lit_through":3839} | 3468 | new id | map.beacon_2: bright warm local-contrast blobs (luma minus its 12 px Gaussian > 40, R > G > B), native 1920x804, tracked (40 px, >= 20 frames); first = first frame with a blob of >= 4 px, fastest = largest one-frame area rise in its first 12 frames, full = first frame at >= 90% of its median area 10-40 frames after it appears; selected first | high |
| map_beacon_3 | — | {"first":3523,"fastest":3525,"full":3527,"lit_through":3839} | 3523 | new id | map.beacon_3: bright warm local-contrast blobs (luma minus its 12 px Gaussian > 40, R > G > B), native 1920x804, tracked (40 px, >= 20 frames); first = first frame with a blob of >= 4 px, fastest = largest one-frame area rise in its first 12 frames, full = first frame at >= 90% of its median area 10-40 frames after it appears; selected first | high |
| map_beacon_4 | — | {"first":3584,"fastest":3588,"full":3588,"lit_through":3839} | 3584 | new id | map.beacon_4: bright warm local-contrast blobs (luma minus its 12 px Gaussian > 40, R > G > B), native 1920x804, tracked (40 px, >= 20 frames); first = first frame with a blob of >= 4 px, fastest = largest one-frame area rise in its first 12 frames, full = first frame at >= 90% of its median area 10-40 frames after it appears; selected first | high |
| one_dark | 3600 | {"first":3722,"last":3785,"seventh_first_visible":3718} | 3722 | +122 **moves the music** | map.one_dark: from the 2nd-to-last beacon's full frame to the frame before the last beacon's first visible frame (bright warm local-contrast blobs (luma minus its 12 px Gaussian > 40, R > G > B), native 1920x804, tracked (40 px, >= 20 frames); first = first frame with a blob of >= 4 px, fastest = largest one-frame area rise in its first 12 frames, full = first frame at >= 90% of its median area 10-40 frames after it appears); selected first | high |
| map_beacon_5 | — | {"first":3647,"fastest":3650,"full":3651,"lit_through":3839} | 3647 | new id | map.beacon_5: bright warm local-contrast blobs (luma minus its 12 px Gaussian > 40, R > G > B), native 1920x804, tracked (40 px, >= 20 frames); first = first frame with a blob of >= 4 px, fastest = largest one-frame area rise in its first 12 frames, full = first frame at >= 90% of its median area 10-40 frames after it appears; selected first | high |
| map_beacon_6 | — | {"first":3680,"fastest":3683,"full":3684,"lit_through":3839} | 3680 | new id | map.beacon_6: bright warm local-contrast blobs (luma minus its 12 px Gaussian > 40, R > G > B), native 1920x804, tracked (40 px, >= 20 frames); first = first frame with a blob of >= 4 px, fastest = largest one-frame area rise in its first 12 frames, full = first frame at >= 90% of its median area 10-40 frames after it appears; selected first | high |
| hammer_alone | 3700 | — | 3742 | +42 **moves the music** | one_dark + 20 frames | musical derivation |
| map_beacon_7 | — | {"first":3718,"fastest":3722,"full":3722,"lit_through":3839} | 3718 | new id | map.beacon_7: bright warm local-contrast blobs (luma minus its 12 px Gaussian > 40, R > G > B), native 1920x804, tracked (40 px, >= 20 frames); first = first frame with a blob of >= 4 px, fastest = largest one-frame area rise in its first 12 frames, full = first frame at >= 90% of its median area 10-40 frames after it appears; selected first | high |
| last_beacon | 3760 | {"first":3786,"fastest":3789,"full":3791} | 3786 | +26 **moves the music** | map.last_catch: as map.beacon_k, for the last beacon to appear; selected first | high |
| all_lit | — | {"first":3786,"fastest":3789,"full":3791} | 3791 | new id | map.last_catch: as map.beacon_k, for the last beacon to appear; selected full | high |
| forges_cold | 3848 | {"frame":3848} | 3848 | +0 | cold.forges_off: sum of flame pixels: luma > 150 and R > 180 (the forge flames' yellow-white cores), 960x402 decode per frame (the Ring's gold included, so the floor is the Ring); the frame of the largest one-frame drop; accepted only if the drop exceeds half the median area C3840-3847; selected frame | high |
| smoke | 3920 | — | 3920 | +0 | Smoke continues from shutdown; no discrete onset. | not measured |
| ring_unfinished | 4000 | — | 4000 | +0 | Section cut; separate gold-drain and storm-thinning ramps measured. | not measured |
| ring_grey | — | {"first":4025,"half":4072,"full":4135} | 4135 | new id | unfinished.ring_drains: mean R - B of the Ring's bright pixels (luma > 110 in x 840-1080, y 120-360 of 1920x804); 10%/50%/90% of the way from frame 4000 to the last 20 frames' median; selected full | high |
| storm_gone | 4160 | {"first":4087,"half":4138,"full":4181} | 4181 | +21 **moves the music** | unfinished.storm_thins: mean R - B over the top 400 rows (1920x804); 10%/50%/90% of the way from frame 4000 to the last 20 frames' median; selected full | high |
| deep_still | 4240 | — | 4240 | +0 | Section cut to held mine drawing; no discrete within-shot event. | not measured |
| watch | 4480 | — | 4480 | +0 | Section cut into already-burning beacons; no catch in the shot. | not measured |
| sunrise | 4720 | — | 4720 | +0 | Sun already partly above ridge at C4720; no onset within delivered 480-frame interval. Preserve section entry. | not measured |
| home | 5040 | — | 5040 | +0 | Musical resolution, not a discrete sunrise event. | not measured |
| plenty | 5200 | — | 5200 | +0 | Section/hymn entry; camera pulls from landscape to illustrated book page continuously. | not measured |
| havens | 5440 | — | 5440 | +0 | Section cut; incoming page-turn transition absent from delivered PEN frames. | not measured |
| line_in | 5460 | — | 5460 | +0 | Optional spoken-line placement, not a picture event. | not measured |
| blank | 5540 | — | 5540 | +0 | Musical return after spoken line; existing PEN frames have no discrete ink event. | not measured |
| title | 5680 | {"first":5702} | 5702 | +22 **moves the music** | title.first_light: count R>150 and luma-Gaussian(sigma=3)>12 in the title line; excludes the book gutter; first = first sample >= 3; selected first | medium |
| plagal | 5840 | — | 5840 | +0 | Musical cadence, not a picture event. | not measured |

## FLINT cross-check against SOUND’s derived mapping

| Event | SOUND derived | Pixel first | Delta | Result |
|---|---:|---:|---:|---|
| strike1 | 2650 | 2650 | 0 | Agrees at the measured spark onset; peak is recorded separately. |
| strike3 | 2698 | 2698 | 0 | Agrees at the measured spark onset; peak is recorded separately. |
| blow | 2724 | unresolved | — | The candidate also fires on the strike flash when nobody is visibly blowing. The mouth is outside this framing; no unique visible onset was isolated. Do not infer it from smoke brightness. |
| catch | 2836 | unresolved | — | The candidate also fires on glowing tinder/smoke before the distinct flame. No measured catch frame; retain pass 1 pending a reliable flame-shape boundary. |

Black detector: 0 black frames / 240 examined; minimum frame-maximum luma 196.670990, threshold ≤2. SOUND’s references to a cut ‘to black’ disagree with the delivered picture. Unresolved blow/catch are unverified, not measured disagreements.

## Negative controls

Each row applies the stated detector threshold to a delivered interval without the target event. Counts include the denominator beside every zero. Failed controls are retained. A clean interval validates only that interval and threshold, not universal specificity. Ramp controls use the event’s fixed baseline/top, never a fresh fit to the control. Inherited argmax/peak controls are identified explicitly; these narrower checks must not be confused with a general semantic-event classifier.

| Event / detector | Negative interval | Triggers / frames examined | Threshold / direction | Interpretation |
|---|---|---|---|---|
| refusal.ink_begins / ink | 2251–2319 | 0 / 69 | 10; direction 1 | completed drawing, no more ink strokes |
| refusal.offering_hand / ink | 2251–2319 | 0 / 69 | 10; direction 1 | completed drawing, no more ink strokes |
| refusal.ring_on_palm / ink | 2251–2319 | 0 / 69 | 10; direction 1 | completed drawing, no more ink strokes |
| refusal.pause_before_figure / ink | 2251–2319 | 0 / 69 | 10; direction 1 | completed drawing, no more ink strokes |
| refusal.figure_begins / ink | 2251–2319 | 0 / 69 | 10; direction 1 | completed drawing, no more ink strokes |
| refusal.raised_hand / ink | 2251–2319 | 0 / 69 | 10; direction 1 | completed drawing, no more ink strokes |
| refusal.ink_ends / ink | 2251–2319 | 0 / 69 | 10; direction 1 | completed drawing, no more ink strokes |
| trap.low_forge_sinks / signal | 2320–2379 | 0 / 60 | -15.25; direction -1 | no extinction crossing below 10% in the opening plateau; the 90% onset is assigned only before that crossing |
| trap.low_forge_sinks / signal | 2320–2379 | 9 / 60 | -137.25; direction -1 | TRIGGERS; see interpretation. 90% alone also detects ordinary flicker; the full sink rule additionally requires a subsequent extinction crossing |
| trap.low_forge_returns / signal | 2500–2530 | 0 / 31 | 20; direction 1 | fixed pre-event control; original detector threshold |
| trap.forges_surge / signal | 2320–2399 | 0 / 80 | 2287.9; direction 1 | fixed pre-event control; original detector threshold |
| trap.leader_left_pulls_ahead / signal | 2520–2559 | 0 / 40 | -202.05; direction -1 | fixed pre-event control; original detector threshold |
| trap.leader_right_pulls_ahead / signal | 2520–2559 | 0 / 40 | -199.6; direction -1 | fixed pre-event control; original detector threshold |
| trap.hammer_strikes / signal | 2321–2399 | 0 / 79 | 0.5; direction 1 | fixed pre-event control; original detector threshold |
| reveal.both_fires_ignite / wash | 3000–3119 | 0 / 120 | 9756.0; direction 1 | settled already-burning fire; fixed 90% peak wash |
| reveal.both_fires_ignite / wash | 3000–3119 | 0 / 120 | 1342.8; direction 1 | settled already-burning fire; fixed 90% peak wash |
| reveal.near_flame_grows / ink | 2880–2880 | 0 / 1 | 866.15; direction 1 | only one available pre-growth frame; limited negative-control coverage |
| map.beacon_1 / births | 3792–3839 | 0 / 48 | 1; direction 1 | all eight established fires, no new catch |
| map.beacon_2 / births | 3792–3839 | 0 / 48 | 1; direction 1 | all eight established fires, no new catch |
| map.beacon_3 / births | 3792–3839 | 0 / 48 | 1; direction 1 | all eight established fires, no new catch |
| map.beacon_4 / births | 3792–3839 | 0 / 48 | 1; direction 1 | all eight established fires, no new catch |
| map.beacon_5 / births | 3792–3839 | 0 / 48 | 1; direction 1 | all eight established fires, no new catch |
| map.beacon_6 / births | 3792–3839 | 0 / 48 | 1; direction 1 | all eight established fires, no new catch |
| map.beacon_7 / births | 3792–3839 | 0 / 48 | 1; direction 1 | all eight established fires, no new catch |
| map.beacon_8 / births | 3792–3839 | 0 / 48 | 1; direction 1 | all eight established fires, no new catch |
| map.one_dark / births | 3792–3839 | 0 / 48 | 1; direction 1 | all eight established fires, no new catch |
| map.last_catch / births | 3792–3839 | 0 / 48 | 1; direction 1 | all eight established fires, no new catch |
| map.all_lit / births | 3792–3839 | 0 / 48 | 1; direction 1 | all eight established fires, no new catch |
| cold.forges_off / signal | 3860–3999 | 0 / 140 | 3887.0; direction 1 | post-shutdown Ring and cold forges; same absolute drop threshold |
| unfinished.ring_drains / signal | 4000–4010 | 0 / 11 | -94.38228149414063; direction -1 | pre-event interval, same fixed threshold |
| unfinished.storm_thins / signal | 4000–4010 | 0 / 11 | -9.369588760379703; direction -1 | pre-event interval, same fixed threshold |
| unfinished.still_hold / same | 4200–4216 | 0 / 17 | 1; direction 1 | before the final held frame, adjacent files must still differ |
| deep.no_discrete_event / signal | 4241–4479 | 0 / 239 | 1.1132685393095016; direction 1 | held illustration, all frame differences examined |
| watch.beacons_burn_on / births | 4480–4719 | 2 / 240 | 1; direction 1 | TRIGGERS; see interpretation. already-burning beacons across the full shot; occlusion remains a possible false positive |
| watch.beacons_burn_on / births | 4480–4600 | 0 / 121 | 1; direction 1 | opening already-burning landscape, before either later ridge-uncovering candidate |
| pen.no_discrete_event / signal | 5441–5679 | 0 / 239 | 1.3657788038253784; direction 1 | held illustration, all frame differences examined |
| opening.riffle / delta | 380–399 | 0 / 20 | 10; direction 1 | inspected interval without this event |
| letters.page_turn / delta | 650–679 | 0 / 30 | 10; direction 1 | inspected interval without this event |
| letters.glow / glow | 680–699 | 0 / 20 | 3123.65; direction 1 | inspected pre-event interval; same fixed 10% presence threshold as the event |
| letters.lift / lift | 730–734 | 0 / 5 | 717.45; direction 1 | inspected pre-event interval; same fixed 10% presence threshold as the event |
| letters.catch / fire | 797–800 | 0 / 4 | 100; direction 1 | inspected interval without this event |
| letters.burn / hole | 820–839 | 0 / 20 | 14400.35; direction 1 | inspected pre-event interval; same fixed 10% presence threshold as the event |
| forge.towers / towers | 1040–1059 | 0 / 20 | 2380.1; direction 1 | inspected pre-event interval; same fixed 10% presence threshold as the event |
| forge.inscription / inscription | 1310–1319 | 0 / 10 | 120; direction 1 | inspected interval without this event |
| forge.ring_rises / body_y | 1378–1384 | 0 / 7 | -268.3226946083655; direction -1 | inspected pre-event interval; same fixed 10% presence threshold as the event |
| forge.anvil_test / anvil impulse | 1041–1199 | 0 / 159 | dY > 3 * 15-frame running median + 0.5; direction 1 | the opening tower emergence, with no pictured anvil contact |
| eye.burn / hole | 1905–1920 | 0 / 16 | 5364.950000000001; direction 1 | inspected pre-event interval; same fixed 10% presence threshold as the event |
| eye.slit / slit | 1992–2000 | 0 / 9 | 400.8; direction 1 | inspected pre-event interval; same fixed 10% presence threshold as the event |
| flint.strike1 / flash | 2640–2649 | 0 / 10 | 1000; direction 1 | inspected interval without this event |
| flint.strike3 / flash | 2670–2697 | 0 / 28 | 1000; direction 1 | inspected interval without this event |
| flint.black_test / max_luma | 2640–2649 | 0 / 10 | -2; direction -1 | inspected hands and blue sky at the opening |
| flint.catch_test / flame | 2828–2835 | 8 / 8 | 100; direction 1 | TRIGGERS; see interpretation. inspected interval without this event |
| flint.blow_test / plume | 2698–2703 | 6 / 6 | 20; direction 1 | TRIGGERS; see interpretation. inspected interval without this event |
| run.beacon_1 / novel_area | 3121–3159 | 0 / 39 | 50; direction 1 | opening landscape with no burning beacon |
| run.beacon_1 / novel_area | 3208–3231 | 0 / 24 | 50; direction 1 | camera and already-burning beacons; no new catch |
| run.beacon_2 / novel_area | 3121–3159 | 0 / 39 | 50; direction 1 | opening landscape with no burning beacon |
| run.beacon_2 / novel_area | 3208–3231 | 0 / 24 | 50; direction 1 | camera and already-burning beacons; no new catch |
| run.beacon_3 / novel_area | 3121–3159 | 0 / 39 | 50; direction 1 | opening landscape with no burning beacon |
| run.beacon_3 / novel_area | 3208–3231 | 0 / 24 | 50; direction 1 | camera and already-burning beacons; no new catch |
| run.beacon_4 / novel_area | 3121–3159 | 0 / 39 | 50; direction 1 | opening landscape with no burning beacon |
| run.beacon_4 / novel_area | 3208–3231 | 0 / 24 | 50; direction 1 | camera and already-burning beacons; no new catch |
| run.beacon_5 / novel_area | 3121–3159 | 0 / 39 | 50; direction 1 | opening landscape with no burning beacon |
| run.beacon_5 / novel_area | 3208–3231 | 0 / 24 | 50; direction 1 | camera and already-burning beacons; no new catch |
| run.beacon_6 / novel_area | 3121–3159 | 0 / 39 | 50; direction 1 | opening landscape with no burning beacon |
| run.beacon_6 / novel_area | 3208–3231 | 0 / 24 | 50; direction 1 | camera and already-burning beacons; no new catch |
| run.beacon_7 / novel_area | 3121–3159 | 0 / 39 | 50; direction 1 | opening landscape with no burning beacon |
| run.beacon_7 / novel_area | 3208–3231 | 0 / 24 | 50; direction 1 | camera and already-burning beacons; no new catch |
| title.first_light / glow | 5680–5700 | 0 / 21 | 3; direction 1 | inspected interval without this event |

The illumination and PLENTY records are inspected coverage notes, with no accepted event onset; they are not successful detectors and have no claimed negative-control pass. The anvil impulse candidate at C1262 was visually rejected (adjacent stills show a rotating Ring). Exact numeric traces are in the `*_trace.json` files; accepted events carry their deciding samples and fixed thresholds in `events_C5_measured.json`.

The inherited low-forge sink requires a subsequent <10% extinction crossing before assigning its preceding 90% onset; the 90% threshold alone also trips on baseline flicker (retained as a diagnostic control). WATCH produces two occlusion candidates across its full shot, so that tracker cannot establish ignition semantics; the opening interval supplies a separate clean control. Neither observation is hidden by a zero from a shorter interval.

## Verification and close-out

- Coverage audit: all 5,920 cut frames exactly once across 18 measured ranges; 60 event records and all 72 sync ids. New scalar traces reproduce every added event decision (audit.json).
- Full sequential pixel rescan: measure_c5_events.py --trace-dir ../../review/codex-events-c5; peak RSS 291,913,728 bytes against a 1,500,000,000-byte guard (measure.log). Original nine source-frame hashes remain unchanged.
- TRAP and WATCH were remeasured after their control annotations were refined; no event frames changed in that refinement (remeasure.log). The final metadata-only correction explicitly labels cut coordinates; all pixel decisions were retained.
- Original HEAD measurer independently returns right-tower half-rise 2603, versus saved 2601. First/full stay 2591/2624; no selected score cue uses the corrected half frame (baseline_trap.json). Cause of the older saved value is not established.
- barmap_c5p2.py regeneration: 0 errors, 0 warnings; --check exits 0 (barmap.log). cues_C5P2.json regenerated byte-identically.
- score_v3_C5P2.py: 54 parts, 1,167 notes, 49 score sync points, 0 rule warnings (score.log); no audio rendered.
- Full prescribed validation-venv pytest run over music plus edit/tests: 152 passed, 6 subtests passed, 4 Pillow/Raqm fallback warnings (tests.log).
- After the frame-coordinate metadata wording was corrected, focused schema/freshness/failed-control/sound-snapshot checks passed (metadata-checks.log); bar-map and score checks repeated cleanly.
- The initial suite exposed an obsolete same-times comparison and stale sound input hashes. The revised comparison feeds both shared compositions the same updated picture timings; the original pass-1 digest test still passes unchanged. sound_c5_table.py --check passes, and its generated event rows are unchanged; only two input hashes changed.
- Structured close-out audit: verified source mapping/composition, pixel/trace consistency, negative controls including failures, every sync row and delta, selected score timing consumption, scope, and diff whitespace. No EDL, renderer, pass-1 score, pass-1 barmap, or effects-event changes.

Unresolved picture items remain explicitly at pass 1. The FLINT flame/smoke candidates fail controls; sunrise onset precedes the delivered interval; anvil contact and a separate small Ring drawing stroke were not established. Other retained cues are musical/caption/section placements or continuous motion without an isolated event. Threshold sensitivity and left-censoring are stated above. No continuous-motion or audio judgment is claimed.

Commit blocked: git add could not create the shared repository’s .git/worktrees/codex-events/index.lock (Operation not permitted). No staging or commit succeeded; all deliverables remain uncommitted in this worktree. No push was attempted.

## Owner follow-up

Review the moved cues against picture and sound before a new mix. FLINT catch/blow still need a reliable visible boundary or an explicit musical placement; the retained SOUND frames remain derived. SOUND’s existing “not on this Mac” verification text and black-cut descriptions are stale; this lane refreshes its source hashes for snapshot consistency but does not redesign effects. No EDL/source-selection edits, frame renders, audio renders, pushes, or merges were performed.
