# Codex handover 3: from the director (Claude), 28 Sep 2026 ~01:20Z

**Status of your work:**
- PR #3 (terrain-ceiling audit) is merged. It found nothing affected except A15, so no re-renders are needed.
- PR #4 (legacy run_job exit) is merged. Both test suites pass on the production Mac.
- If your frame-QC tool (task 3) and your A cloud-sea relief study (task 4) are still in progress, finish them first. Task 4 feeds the job below.

**What has changed since handover 2:**
- The user **dropped film B**. Only A (with its coded-towers ALT) and C remain.
- The user found the procedural **people movement** "awful and glitchy". No figure may slide, jitter, pop or walk with a robotic cycle. Prefer still or slow silhouettes and small figures.
- A's clarity is being reworked by a lane called A-FIX, in bars 1-49 and A's ending. Don't touch those.

## Your new lane: A's night shots, A13-A15 (you take over RUN-A5, which is retired)

You already know these shots: you fixed A14's camera and A15's ceiling, and you ran the cairn studies.

Read these blocks in `the-long-dawn/shots/run/NOTES.md`:
- **RUN-A5 STATE (00:05Z).** It has the diagnosis and the plan. The key points:
  - the fire's 3 m depth bias paints the flame over the front cairn stones and over any hood within 3 m;
  - the flat-albedo sdfppl stones read as potatoes or cubes;
  - the lit rock step behind the fire reads as clay;
  - the smoke ambient is darker than the sky;
  - a single glow shadow ray reads as a searchlight;
  - the plan is a `hearth_a.py` with its own SDF tracer.
- **RUN-A3's** block and **RUN-A-L's** block, for A13's pending re-render and A14's issues.

The shots and what they need:
- **A14, the BEACON RUN** (`beaconrun_a.py`, A 3920-4239):
  - the glide fires read too small;
  - the catch flare is weak;
  - the seventh fire's cairn reads as stacked cubes;
  - the lighter is lost in a dark pillar.
- **A15, the WATCHERS** (`watchers_a.py`, A 4240-4399):
  - the cairn reads as a lit pile of smooth lumps;
  - the ledge reads as clay.

  The user **rejected all three of your geometry cairns**. Use RUN-A5's hearth plan (a low battered dry-stone ring of flattened, chipped, pitted stones, with per-stone albedo, soot, joint and ground occlusion, and bounce), or RUN-A3's staging fix (let the lighter's body hide most of the cairn, and pitch the plate up so the ledge leaves the frame). The result must read as real weathered stone at a glance.
- **A13, the REVEAL** (`reveal_a.py`): re-render with the cloud-sea relief (your task 4, once the director approves it) and the haze-slab fix (nighta 2913ad7).

  Note: A-FIX may replace part of A13 (A 3612-3759) with the dropped film B's reveal shot. Check `the-long-dawn/review/A_FIX_NOTES.md` before rendering A13, and don't re-render frames A-FIX has replaced.

**Constraints:**
- These are the gates. Look at every test yourself, at reduced size for composition and at 1:1 crops for detail:
  - Can a stranger tell the story?
  - Nothing evokes a real place, company or country.
  - Nothing would get a laugh.
  - Nothing reads as a doll, clip-art, a screensaver or a tech demo.
  - Silhouettes and gloved hands only, never a lit face. The red shawl.
- Render tests locally on the M4 at 960×402, or at full resolution for single frames.
- **You cannot launch farm finals.** Deliver a PR that contains:
  - the code;
  - the job JSON (the format of `cloud/jobs/watchers_a_1.json` and `reveal_a_1.json`);
  - 3-4 check stills;
  - a 24-frame filmstrip for any moving figure.

  The director reviews the PR, merges it and launches the final.
- Don't touch `farm.py`, EDIT's files, the sound code, or A-FIX's shots (A 0-3919 and A19/A20).

Order: A15, then A14, then A13. One PR per shot is fine.
