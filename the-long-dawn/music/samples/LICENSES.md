# music/samples: sources and licences (THE LONG DAWN)

The libraries in this folder stay out of git (`the-long-dawn/.gitignore` excludes `music/samples/`). This one file is force-added so the record travels with the repo. It uses about 3.3 GB of disk: VSCO 3.3 GB, sfizz 8 MB, maps 1 MB.

| What | Where | Source | Licence | Commercial use |
|---|---|---|---|---|
| **VSCO 2 Community Edition** (Versilian Studios; recorded by Sam Gossner and Simon Dalzell, cut by Elan Hickler / Soundemote) | `VSCO-2-CE/` | Sparse clone of https://github.com/sgossner/VSCO-2-CE, commit `4403009` (4 Aug 2020, release line 1.1.0). See `VSCO-2-CE/LICENSE` and `Readme.txt`. | **CC0 1.0 Universal** (public-domain dedication) | Yes, without restriction or attribution. Credit is optional; the film credits it as a courtesy. |
| **VSCO 2 CE SFZ maps** | `sfz/` | The `.sfz` files from the same repo's `SFZ` branch | CC0 1.0 (as above) | Yes |
| **sfizz 1.2.3**: `sfizz_render` and `libsfizz` (the SFZ player that renders the A/B's B side and any all-samples render) | `tools/sfizz-1.2.3/` | Official release asset `sfizz-1.2.3-macos.tar.gz` from https://github.com/sfztools/sfizz/releases/tag/1.2.3. It is an x86_64 build and runs under Rosetta 2. Nothing is compiled locally. | **BSD-2-Clause** | Yes. It is a tool: rendered audio carries no licence obligation, and the binary is not redistributed. |

How it is used:

- `src/sampler.py` is our own numpy SFZ player. It plays VSCO for every orchestral part of scores A, B and C.
- `src/sfizz_v3.py` (COMPOSER-C2) plays the same VSCO regions through sfizz_render, through SFZs it derives into `music/cache/sfizz/sfz/`. It also maps C's 11 synthesised colour voices onto VSCO's real recordings: the anvil, gong, suspended-cymbal crescendos, glockenspiel, tubular bells, bass drum, harp and the quiet violin section.

The real sound effects (Freesound, Sonniss GDC, ElevenLabs) belong to SOUND-C's lane, and their credits are in `music/SFX_CREDITS.md`.

To re-create:

- VSCO: `git clone --filter=blob:none --sparse https://github.com/sgossner/VSCO-2-CE`, then `git sparse-checkout set` on the folders `sampler.py` uses.
- sfizz: `gh release download 1.2.3 -R sfztools/sfizz -p sfizz-1.2.3-macos.tar.gz`, then copy `usr/local` to `tools/sfizz-1.2.3/`.
