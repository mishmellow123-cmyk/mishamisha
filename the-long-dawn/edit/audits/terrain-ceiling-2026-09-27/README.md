# Terrain-ceiling audit: partial evidence

All 16 explicitly requested cut-frame samples are complete. Additional bluehour/beacon/probe views and the A15 positive control remain pending; this is not the completed audit.

| Shot | Frame | Depth changes | Max finished RGB difference (8-bit) | Scope |
|---|---:|---:|---:|---|
| A2 | 80 | 0 | 0 | finished full frame |
| A2 | 300 | 0 | 0 | finished full frame |
| A2 | 380 | 0 | 0 | finished full frame |
| A2 | 500 | 0 | 0 | finished full frame |
| A2 | 559 | 0 | 0 | finished full frame |
| A11 | 3120 | 0 | 0 | finished full frame |
| A11 | 3359 | 0 | 0 | finished full frame |
| A18 | 4880 | 0 | 0 | finished full frame |
| A18 | 5100 | 0 | 0 | finished full frame |
| A18 | 5300 | 0 | 0 | finished full frame |
| A18 | 5480 | 0 | 0 | finished full frame |
| A18 | 5700 | 0 | 0 | finished full frame |
| A18 | 5839 | 0 | 0 | finished full frame |
| A13 | 3680 | 0 | 0 | world layer; reveal gain 1; diagnostic finish; foreground excluded |
| A13 | 3740 | 0 | 0 | world layer; reveal gain 1; diagnostic finish; foreground excluded |
| A13 | 3799 | 0 | 0 | world layer; reveal gain 1; diagnostic finish; foreground excluded |

These compare 700 m with 3000 m over every source row and column, at 960×402 output. A2/A11 also match at the accepted 2784.5 m bound. A13 is background-only with diagnostic finish; its pushed source is untouched. The JSON records source commits and shot hashes. This is a local reconstruction, not a check of production JPEGs, all frame numbers or all resolutions.

A2/A11 skyline caches reset per condition, and A11 static depth caches reset. A18 terrain is initialized before either condition. PNGs use deterministic quantization without production dither. No source ceiling fix is supported by these samples. The remaining tests and positive control must finish before the PR is marked ready.
