# THE LONG DAWN: credits for tools and data (for the end crawl, the about page and the breakdowns)

## Film emulation

**On screen (end crawl, technical credits):**

> Film print emulation powered by spektrafilm, by Andrea Volpato

**In the about page, the press notes and every breakdown:**

> The films' photographic finish (Kodak Vision3 negative printed to 2383, grain and halation) uses spectral film
> modeling powered by spektrafilm by Andrea Volpato, https://github.com/andreavolpato/spektrafilm (v0.3.3).
> spektrafilm LUTs by Andrea Volpato, licensed CC BY-SA 4.0.

**What we use, and under which terms:**

- The LUTs (film exposure, negative development, print and scan) are baked by spektrafilm's own LUT creator
  (`finish/bake_luts.sh`) and used unmodified. They are CC BY-SA 4.0 with the author's attribution terms
  (`SPEKTRAFILM_LICENSE.txt`, shipped inside every baked bundle): name Andrea Volpato, link the repository, keep the
  licence notice with any copy of the files. They are not resold and never shipped as a product.
- Our own finishing code (`finish/filmfinish.py`, `finish/filmfast.py`) applies those LUTs and adds halation and grain
  that follow spektrafilm's published models. It does not include spektrafilm's code, but because it follows its
  methods it is licensed GPL-3.0-or-later, as the author asks of derivative work.
- spektrafilm itself (GPLv3) runs only in its own environment (`~/.venvs/finish`) to bake the LUTs.
- Factual reference only: "spektrafilm" is not used as branding.
