"""v2 additions to the VSCO-2-CE sampler (sampler.py itself is unchanged, so the
v1 first half renders exactly as composed).  Importing this module registers the
extra patches used by the v2 second halves.

Organ (Ivy Audio / VSCO-2-CE):
  organ      OrganLoud      manual, full principal chorus ("Man3Open"), sounds at key pitch
  organ_q    OrganQuiet     manual, soft stopped flute ("Man3Quiet"), key pitch
  organ_ped  OrganLoudPedal loud pedal: 16' fundamental, sounds an octave BELOW the key
                            (written D2 -> D1 + D2 + upper ranks; measured, see NOTES_v2.md)
  organ_pedq OrganQuietPedal quiet pedal flue, sounds at key pitch
"""
import sampler as S

S.patch("organ_pedq", "OrganQuietPedal", rel=0.7, antic=0.02)
S.patch("oboe_nv", "OboeSusNV", rel=0.25, antic=0.03)
S.patch("trombone_vib", "TromboneVib", rel=0.35, antic=0.035, lskip=0.08)
S.patch("svln_q", "SViolinVib-Quiet", rel=0.4, antic=0.05, lskip=0.10)
S.patch("vln_q", "ViolinEnsSusVib-Quiet", rel=0.6, antic=0.06, lskip=0.10, lfade=0.09)
S.patch("vla_q", "ViolaEnsSusVib-Quiet", rel=0.6, antic=0.06, lskip=0.10, lfade=0.09)
S.patch("vc_q", "CelloEnsSusVib-Quiet", rel=0.6, antic=0.06, lskip=0.10, lfade=0.09)
S.patch("cb_q", "ContrabassSusVB-Quiet", rel=0.6, antic=0.06, lskip=0.10)
S.patch("bassoon_v2", "BassoonSus", rel=0.3, antic=0.03)
S.patch("clarinet_v2", "ClarinetSus", rel=0.3, antic=0.03)
