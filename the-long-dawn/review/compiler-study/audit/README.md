# Candidate evidence audit

Two read-only audit lanes checked the candidate suite at `dc82b0d`; neither launched a renderer or modified its inputs.

- [Harness and publication audit](final-harness-audit.json): examined the comparison code and independently checked the receipts, timing claims, source inventory, publication hashes and preview provenance. It covers five frames with four runs each and 17 SP case/frame groups with four runs each. All ten input records per SP call are included.
- [Saved-array audit](final-raw-array-audit.json): decoded all 122 NPY files, verified their file and array hashes, shape, dtype and finiteness, and compared all 61 cold/cache pairs byte for byte. The 12 material fixtures all affect their renders; the report records the material-switch, emission, displacement, fog and occlusion controls. The five finished PNGs also match the specified quantization of the saved arrays.

Both audits reported zero failures within those scopes. The raw-array audit covers the saved first pass of each case and phase; warm repeats have captured hashes but no separately saved NPY files. These records establish candidate repeatability and publication integrity. They do not establish original-versus-candidate equivalence or approve adoption.

[Packaging provenance](packaging.json) records original and published audit-file hashes. Only local path prefixes were normalized. Each report distinguishes direct measurements from source inspection and retained receipt values.
