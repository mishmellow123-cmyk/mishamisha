# Independent final-delivery sample audits

The final JPEGs now have independent, bounded visual audits in addition to the full-delivery numerical checks in the existing handback. All three sample audits inspected native first/middle/end frames, reduced composition and a 24-frame consecutive window. Each selected RGB image and matching matte was compared with its original delivery receipt. The renderer remains frozen at `ebcdd0a0b51691a74f3635f7a64c55d6546a2aa0`.

| Shot | Distinct RGB / matte sample pairs | Consecutive window | Report and evidence |
|---|---:|---|---|
| Refusal | 33 / 240 | C2192–2215 | [Report](refusal/audit.md), [hash evidence](refusal/sample-verification.json), [strip](refusal/refusal-consecutive-2192-2215-480.png) |
| Deep Abandoned | 30 / 240 | C4352–4375 | [Report](deep/audit.md), [hash evidence](deep/sample-verification.json), [strip](deep/deep-consecutive-4352-4375-480.png) |
| Pen | 26 / 240 | C5550–5573 | [Report](pen/REPORT.md), [hash evidence](pen/evidence.json), [native nib strip](pen/motion-nib-native-5550-5573.png) |

The reports preserve the limits of the observations. Refusal's turned-away pose is revealed by drawing; it does not perform a separate body-turn animation. Deep retains visible gold and empty galleries, but at 480px the lantern can read as a tiny building and ladder rungs become difficult to distinguish; the audit does not establish at-a-glance abandonment at that size. Pen contact and shadow hold in the samples; its ferrule and nib split are subdued at 480px, where silhouette and book context carry the object.

Complete-sequence visual review, full-speed playback, audience comprehension, captions, sound and EDIT joins remain unapproved by these sample audits. Pen is the permitted blank-spread insert and does not supply the incoming page turn. The full numerical checks and these visual samples address different questions; neither extends the other's denominator.

These are public copies of local audit reports/evidence/images. Private absolute paths are replaced by `<validation-root>` or `~/ldfarm`, and Pen's handback link is adjusted for this directory. Helper scripts are excluded. Each shot's `PUBLICATION.json` records original/public file hashes and transformations; evidence JSON also labels its publication metadata. Original delivery-receipt SHA256 and all sample/source hashes are preserved. Images are copied without byte changes. Raw report/evidence/image mirrors are at `~/ldfarm/comms/files/C5_{refusal,deep,pen}_final_audit/`; the pre-existing Refusal files were preserved after byte comparison.

No renderer, frame, matte, original delivery manifest, handback, EDIT or sound file changed in this publication.
