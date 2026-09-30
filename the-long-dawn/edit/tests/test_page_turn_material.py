"""The folded leaf carries its source paper, including its edge, through the camera crop.

Synthetic plates only. Each check also runs a deliberately damaged result and requires that result to fail;
the controls exercise the assertion rather than relying on the compositor's previous implementation.
"""
import math
import inspect
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

EDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDIT))
import assemble as AS
import deliver as D
import edl_v3 as EDL


def paper_plate(h=192, w=384, deckled=True):
    """Two similarly bright paper colours distinguish the reflected leaf from the paper beneath it."""
    y, x = np.mgrid[:h, :w]
    right = 320 + (np.rint(6 * np.sin(2 * np.pi * np.arange(h) / 48)) if deckled else np.zeros(h))
    paper = (x >= 30) & (x <= right[:, None])
    image = np.full((h, w, 3), 0.03, np.float32)
    image[paper & (x < 225)] = [0.96, 0.63, 0.30]
    image[paper & (x >= 225)] = [0.66, 0.74, 0.58]
    incoming = np.full_like(image, 0.8)
    return image, incoming, right


def flat_back_coordinates(shape, p, tilt=0, radius=0.11):
    """Independent geometric oracle: a laid-back leaf reflects across the cylinder's half-circumference."""
    h, w = shape[:2]
    y, x = np.mgrid[:h, :w].astype(np.float32)
    a = math.radians(tilt)
    ca, sa = math.cos(a), math.sin(a)
    r = max(2.0, radius * w)
    u = x * ca + (y - h / 2) * sa
    fold = (float(u.max()) + 0.02 * r) * (1 - p) + (float(u.min()) - 1.2 * r) * p
    delta = 2 * fold + np.pi * r - 2 * u
    return x + delta * ca, y + delta * sa, u, fold


def changed_helper(o, i, p, tilt=8.0, radius=0.11):
    """Distinct source identity for the segment-cache test; no frames are rendered with it."""
    return i


class PageTurnMaterialTests(unittest.TestCase):
    def rejects(self, check, damaged):
        check(AS.page_turn)
        with self.assertRaises(AssertionError, msg='the negative control escaped the new check') as rejection:
            check(damaged)
        print(f'NEGATIVE CONTROL {self._testMethodName}/{damaged.__name__}: '
              f'{str(rejection.exception).strip().splitlines()[0]}')

    def test_reflected_silhouette_keeps_the_observed_deckle_and_excludes_room(self):
        outgoing, incoming, right = paper_plate()
        sx, _, u, fold = flat_back_coordinates(outgoing.shape, 0.5)
        # At zero tilt, the source x at destination x=0 is the reflection's intercept.
        edge = sx[:, 0] - right
        rows = np.arange(20, len(right) - 20)

        def check(turn):
            result = turn(outgoing, incoming, 0.5, tilt=0)
            # A green rise identifies pale source paper laid over the warmer stationary paper.
            leaf = result[rows, :int(fold) - 5, 1] > 0.665
            self.assertTrue(leaf.any(axis=1).all(), 'reflected leaf disappeared')
            measured = leaf.argmax(axis=1)
            np.testing.assert_allclose(measured, edge[rows], atol=1.5,
                                       err_msg='the reflected boundary lost its source silhouette')
            self.assertGreater(float(np.ptp(measured)), 8, 'deckle became a straight rectangle')

        def straight_sheet(o, i, p, **kw):
            straight, _, _ = paper_plate(deckled=False)
            return AS.page_turn(straight, i, p, **kw)

        self.rejects(check, straight_sheet)

    def test_camera_crop_does_not_become_a_horizontal_leaf_edge(self):
        outgoing, incoming, _ = paper_plate(h=128, deckled=False)
        sx, sy, u, fold = flat_back_coordinates(outgoing.shape, 0.5, tilt=20)
        beyond_crop = (sy > outgoing.shape[0] + 3) & (sx > 250) & (sx < 300) & (u < fold - 3)
        self.assertGreater(int(beyond_crop.sum()), 10, 'fixture never samples beyond the camera crop')

        def check(turn):
            result = turn(outgoing, incoming, 0.5, tilt=20)
            self.assertGreater(float(result[..., 1][beyond_crop].min()), 0.67,
                               'paper vanished when the reflected source crossed the camera crop')

        def clipped_to_crop(o, i, p, **kw):
            result = AS.page_turn(o, i, p, **kw)
            result[beyond_crop] = o[beyond_crop]
            return result

        self.rejects(check, clipped_to_crop)

    def test_paper_material_follows_a_sloping_silhouette_beyond_the_crop(self):
        h, w = 128, 384
        y, x = np.mgrid[:h, :w]
        outgoing = np.full((h, w, 3), 0.03, np.float32)
        outgoing[(x >= 30) & (x <= 120 + 1.2 * y)] = 0.8
        incoming = np.full_like(outgoing, 0.25)
        sx, sy, u, fold = flat_back_coordinates(outgoing.shape, 0.5, tilt=20)
        reflected_y = np.abs((sy + h - 1) % (2 * (h - 1)) - (h - 1))
        # These points are securely inside the continued sloping paper, yet the same x at the reflected
        # camera row lies in the room. Moving the coverage alone would paint that room onto the paper.
        continued = (sy > h + 20) & (sx < 120 + 1.2 * sy - 30) \
            & (sx > 120 + 1.2 * reflected_y + 60) & (u < fold - 5)
        self.assertGreater(int(continued.sum()), 10, 'fixture does not distinguish support from texture')

        def check(turn):
            result = turn(outgoing, incoming, 0.5, tilt=20)
            self.assertGreater(float(result[continued].min()), 0.72,
                               'the continued paper is filled with room sampled across the sloping edge')

        def room_on_paper(o, i, p, **kw):
            result = AS.page_turn(o, i, p, **kw)
            result[continued] = 0.03
            return result

        self.rejects(check, room_on_paper)

    def test_smooth_full_frame_lighting_is_not_mistaken_for_a_paper_boundary(self):
        h, w, p = 128, 384, 0.8
        y, x = np.mgrid[:h, :w]
        outgoing = np.repeat((0.35 + 0.55 * x / (w - 1))[..., None], 3, axis=2).astype(np.float32)
        incoming = np.full_like(outgoing, [0.1, 0.4, 0.9])
        sx, _, u, fold = flat_back_coordinates(outgoing.shape, p)
        opaque = (y > 20) & (y < h - 20) & (u < fold - 3)
        # A linear source ramp survives a symmetric material blur. Its reflected value distinguishes the
        # leaf from the much darker underlying part of the same ramp without depending on a segmentation mask.
        expected = 0.97 * (0.35 + 0.55 * sx[opaque] / (w - 1))

        def check(turn):
            result = turn(outgoing, incoming, p, tilt=0)
            np.testing.assert_allclose(result[..., 1][opaque], expected, atol=0.001, rtol=0,
                                       err_msg='lighting variation was interpreted as the edge of a leaf')

        def false_boundary(o, i, progress, **kw):
            result = AS.page_turn(o, i, progress, **kw)
            result[opaque] = o[opaque]
            return result

        self.rejects(check, false_boundary)

    def test_broad_paper_texture_is_carried_at_its_reflected_source_position(self):
        h, w, p = 192, 384, 0.62
        y, x = np.mgrid[:h, :w]
        source = np.full((h, w, 3), 0.03, np.float32)
        # This chroma variation has zero Rec.709 luma: a blurred luma field plus tiny high-pass fibres
        # cannot impersonate the paper's broad colour texture.
        wave = np.sin(2 * np.pi * x / 72).astype(np.float32)
        colour = np.array([0.78, 0.69, 0.52], np.float32)
        axis = np.array([-0.0722 / 0.2126, 0, 1], np.float32)
        material = colour + 0.06 * wave[..., None] * axis
        paper = (x >= 30) & (x <= 350)
        source[paper] = material[paper]
        incoming = np.full_like(source, 0.8)
        sx, _, u, fold = flat_back_coordinates(source.shape, p)
        sample = (y > 24) & (y < h - 24) & (sx > 260) & (sx < 330) & (u < fold - 5)
        expected = np.sin(2 * np.pi * sx[sample] / 72)
        expected -= expected.mean()

        def check(turn):
            result = turn(source, incoming, p, tilt=0)
            blue = result[..., 2][sample]
            blue = blue - blue.mean()
            gain = float(np.dot(blue, expected) / np.dot(expected, expected))
            self.assertGreater(gain, 0.025, 'broad source texture was flattened into a uniform backface')
            self.assertGreater(float(np.corrcoef(blue, expected)[0, 1]), 0.98,
                               'texture stayed in screen space instead of following the reflected paper')

        def flat_material(o, i, progress, **kw):
            result = AS.page_turn(o, i, progress, **kw)
            result[sample] = result[sample].mean(axis=0)
            return result

        self.rejects(check, flat_material)

    def test_opaque_backface_does_not_borrow_the_incoming_picture_colour(self):
        outgoing, incoming, _ = paper_plate(deckled=False)
        sx, _, u, fold = flat_back_coordinates(outgoing.shape, 0.5)
        opaque = (sx > 270) & (sx < 307) & (u < fold - 5)
        warm = np.full_like(incoming, [0.85, 0.25, 0.12])
        cool = np.full_like(incoming, [0.12, 0.35, 0.85])

        def check(turn):
            np.testing.assert_array_equal(turn(outgoing, warm, 0.5, tilt=0)[opaque],
                                          turn(outgoing, cool, 0.5, tilt=0)[opaque],
                                          err_msg='opaque paper depends on the scene behind it')

        def borrowed_colour(o, i, p, **kw):
            result = AS.page_turn(o, i, p, **kw)
            result[opaque] = 0.9 * result[opaque] + 0.1 * i[opaque]
            return result

        self.rejects(check, borrowed_colour)

    def test_contact_shadow_is_local_to_the_folded_paper_edge(self):
        h, w, right, edge, radius = 192, 480, 400, 160, 0.11
        outgoing = np.full((h, w, 3), 0.03, np.float32)
        outgoing[:, 40:right + 1] = 0.8
        incoming = np.full_like(outgoing, 0.8)
        r = radius * w
        fold = (edge + right - np.pi * r) / 2
        p = (w - 1 + 0.02 * r - fold) / (w - 1 + 1.22 * r)
        fit_x = np.arange(edge - 40, edge - 10)

        def check(turn):
            profile = turn(outgoing, incoming, p, tilt=0)[h // 2, :, 1]
            # The broad cast shadow is smooth over this short interval. Its extrapolated profile separates
            # the narrow dark contact from a generally darker scene or a wider cast shadow.
            # A cubic follows the cast shadow's changing curvature; a quadratic falsely attributes its
            # extrapolation error to contact. The sample seven pixels out is held out from this fit.
            broad = np.poly1d(np.polyfit(fit_x - edge, profile[fit_x], 3))
            self.assertGreater(float(broad(-1) - profile[edge - 1]), 0.01,
                               'there is no narrow contact shadow outside the paper')
            self.assertLess(abs(float(broad(-7) - profile[edge - 7])), 0.004,
                            'the contact darkening spreads into a broad band')

        def erased_contact(o, i, progress, **kw):
            result = AS.page_turn(o, i, progress, **kw)
            profile = result[h // 2, :, 1]
            broad = np.poly1d(np.polyfit(fit_x - edge, profile[fit_x], 3))
            result[:, edge - 1] = broad(-1)
            return result

        self.rejects(check, erased_contact)

        def broad_dark_band(o, i, progress, **kw):
            result = AS.page_turn(o, i, progress, **kw)
            result[:, edge - 9:edge - 4] *= 0.97
            return result

        self.rejects(check, broad_dark_band)

    def test_helper_edit_invalidates_only_c21_and_c22_segment_keys(self):
        code = {k: 'synthetic-code' for k in ('frame', 'text', 'slate', 'x2', 'title')}

        def plan(shot, cut, variant):
            return dict(kind='take', take=(shot.get('takes') or [None])[0],
                        have=shot['f1'] - shot['f0'], alt=False)

        def keys():
            with mock.patch.object(D, '_TRANS_CODE', []):
                return {s['sec']: D.segment_key('C', None, D.PROFILES['master'], n, s,
                                                plan(s, 'C', None), code, [])
                        for n, s in enumerate(EDL.EDL['C'])}

        def check():
            before = keys()
            with mock.patch.dict(AS.KIND_HELPERS, page_turn=changed_helper):
                after = keys()
            self.assertEqual({sec for sec in before if before[sec] != after[sec]}, {'C21', 'C22'})

        transition_code = AS.transition_code

        def omitted_helper(kind):
            return inspect.getsource(AS.TKINDS_PAIR[kind]) if kind == 'page_turn' else transition_code(kind)

        with mock.patch.object(AS, 'plan_shot', plan), \
                mock.patch.object(D, 'frame_sources', side_effect=lambda *a: []), \
                mock.patch.object(D, '_stat', return_value='synthetic-source'), \
                mock.patch.object(AS, 'transition_layers', return_value={}):
            check()
            with mock.patch.object(AS, 'transition_code', side_effect=omitted_helper):
                with self.assertRaises(AssertionError,
                                       msg='missing helper dependency escaped the cache check') as rejection:
                    check()
            print(f'NEGATIVE CONTROL {self._testMethodName}/omitted_helper: {rejection.exception}')


if __name__ == '__main__':
    unittest.main()
