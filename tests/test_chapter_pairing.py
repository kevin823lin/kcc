from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from kindlecomicconverter.chapter_pairing import (
    analyze_chapters, pair_geometry, resize_method, sample_edges, sample_strips,
    score_edge_profiles, score_edges, score_strips,
)


def patterned(low=60, high=180):
    return [(low, low, low) if i % 2 else (high, high, high) for i in range(50)]


class EdgeTests(unittest.TestCase):
    def test_exact_sampling_coordinates_and_narrow_sources(self):
        image = Image.new('RGB', (7, 103))
        for y in range(103):
            image.putpixel((2, y), (y, 1, 2))
            image.putpixel((4, y), (y, 3, 4))
        left, right = sample_edges(image)
        self.assertEqual(left, [(i * 103 // 50, 1, 2) for i in range(50)])
        self.assertEqual(right, [(i * 103 // 50, 3, 4) for i in range(50)])
        self.assertEqual(sample_edges(Image.new('RGB', (1, 2), 'red'))[0], [(255, 0, 0)] * 50)

    def test_alpha_composition_and_palette_transparency(self):
        self.assertEqual(sample_edges(Image.new('RGBA', (3, 50), (0, 0, 0, 128)))[0],
                         [(127, 127, 127)] * 50)
        palette = Image.new('P', (3, 50), 0)
        palette.info['transparency'] = 0
        self.assertEqual(sample_edges(palette)[0], [(255, 255, 255)] * 50)

    def test_16_bit_matches_native_eight_bit_conversion(self):
        image = Image.new('I;16', (3, 50), 32768)
        self.assertEqual(sample_edges(image)[0], [(128, 128, 128)] * 50)

    def test_directions_and_strict_distance_threshold(self):
        edge = patterned()
        shifted = [tuple(c + 10 for c in rgb) for rgb in edge]
        other = patterned(0, 255)
        self.assertEqual(score_edges((edge, other), (other, edge), 'rtl'), 0)
        self.assertEqual(score_edges((other, edge), (edge, other), 'ltr'), 0)
        self.assertIsNone(score_edges((edge, edge), (shifted, shifted), 'rtl'))
        shifted = [tuple(c + 9 for c in rgb) for rgb in edge]
        self.assertEqual(score_edges((edge, edge), (shifted, shifted), 'rtl'), 27)

    def test_blank_and_low_variance_rejections(self):
        for edge in ([(255, 255, 255)] * 40 + [(0, 0, 0)] * 10,
                     [(0, 0, 0)] * 40 + [(255, 255, 255)] * 10,
                     patterned(100, 101)):
            with self.subTest(edge=edge[0]):
                self.assertIsNone(score_edges((edge, edge), (edge, edge), 'rtl'))
        for edge in (patterned(50, 80), patterned(200, 220)):
            self.assertEqual(score_edges((edge, edge), (edge, edge), 'rtl'), 0)

    def test_low_variance_only_rejects_when_both_are_flat(self):
        flat = [(120, 120, 120)] * 50
        varied = [(120, 102, 120) if i % 2 else (120, 138, 120) for i in range(50)]
        self.assertEqual(score_edges((flat, flat), (varied, varied), 'rtl'), 18)

    def test_invalid_direction_and_sample_count(self):
        with self.assertRaises(ValueError):
            score_edges(([], []), ([], []), 'vertical')
        with self.assertRaises(ValueError):
            score_edges(([], []), ([], []), 'rtl')

    def test_strip_signature_ignores_shared_flat_borders_and_checks_signal(self):
        signal = [80 if i % 2 else 170 for i in range(50)]
        shifted = [value + 8 for value in signal]
        self.assertAlmostEqual(score_strips((signal, signal), (shifted, shifted), 'rtl'), 1)
        border_only = [signal[i] if 20 <= i < 30 else 255 for i in range(50)]
        self.assertIsNone(score_strips((border_only, border_only), (border_only, border_only), 'rtl'))
        self.assertIsNone(score_strips((signal, signal), ([value + 17 for value in signal],) * 2, 'rtl'))
        flat = [120] * 50
        self.assertIsNone(score_strips((flat, flat), (flat, flat), 'rtl'))
        for shade in (230, 40):
            tinted_margin = [shade] * 40 + [100, 180] * 5
            self.assertIsNone(score_strips((tinted_margin,) * 2, (tinted_margin,) * 2, 'rtl'))
        mixed_flat = [120 if i < 20 else 255 for i in range(50)]
        self.assertIsNone(score_strips((mixed_flat, mixed_flat), (mixed_flat, mixed_flat), 'rtl'))
        inverted = [255] * 10 + [80 if i % 2 else 170 for i in range(30)] + [255] * 10
        unrelated = [255] * 10 + [170 if i % 2 else 80 for i in range(30)] + [255] * 10
        self.assertIsNone(score_strips((inverted, inverted), (unrelated, unrelated), 'rtl'))

    def test_strip_sampling_uses_both_inner_eight_pixel_strips(self):
        image = Image.new('RGB', (20, 50), 'white')
        for y in range(50):
            image.putpixel((0, y), (y, y, y))
            image.putpixel((19, y), (255 - y, 255 - y, 255 - y))
        left, right = sample_strips(image)
        self.assertEqual(left[0], round((7 * 255 + 0) / 8))
        self.assertEqual(right[0], 255)

    def test_textured_seam_can_match_despite_local_halftone_differences(self):
        # The shared white margin contributes to the full-page score. The art still
        # has to correlate independently even with alternating dark/light offsets.
        edge = [255] * 30 + [100, 150, 100, 150] * 5
        textured = [255] * 30 + [82, 132, 118, 168] * 5
        score = score_strips((edge, edge), (textured, textured), 'rtl')
        self.assertGreaterEqual(score, .93)

    def test_strip_sampling_scales_16_bit_and_composites_transparency(self):
        for width in (1, 8):
            self.assertEqual(sample_strips(Image.new('I;16', (3, 50), 32768), width=width)[0], [128] * 50)
            self.assertEqual(sample_strips(Image.new('RGBA', (3, 50), (0, 0, 0, 128)), width=width)[0], [127] * 50)

    def test_edge_profiles_require_separated_corroborating_contours(self):
        profile = [0] * 20 + [160] + [0] * 26 + [40, 160, 150]
        shifted = [value + 3 for value in profile]
        blank = [255] * 50
        self.assertIsNone(score_strips((profile,) * 2, (shifted,) * 2, 'rtl', fallback=True))
        for direction, earlier, later in (
                ('rtl', (profile, blank), (blank, shifted)),
                ('ltr', (blank, profile), (shifted, blank))):
            self.assertAlmostEqual(score_edge_profiles(earlier, later, direction), 1)
        for other in (blank, list(range(0, 250, 5)), [0] * 25 + [160] * 25,
                      [0] * 20 + [40, 160] + [160] * 5 + [40, 0] + [0] * 21,
                      [0] * 20 + [40, 160, 40, 160, 0] + [0] * 25):
            self.assertIsNone(score_edge_profiles((other,) * 2, (other,) * 2, 'rtl'))
        self.assertIsNone(score_edge_profiles((profile,) * 2,
                                             ([value + 17 for value in profile],) * 2, 'rtl'))
        # Smooth profiles may correlate while their small contours disagree.
        ramp = [30 + 3 * i + (12 if i % 2 else -12) for i in range(50)]
        jittered = [30 + 3 * i + (12 if i % 3 else -12) for i in range(50)]
        self.assertIsNone(score_edge_profiles((ramp,) * 2, (jittered,) * 2, 'rtl'))
        for direction in ('rtl', 'vertical'):
            with self.assertRaises(ValueError):
                score_edge_profiles(([], []), ([], []), direction)

    def test_fallback_accepts_noisy_seams_but_keeps_signal_and_distance_guards(self):
        signal = [80 if i % 2 else 170 for i in range(50)]
        noisy = [value + (35, -35, 0)[i % 3] for i, value in enumerate(signal)]
        for direction in ('rtl', 'ltr'):
            self.assertIsNone(score_strips((signal,) * 2, (noisy,) * 2, direction))
            self.assertGreaterEqual(score_strips((signal,) * 2, (noisy,) * 2, direction, fallback=True), .82)
        for other in ([value + 26 for value in signal], signal[::-1], [120] * 50):
            self.assertIsNone(score_strips((signal,) * 2, (other,) * 2, 'rtl', fallback=True))
        for signal in ([255] * 40 + [80, 170] * 5, [0] * 40 + [80, 170] * 5,
                       [120] * 50, [255] * 36 + [80, 170] * 7):
            self.assertIsNone(score_strips((signal,) * 2, (signal,) * 2, 'rtl', fallback=True))


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def pages(self, count, start=1, wide_at=None):
        result = []
        for index in range(count):
            path = self.root / f'{start + index}.png'
            Image.new('RGB', (100, 100) if index == wide_at else (50, 100), 'white').save(path)
            result.append((start + index, path))
        return result

    def analyze_matches(self, pages, matches, direction='rtl', shift=True):
        scored = []

        def samples(image, **kwargs):
            return int(Path(image.filename).stem)

        def score(method, earlier, later, actual_direction):
            self.assertEqual(actual_direction, direction)
            scored.append((method, earlier, later))
            return matches.get((method, earlier, later))

        with (patch('kindlecomicconverter.chapter_pairing.sample_edges', side_effect=samples) as sample,
              patch('kindlecomicconverter.chapter_pairing.sample_strips', side_effect=samples),
              patch('kindlecomicconverter.chapter_pairing.score_edges',
                    side_effect=lambda a, b, d: score('rgb', a, b, d)),
              patch('kindlecomicconverter.chapter_pairing.score_strips',
                    side_effect=lambda a, b, d, fallback=False:
                    score('strip_fallback' if fallback else 'strip', a, b, d)),
              patch('kindlecomicconverter.chapter_pairing.score_edge_profiles',
                    side_effect=lambda a, b, d: score('edge_profile', a, b, d))):
            report = analyze_chapters([('chapter', pages)], [100, 200], direction, shift)
        return report, scored, sample.call_count

    def test_body_first_match_parity_and_early_stop(self):
        for anchor_index in (1, 2, 3, 4):
            with self.subTest(anchor_index=anchor_index):
                report, scored, samples = self.analyze_matches(
                    self.pages(7), {('rgb', anchor_index + 1, anchor_index + 2): 12.5})
                chapter = report['chapters'][0]
                self.assertEqual(chapter['spreadshift'], bool(anchor_index % 2))
                self.assertIsNone(chapter['opening_anchor'])
                self.assertEqual(chapter['anchor'], dict(source_ids=[anchor_index + 1, anchor_index + 2], score=12.5))
                self.assertEqual(samples, anchor_index + 3)  # p2 is sampled again for the independent body.
                self.assertEqual(sum(method == 'rgb' for method, _, _ in scored), anchor_index + 1)
                expected = [[1], [2, 3], [4, 5], [6, 7]] if anchor_index % 2 else [[1, 2], [3, 4], [5, 6], [7]]
                self.assertEqual([g['source_ids'] for g in report['groups']], expected)

    def test_opening_and_body_use_all_tiers_independently_without_scoring_2_3(self):
        methods = ('rgb', 'strip', 'strip_fallback', 'edge_profile')
        for direction in ('rtl', 'ltr'):
            for opening in methods:
                for body in methods:
                    with self.subTest(direction=direction, opening=opening, body=body):
                        report, scored, _ = self.analyze_matches(
                            self.pages(7, start=11),
                            {(opening, 11, 12): .97, (body, 14, 15): .98,
                             ('rgb', 12, 13): 0}, direction)
                        chapter = report['chapters'][0]
                        self.assertEqual(chapter['opening_anchor']['source_ids'], [11, 12])
                        self.assertEqual(chapter['opening_anchor'].get('method', 'rgb'), opening)
                        self.assertEqual(chapter['anchor']['source_ids'], [14, 15])
                        self.assertEqual(chapter['anchor'].get('method', 'rgb'), body)
                        self.assertFalse(any((a, b) == (12, 13) for _, a, b in scored))
                        self.assertEqual(chapter['reason'], 'detected')
                        self.assertTrue(chapter['spreadshift'])
                        self.assertEqual([g['source_ids'] for g in report['groups']],
                                         [[11, 12], [13], [14, 15], [16, 17]])

    def test_no_opening_keeps_2_3_eligible_in_all_tiers(self):
        for direction in ('rtl', 'ltr'):
            for method in ('rgb', 'strip', 'strip_fallback', 'edge_profile'):
                with self.subTest(direction=direction, method=method):
                    report, _, _ = self.analyze_matches(
                        self.pages(5, start=20), {(method, 21, 22): .98}, direction, False)
                    chapter = report['chapters'][0]
                    self.assertIsNone(chapter['opening_anchor'])
                    self.assertEqual(chapter['anchor']['source_ids'], [21, 22])
                    self.assertEqual(chapter['anchor'].get('method', 'rgb'), method)
                    self.assertTrue(chapter['spreadshift'])
                    self.assertEqual([g['source_ids'] for g in report['groups']],
                                     [[20], [21, 22], [23, 24]])

    def test_opening_with_body_3_4_keeps_global_even_parity(self):
        for direction in ('rtl', 'ltr'):
            report, _, _ = self.analyze_matches(
                self.pages(6), {('edge_profile', 1, 2): .98, ('rgb', 3, 4): 0}, direction)
            chapter = report['chapters'][0]
            self.assertEqual(chapter['anchor']['source_ids'], [3, 4])
            self.assertFalse(chapter['spreadshift'])
            self.assertEqual([group['source_ids'] for group in report['groups']], [[1, 2], [3, 4], [5, 6]])

    def test_opening_only_overrides_config_and_keeps_normal_body(self):
        for shift in (False, True):
            for method in ('rgb', 'strip', 'strip_fallback', 'edge_profile'):
                with self.subTest(shift=shift, method=method):
                    report, _, _ = self.analyze_matches(
                        self.pages(5), {(method, 1, 2): .98}, shift=shift)
                    chapter = report['chapters'][0]
                    self.assertEqual(chapter['reason'], 'opening')
                    self.assertIsNone(chapter['anchor'])
                    self.assertFalse(chapter['spreadshift'])
                    self.assertEqual([g['source_ids'] for g in report['groups']], [[1, 2], [3, 4], [5]])

    def test_one_and_two_page_chapters_use_opening_or_config_without_body(self):
        for count in (1, 2):
            for matched in (False, True):
                for shift in (False, True):
                    with self.subTest(count=count, matched=matched, shift=shift):
                        report, scored, _ = self.analyze_matches(
                            self.pages(count), {('rgb', 1, 2): 0} if matched else {}, shift=shift)
                        chapter = report['chapters'][0]
                        has_opening = matched and count == 2
                        self.assertIsNone(chapter['anchor'])
                        self.assertEqual(chapter['reason'], 'opening' if has_opening else 'spreadshift')
                        self.assertEqual(chapter['spreadshift'], shift and not has_opening)
                        self.assertEqual(chapter['opening_anchor'] is not None, has_opening)
                        expected = [[1]] if count == 1 else [[1, 2]] if has_opening or not shift else [[1], [2]]
                        self.assertEqual([g['source_ids'] for g in report['groups']], expected)
                        self.assertEqual(sum(method == 'rgb' for method, _, _ in scored), count - 1)

    def test_no_candidate_uses_fallback_with_single_edges(self):
        for shift, expected in ((False, [[1, 2], [3]]), (True, [[1], [2, 3]])):
            report = analyze_chapters([('nested/chapter', self.pages(3))], [100, 200], 'ltr', shift)
            self.assertEqual(report['chapters'][0]['reason'], 'spreadshift')
            self.assertIsNone(report['chapters'][0]['anchor'])
            self.assertEqual([g['source_ids'] for g in report['groups']], expected)
            self.assertEqual(report['version'], 3)
            self.assertEqual(report['state'], 'analyzing')
            self.assertEqual(report['pages'], [])

    def test_real_pixels_detect_both_reading_directions(self):
        for direction in ('rtl', 'ltr'):
            pages = self.pages(3)
            first_x, second_x = (2, 47) if direction == 'rtl' else (47, 2)
            for (_, path), x in zip(pages[:2], (first_x, second_x)):
                with Image.open(path) as source:
                    image = source.copy()
                for y in range(100):
                    value = 60 if y // 2 % 2 else 180
                    image.putpixel((x, y), (value, value, value))
                image.save(path)
            report = analyze_chapters([('chapter', pages)], [100, 200], direction, True)
            self.assertEqual(report['chapters'][0]['opening_anchor'], dict(source_ids=[1, 2], score=0))
            self.assertFalse(report['chapters'][0]['spreadshift'])

    def test_real_inner_strips_detect_both_directions_when_edges_differ(self):
        for direction in ('rtl', 'ltr'):
            pages = self.pages(3)
            edge_columns = ((0, 7), (42, 49))
            first_side, second_side = edge_columns if direction == 'rtl' else edge_columns[::-1]
            for page, side, offset in ((pages[0], first_side, 0), (pages[1], second_side, 12)):
                _, path = page
                with Image.open(path) as source:
                    image = source.copy()
                for y in range(100):
                    value = (60 if y // 2 % 2 else 180) + offset
                    for x in range(side[0], side[1] + 1):
                        image.putpixel((x, y), (value, value, value))
                image.save(path)
            report = analyze_chapters([('chapter', pages)], [100, 200], direction, True)
            anchor = report['chapters'][0]['opening_anchor']
            self.assertEqual(anchor['source_ids'], [1, 2])
            self.assertEqual(anchor['method'], 'strip')
            self.assertEqual(set(anchor), {'source_ids', 'method', 'score'})
            self.assertGreaterEqual(anchor['score'], .93)
            self.assertFalse(report['chapters'][0]['spreadshift'])

    def test_body_later_stronger_tier_overrides_earlier_candidates(self):
        for winner, matches, expected_pair in (
                ('rgb', {('edge_profile', 2, 3): .98, ('strip_fallback', 3, 4): .85,
                         ('strip', 4, 5): .95, ('rgb', 5, 6): 12.5}, [5, 6]),
                ('strip', {('edge_profile', 2, 3): .98, ('strip_fallback', 3, 4): .85,
                           ('strip', 4, 5): .95}, [4, 5]),
                ('strip_fallback', {('edge_profile', 2, 3): .98,
                                    ('strip_fallback', 3, 4): .85}, [3, 4]),
                ('edge_profile', {('edge_profile', 2, 3): .98,
                                  ('edge_profile', 3, 4): .99}, [2, 3])):
            with self.subTest(winner=winner):
                report, _, _ = self.analyze_matches(self.pages(7), matches)
                chapter = report['chapters'][0]
                self.assertIsNone(chapter['opening_anchor'])
                self.assertEqual(chapter['anchor'].get('method', 'rgb'), winner)
                self.assertEqual(chapter['anchor']['source_ids'], expected_pair)
                self.assertEqual(chapter['spreadshift'], bool((expected_pair[0] - 1) % 2))

    def test_real_outermost_profiles_detect_narrow_contours_in_both_directions(self):
        profile = [0] * 20 + [160] + [0] * 26 + [40, 160, 150]
        for direction in ('rtl', 'ltr'):
            pages = self.pages(4)
            columns = (0, 49) if direction == 'rtl' else (49, 0)
            for (_, path), x in zip(pages[1:3], columns):
                with Image.open(path) as source:
                    image = source.copy()
                for y in range(100):
                    image.putpixel((x, y), (profile[y // 2],) * 3)
                image.save(path)
            chapter = analyze_chapters([('chapter', pages)], [100, 200], direction)['chapters'][0]
            self.assertEqual(chapter['anchor'], dict(source_ids=[2, 3], score=1, method='edge_profile'))
            self.assertTrue(chapter['spreadshift'])

    def test_last_wide_page_forces_whole_chapter_native_without_sampling(self):
        with patch('kindlecomicconverter.chapter_pairing.sample_edges') as sample:
            report = analyze_chapters([('wide', self.pages(5, wide_at=4))], [100, 200], 'rtl', True)
        sample.assert_not_called()
        self.assertEqual(report['chapters'][0]['reason'], 'wide_image')
        self.assertIsNone(report['chapters'][0]['opening_anchor'])
        self.assertIsNone(report['chapters'][0]['anchor'])
        self.assertTrue(report['chapters'][0]['spreadshift'])
        self.assertEqual([g['mode'] for g in report['groups']], ['native'] * 5)

    def test_chapters_and_source_ids_sort_numerically_with_unique_global_groups(self):
        pages = self.pages(4, start=9)
        report = analyze_chapters([('second', pages[2:][::-1]), ('first', pages[:2][::-1])], [100, 200], 'rtl')
        self.assertEqual([c['source_ids'] for c in report['chapters']], [[9, 10], [11, 12]])
        self.assertEqual([c['id'] for c in report['chapters']], [1, 2])
        self.assertEqual([g['id'] for g in report['groups']], [1, 2])
        self.assertEqual([g['chapter_id'] for g in report['groups']], [1, 2])

    def test_single_page_never_scores_or_crosses_chapter_boundary(self):
        pages = self.pages(2)
        with patch('kindlecomicconverter.chapter_pairing.score_edges') as score:
            report = analyze_chapters([('a', pages[:1]), ('b', pages[1:])], [100, 200], 'rtl')
        score.assert_not_called()
        self.assertEqual([g['mode'] for g in report['groups']], ['single', 'single'])

    def test_duplicate_and_interleaved_source_ranges_rejected(self):
        pages = self.pages(4)
        for chapters in ([('a', pages[:2]), ('b', pages[1:])],
                         [('a', pages[::2]), ('b', pages[1::2])], [('a', [])]):
            with self.assertRaises(ValueError):
                analyze_chapters(chapters, [100, 200], 'rtl')


class GeometryTests(unittest.TestCase):
    def test_normalized_vertical_union_and_preserved_inner_and_outer_sides(self):
        geometry = pair_geometry([(100, 300), (120, 500)],
                                 [(4.5, 31, 96.1, 270), (7.1, 49, 113.2, 451)],
                                 [150, 400])
        self.assertEqual([g['crop_box'] for g in geometry], [[4, 29, 97, 271], [7, 49, 114, 451]])
        self.assertEqual([g['rendered_size'][1] for g in geometry], [390, 390])
        self.assertEqual([g['rendered_size'][0] for g in geometry], [149, 103])
        self.assertTrue(all(set(g) == {'crop_box', 'rendered_size'} for g in geometry))

    def test_no_upscale_uses_smallest_cropped_height(self):
        for upscale, expected in ((False, 100), (True, 300)):
            geometry = pair_geometry([(50, 100), (100, 200)], [(0, 0, 50, 100), (0, 0, 100, 200)],
                                     [200, 300], upscale)
            self.assertEqual([g['rendered_size'] for g in geometry], [[expected // 2, expected]] * 2)

    def test_exact_fraction_boundaries_and_full_source_candidates(self):
        geometry = pair_geometry([(100, 300), (100, 300)], [(0, 29, 100, 271)] * 2, [100, 300])
        self.assertEqual(geometry[0]['crop_box'], [0, 29, 100, 271])
        self.assertEqual(geometry[0]['rendered_size'], [100, 242])
        full = pair_geometry([(100, 200)] * 2, [(0, 0, 100, 200)] * 2, [100, 200])
        self.assertEqual(full[0], dict(crop_box=[0, 0, 100, 200], rendered_size=[100, 200]))

    def test_minimum_width_and_resize_filter(self):
        geometry = pair_geometry([(1, 1000)] * 2, [(0, 0, 1, 1000)] * 2, [100, 100])
        self.assertEqual(geometry[0]['rendered_size'], [1, 100])
        self.assertEqual(resize_method([0, 0, 1, 1000], [1, 100]), Image.Resampling.LANCZOS)
        self.assertEqual(resize_method([0, 0, 100, 200], [100, 200]), Image.Resampling.BICUBIC)
        self.assertEqual(resize_method([0, 0, 100, 200], [200, 400]), Image.Resampling.BICUBIC)

    def test_bad_crop_and_unrenderable_target_rejected(self):
        for box in ((-1, 0, 10, 20), (0, 0, 11, 20), (0, 10, 10, 10)):
            with self.assertRaises(ValueError):
                pair_geometry([(10, 20)] * 2, [box] * 2, [10, 20])
        with self.assertRaises(ValueError):
            pair_geometry([(100, 1)] * 2, [(0, 0, 100, 1)] * 2, [1, 1])


if __name__ == '__main__':
    unittest.main()
