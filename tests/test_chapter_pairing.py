from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from kindlecomicconverter.chapter_pairing import (
    analyze_chapters, pair_geometry, resize_method, sample_edges, score_edges,
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

    def test_first_match_parity_and_early_stop(self):
        for anchor_index in (0, 1, 2, 3):
            with self.subTest(anchor_index=anchor_index):
                pages = self.pages(7)
                with (patch('kindlecomicconverter.chapter_pairing.sample_edges', return_value=()) as sample,
                     patch('kindlecomicconverter.chapter_pairing.score_edges',
                           side_effect=[None] * anchor_index + [12.5]) as score):
                    report = analyze_chapters([('chapter', pages)], [100, 200], 'rtl', True)
                chapter = report['chapters'][0]
                self.assertEqual(chapter['spreadshift'], bool(anchor_index % 2))
                self.assertEqual(chapter['anchor'], dict(source_ids=[anchor_index + 1, anchor_index + 2], score=12.5))
                self.assertEqual(sample.call_count, anchor_index + 2)
                self.assertEqual(score.call_count, anchor_index + 1)
                expected = [[1], [2, 3], [4, 5], [6, 7]] if anchor_index % 2 else [[1, 2], [3, 4], [5, 6], [7]]
                self.assertEqual([g['source_ids'] for g in report['groups']], expected)

    def test_no_candidate_uses_fallback_with_single_edges(self):
        for shift, expected in ((False, [[1, 2], [3]]), (True, [[1], [2, 3]])):
            report = analyze_chapters([('nested/chapter', self.pages(3))], [100, 200], 'ltr', shift)
            self.assertEqual(report['chapters'][0]['reason'], 'spreadshift')
            self.assertIsNone(report['chapters'][0]['anchor'])
            self.assertEqual([g['source_ids'] for g in report['groups']], expected)
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
            self.assertEqual(report['chapters'][0]['anchor'], dict(source_ids=[1, 2], score=0))
            self.assertFalse(report['chapters'][0]['spreadshift'])

    def test_last_wide_page_forces_whole_chapter_native_without_sampling(self):
        with patch('kindlecomicconverter.chapter_pairing.sample_edges') as sample:
            report = analyze_chapters([('wide', self.pages(5, wide_at=4))], [100, 200], 'rtl', True)
        sample.assert_not_called()
        self.assertEqual(report['chapters'][0]['reason'], 'wide_image')
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
                                 [150, 400], 'rtl')
        self.assertEqual([g['crop_box'] for g in geometry], [[4, 29, 97, 271], [7, 49, 114, 451]])
        self.assertEqual([g['rendered_size'][1] for g in geometry], [390, 390])
        self.assertEqual(geometry[0]['offset'], [0, 5])
        self.assertEqual(geometry[1]['offset'], [47, 5])

    def test_no_upscale_uses_smallest_cropped_height(self):
        for upscale, expected in ((False, 100), (True, 300)):
            geometry = pair_geometry([(50, 100), (100, 200)], [(0, 0, 50, 100), (0, 0, 100, 200)],
                                     [200, 300], 'ltr', upscale)
            self.assertEqual([g['rendered_size'] for g in geometry], [[expected // 2, expected]] * 2)
            self.assertEqual(geometry[0]['offset'][0], 200 - expected // 2)
            self.assertEqual(geometry[1]['offset'][0], 0)

    def test_exact_fraction_boundaries_and_full_source_candidates(self):
        geometry = pair_geometry([(100, 300), (100, 300)], [(0, 29, 100, 271)] * 2, [100, 300], 'rtl')
        self.assertEqual(geometry[0]['crop_box'], [0, 29, 100, 271])
        self.assertEqual(geometry[0]['rendered_size'], [100, 242])
        full = pair_geometry([(100, 200)] * 2, [(0, 0, 100, 200)] * 2, [100, 200], 'rtl')
        self.assertEqual(full[0], dict(crop_box=[0, 0, 100, 200], rendered_size=[100, 200], offset=[0, 0]))

    def test_minimum_width_and_resize_filter(self):
        geometry = pair_geometry([(1, 1000)] * 2, [(0, 0, 1, 1000)] * 2, [100, 100], 'rtl')
        self.assertEqual(geometry[0]['rendered_size'], [1, 100])
        self.assertEqual(resize_method([0, 0, 1, 1000], [1, 100]), Image.Resampling.LANCZOS)
        self.assertEqual(resize_method([0, 0, 100, 200], [100, 200]), Image.Resampling.BICUBIC)
        self.assertEqual(resize_method([0, 0, 100, 200], [200, 400]), Image.Resampling.BICUBIC)

    def test_bad_crop_and_unrenderable_target_rejected(self):
        for box in ((-1, 0, 10, 20), (0, 0, 11, 20), (0, 10, 10, 10)):
            with self.assertRaises(ValueError):
                pair_geometry([(10, 20)] * 2, [box] * 2, [10, 20], 'rtl')
        with self.assertRaises(ValueError):
            pair_geometry([(100, 1)] * 2, [(0, 0, 100, 1)] * 2, [1, 1], 'rtl')


if __name__ == '__main__':
    unittest.main()
