from pathlib import Path
from multiprocessing import Pool
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from kindlecomicconverter import comic2ebook as c2e
from kindlecomicconverter import image as image_module


class PairingProcessingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ('options', 'pairing_plan', 'pairing_output', 'GUI', 'workerPool', 'workerOutput'):
            if hasattr(c2e, name):
                self.addCleanup(setattr, c2e, name, getattr(c2e, name))
        c2e.pairing_plan = None
        c2e.GUI = None

    def options(self, *extra):
        return c2e.checkOptions(c2e.makeParser().parse_args([
            '-p', 'KoLC', '-f', 'EPUB', '--chapter-pairing', '--customwidth', '100',
            '--customheight', '160', '--cropping', '0', '--forcepng', '--noquantize',
            '--noautocontrast', '--gamma', '1', '--whiteborders', *extra]))

    def source(self, source_id, size, color=80, folder=None):
        parent = self.root if folder is None else self.root / folder
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / f'kcc-{source_id:04d}.png'
        Image.new('RGB', size, (color, color, color)).save(path)
        return path

    def pair_work(self, opt, sizes=((60, 100), (80, 200))):
        paths = [self.source(i, size, i * 80) for i, size in enumerate(sizes, 1)]
        entries = [[path.name, str(path.parent), opt, index == 0] for index, path in enumerate(paths)]
        return (entries, 7, [1, 2]), paths

    def test_cli_default_off_and_rejects_unsupported_profiles_formats_and_options(self):
        self.assertFalse(c2e.makeParser().parse_args([]).chapter_pairing)
        for args in (['-p', 'KV'], ['-p', 'Rmk1'], ['-f', 'CBZ'], ['-f', 'PDF'],
                     ['--noprocessing'], ['--onepagelandscape'], ['--ipc', '1'],
                     ['--webtoon'], ['--stretch'], ['--wallpaper'], ['--maximizestrips'],
                     ['--lightnovel'], ['--invertdirection'], ['--vertical4panel']):
            with self.subTest(args=args), self.assertRaises(UserWarning):
                self.options(*args)

    def test_real_pair_content_sizes_offsets_canvas_and_shared_downsample_filter(self):
        opt = self.options('--upscale', '--manga-style')
        work, paths = self.pair_work(opt)
        with patch.object(c2e, 'processImagePage', wraps=c2e.processImagePage) as process:
            result = c2e.imgPairProcessing(work)
        self.assertIsInstance(result, list, result)
        self.assertEqual([call.args[2] for call in process.call_args_list], [Image.Resampling.LANCZOS] * 2)
        self.assertEqual([page[0] for page in result], ['kcc-0001-kcc-x.png', 'kcc-0002-kcc-x.png'])
        self.assertEqual([page[2:4] for page in result], [[1, 7], [2, 7]])
        self.assertEqual([page[4]['rendered_size'] for page in result], [[96, 160], [64, 160]])
        self.assertEqual([page[4]['offset'] for page in result], [[0, 0], [36, 0]])
        for index, page in enumerate(result):
            self.assertFalse(paths[index].exists())
            with Image.open(self.root / page[0]) as output:
                self.assertEqual(output.size, (100, 160))
                self.assertEqual(output.getpixel((20 if index == 0 else 80, 80)), 80 * (index + 1))
                self.assertEqual(output.getpixel((99 if index == 0 else 0, 80)), 255)

    def test_no_upscale_and_ltr_keep_same_vertical_position(self):
        work, _ = self.pair_work(self.options())
        result = c2e.imgPairProcessing(work)
        self.assertIsInstance(result, list, result)
        self.assertEqual([page[4]['rendered_size'] for page in result], [[60, 100], [40, 100]])
        self.assertEqual([page[4]['offset'] for page in result], [[40, 30], [0, 30]])
        for index, page in enumerate(result):
            with Image.open(self.root / page[0]) as output:
                self.assertEqual(output.getpixel((50 if index == 0 else 20, 30)), 80 * (index + 1))
                self.assertEqual(output.getpixel((50, 29)), 255)

    def test_native_black_background_and_flags_survive_pair_processing(self):
        work, _ = self.pair_work(self.options('--blackborders'))
        result = c2e.imgPairProcessing(work)
        self.assertIsInstance(result, list, result)
        self.assertEqual([page[1] for page in result], [['BlackBackground']] * 2)
        with Image.open(self.root / result[0][0]) as output:
            self.assertEqual(output.getpixel((50, 29)), 0)

    def test_pair_uses_bicubic_when_neither_page_shrinks(self):
        work, _ = self.pair_work(self.options('--upscale'), sizes=((60, 100), (40, 100)))
        with patch.object(c2e, 'processImagePage', wraps=c2e.processImagePage) as process:
            result = c2e.imgPairProcessing(work)
        self.assertIsInstance(result, list, result)
        self.assertEqual([call.args[2] for call in process.call_args_list], [Image.Resampling.BICUBIC] * 2)

    def test_crop_modes_use_native_candidates_without_cropping_before_geometry(self):
        for mode in (0, 1, 2):
            with self.subTest(mode=mode):
                opt = self.options('--cropping', str(mode))
                path = self.source(1, (100, 200), 255)
                with Image.open(path) as source:
                    image = source.copy()
                ImageDraw.Draw(image).rectangle((10, 20, 89, 179), fill=(70, 70, 70))
                image.save(path)
                native = image_module.ComicPageParser((str(self.root), path.name), False, opt)
                deferred = image_module.ComicPageParser((str(self.root), path.name), False, opt, defer_processing=True)
                self.assertEqual(deferred.image.size, (100, 200))
                self.assertEqual(deferred.payload, [])
                self.assertEqual(native.crop_box, deferred.crop_box)
                self.assertEqual(native.image.size, deferred.image.crop(deferred.crop_box).size)
                if mode == 0:
                    self.assertEqual(deferred.crop_box, (0, 0, 100, 200))
                else:
                    self.assertNotEqual(deferred.crop_box, (0, 0, 100, 200))

    def test_pair_crops_actual_pixels_using_both_native_crop_modes(self):
        for mode in (1, 2):
            with self.subTest(mode=mode):
                work, paths = self.pair_work(self.options('--cropping', str(mode)), sizes=((100, 200), (150, 300)))
                for path in paths:
                    with Image.open(path) as source:
                        image = Image.new('RGB', source.size, 'white')
                    w, h = image.size
                    ImageDraw.Draw(image).rectangle((w // 10, h // 10, w * 9 // 10 - 1, h * 9 // 10 - 1), fill=(70, 70, 70))
                    image.save(path)
                result = c2e.imgPairProcessing(work)
                self.assertIsInstance(result, list, result)
                self.assertEqual(result[0][4]['rendered_size'][1], result[1][4]['rendered_size'][1])
                self.assertEqual(result[0][4]['offset'][1], result[1][4]['offset'][1])
                for page in result:
                    self.assertGreater(page[4]['crop_box'][0], 0)
                    with Image.open(self.root / page[0]) as output:
                        self.assertEqual(output.size, (100, 160))
                        x, y = page[4]['offset']
                        w, h = page[4]['rendered_size']
                        self.assertEqual(output.getpixel((x + w // 2, y + h // 2)), 70)

    def test_first_color_exemption_margin_and_minimum_area_are_native(self):
        opt = self.options('--cropping', '1', '--preservemargin', '50')
        path = self.source(1, (100, 200), 255)
        bbox = (10, 20, 90, 180)
        with patch.object(image_module, 'get_bbox_crop_margin', return_value=bbox), \
                patch.object(image_module, 'colorCheck', return_value=True):
            first = image_module.ComicPageParser((str(self.root), path.name), True, opt, defer_processing=True)
            later = image_module.ComicPageParser((str(self.root), path.name), False, opt, defer_processing=True)
        self.assertEqual(first.crop_box, (0, 0, 100, 200))
        self.assertEqual(later.crop_box, (5, 10, 95, 190))
        opt.croppingm = .99
        with patch.object(image_module, 'get_bbox_crop_margin', return_value=bbox):
            rejected = image_module.ComicPageParser((str(self.root), path.name), False, opt, defer_processing=True)
        self.assertEqual(rejected.crop_box, (0, 0, 100, 200))

    def test_off_native_worker_still_splits_wide_source_without_canvas(self):
        opt = self.options()
        opt.chapter_pairing = False
        path = self.source(1, (140, 100))
        result = c2e.imgFileProcessing([path.name, str(path.parent), opt, True])
        self.assertIsInstance(result, list, result)
        self.assertEqual([page[0] for page in result], ['kcc-0001-kcc-b.png', 'kcc-0001-kcc-c.png'])
        self.assertTrue(all(len(page) == 2 for page in result))
        with Image.open(self.root / result[0][0]) as output:
            self.assertEqual(output.size, (70, 100))

    def test_analysis_single_nested_cbz_and_fusion_source_titles(self):
        c2e.options = self.options()
        self.source(10, (50, 100), folder='chapter/b')
        self.source(2, (50, 100), folder='chapter/a')
        single = c2e.analyzeChapterPairing(self.root, {}, '42.cbz')
        self.assertEqual([(chapter['title'], chapter['source_ids']) for chapter in single['chapters']], [('42', [2, 10])])
        c2e.options.filefusion = True
        self.source(20, (50, 100), folder='next')
        fusion = c2e.analyzeChapterPairing(self.root, {'chapter': '42 源章名', 'next': '43 源章名'}, 'fused')
        self.assertEqual([(chapter['title'], chapter['source_ids']) for chapter in fusion['chapters']],
                         [('42 源章名', [2, 10]), ('43 源章名', [20])])
        self.assertEqual(fusion['state'], 'analyzing')

    def test_callback_keeps_geometry_by_resource_even_when_results_finish_out_of_order(self):
        c2e.options = self.options()
        c2e.options.imgMetadata = {}
        c2e.pairing_output = {}
        c2e.imgFileProcessingTick([['kcc-0002-kcc-x.png', ['BlackBackground'], 2, 1, {'offset': [0, 0]}]])
        c2e.imgFileProcessingTick([['kcc-0001-kcc-x.png', [], 1, 1, {'offset': [5, 0]}]])
        self.assertEqual(c2e.pairing_output['kcc-0001-kcc-x.png']['source_id'], 1)
        self.assertEqual(c2e.pairing_output['kcc-0002-kcc-x.png']['geometry'], {'offset': [0, 0]})
        self.assertEqual(c2e.options.imgMetadata['kcc-0002-kcc-x.png'], ['BlackBackground'])

    def test_directory_dispatch_passes_only_group_work_and_native_metadata(self):
        c2e.options = self.options()
        self.source(1, (60, 100))
        self.source(2, (60, 100))
        self.source(3, (60, 100))
        c2e.pairing_plan = c2e.analyzeChapterPairing(self.root, {}, 'one.cbz')
        pool = Mock()
        with patch.object(c2e, 'Pool', return_value=pool):
            c2e.imgDirectoryProcessing(str(self.root))
        calls = pool.apply_async.call_args_list
        self.assertEqual([call.kwargs['func'] for call in calls], [c2e.imgPairProcessing, c2e.imgFileProcessing])
        pair = calls[0].kwargs['args'][0]
        self.assertEqual(pair[1:], (1, [1, 2]))
        self.assertFalse(hasattr(pair[0][0][2], 'pairing_plan'))
        self.assertEqual(calls[1].kwargs['args'][0][4], dict(source_id=3, group_id=2))

    def test_real_pool_processes_pair_and_single_without_pickling_book_plan(self):
        c2e.options = self.options()
        for source_id in (1, 2, 3):
            self.source(source_id, (60, 100))
        c2e.pairing_plan = c2e.analyzeChapterPairing(self.root, {}, 'one.cbz')
        with patch.object(c2e, 'Pool', side_effect=lambda **kwargs: Pool(2, **kwargs)):
            c2e.imgDirectoryProcessing(str(self.root))
        self.assertEqual(set(c2e.pairing_output), {f'kcc-{i:04d}-kcc-x.png' for i in (1, 2, 3)})
        for source_id in (1, 2, 3):
            resource = f'kcc-{source_id:04d}-kcc-x.png'
            metadata = c2e.pairing_output[resource]
            self.assertEqual(metadata['source_id'], source_id)
            self.assertEqual(metadata['group_id'], 1 if source_id < 3 else 2)
            self.assertEqual(metadata['geometry'] is None, source_id == 3)
            with Image.open(self.root / resource) as output:
                self.assertEqual(output.size, (100, 160) if source_id < 3 else (60, 100))
        self.assertEqual(c2e.pairing_plan['state'], 'analyzing')
        self.assertEqual(c2e.pairing_plan['pages'], [])


if __name__ == '__main__':
    unittest.main()
