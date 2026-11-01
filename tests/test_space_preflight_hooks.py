import os
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kindlecomicconverter import comic2ebook as c2e


class SpaceHookTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / '1.cbz'
        self.source.write_bytes(b'archive')
        self.original_options = getattr(c2e, 'options', None)
        self.original_plan = c2e.space_plan
        self.addCleanup(setattr, c2e, 'options', self.original_options)
        self.addCleanup(setattr, c2e, 'space_plan', self.original_plan)
        c2e.space_plan = None
        self.options = c2e.checkOptions(c2e.makeParser().parse_args(['-p', 'KoLC', str(self.source)]))
        self.options.tempdir = True
        c2e.options = self.options

    def test_main_requires_both_absolute_environment_paths(self):
        for environment, error in (
            ({'CBZ2KEPUB_SPACE_PLAN': str(self.root / 'plan.json')}, 'Both'),
            ({'CBZ2KEPUB_SPACE_MODULE': str(self.root / 'module.py')}, 'Both'),
            ({'CBZ2KEPUB_SPACE_PLAN': 'plan.json', 'CBZ2KEPUB_SPACE_MODULE': 'module.py'}, 'absolute'),
        ):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                with self.assertRaisesRegex(ValueError, error):
                    c2e.main([])

    def test_main_loads_explicit_module_and_resets_plan_for_native_invocation(self):
        module = self.root / 'module.py'
        module.write_text(
            'class SpacePlan:\n'
            '    @classmethod\n'
            '    def from_file(cls, path):\n'
            '        return ("loaded", path)\n', encoding='utf-8')
        plan = self.root / 'plan.json'
        plan.write_text('{}')
        with patch.dict(os.environ, {'CBZ2KEPUB_SPACE_PLAN': str(plan),
                                     'CBZ2KEPUB_SPACE_MODULE': str(module)}, clear=True):
            self.assertEqual(c2e.main([]), 0)
        self.assertEqual(c2e.space_plan, ('loaded', str(plan)))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(c2e.main([]), 0)
        self.assertIsNone(c2e.space_plan)

    def test_archive_plan_replaces_native_multiplier_before_extract(self):
        plan = Mock()
        c2e.space_plan = plan
        target = self.root / 'extracted'
        with patch.object(c2e, 'disk_usage', side_effect=AssertionError('native multiplier')), \
                patch.object(c2e.comicarchive, 'ComicArchive') as archive, \
                patch.object(c2e, 'sanitizePermissions'):
            archive.return_value.extract.return_value = str(target)
            c2e.getWorkFolder(str(self.source), self.options, str(target))
        plan.before_extract.assert_called_once_with(str(self.source), str(target))
        archive.return_value.extract.assert_called_once_with(str(target))

    def test_archive_plan_failure_stops_before_archive_tool(self):
        c2e.space_plan = SimpleNamespace(before_extract=Mock(side_effect=ValueError('not enough space')))
        with patch.object(c2e.comicarchive, 'ComicArchive') as archive:
            with self.assertRaisesRegex(ValueError, 'not enough space'):
                c2e.getWorkFolder(str(self.source), self.options, str(self.root / 'extracted'))
        archive.assert_not_called()
        self.assertEqual(self.source.read_bytes(), b'archive')

    def test_native_archive_keeps_multiplier_guard(self):
        with patch.object(c2e, 'disk_usage', return_value=(100, 100, 0)), \
                patch.object(c2e.comicarchive, 'ComicArchive') as archive:
            with self.assertRaises(UserWarning):
                c2e.getWorkFolder(str(self.source), self.options, str(self.root / 'extracted'))
        archive.assert_not_called()

    def test_owned_fusion_with_plan_has_no_native_reservation(self):
        fused = self.root / 'fused'
        fused.mkdir()
        (fused / 'page.jpg').write_bytes(b'image')
        c2e.space_plan = Mock()
        with patch.object(c2e, 'disk_usage', side_effect=AssertionError('native multiplier')):
            work = Path(c2e.getWorkFolder(str(fused), self.options, owned_fusion=True))
        self.assertFalse(fused.exists())
        self.assertEqual((work / 'OEBPS' / 'Images' / 'page.jpg').read_bytes(), b'image')

    def book_mocks(self, stack, events):
        work = self.root / 'work'
        images = work / 'OEBPS' / 'Images'
        images.mkdir(parents=True)
        (images / 'page.jpg').write_bytes(b'image')
        for name in ('checkTools', 'checkPre', 'getMetadata', 'removeNonImages', 'detectSuboptimalProcessing'):
            stack.enter_context(patch.object(c2e, name))
        stack.enter_context(patch.object(c2e, 'getWorkFolder', return_value=str(work)))
        stack.enter_context(patch.object(c2e, 'sanitizeTree', return_value=({}, None)))
        stack.enter_context(patch.object(c2e.image, 'Cover', return_value=None))
        stack.enter_context(patch.object(c2e, 'chunk_directory', return_value=[str(work)]))
        stack.enter_context(patch.object(c2e, 'getOutputFilename', return_value=str(self.root / 'book.epub')))
        processing = stack.enter_context(patch.object(c2e, 'imgDirectoryProcessing', side_effect=lambda *a: events.append('process')))
        stack.enter_context(patch.object(c2e, 'buildEPUB', side_effect=lambda *a: events.append('build')))
        zipped = stack.enter_context(patch.object(c2e, 'makeZIP'))
        stack.enter_context(patch.object(c2e, 'available_archive_tools', return_value=[c2e.SEVENZIP]))
        return work, images, processing, zipped

    def test_capacity_checks_precede_processing_and_zip_and_follow_epub_build(self):
        events = []
        plan = Mock()
        plan.before_images.side_effect = lambda *a: events.append('images check')
        plan.after_images.side_effect = lambda *a: events.append('processed check')

        def reject_zip(*args, **kwargs):
            events.append('zip check')
            raise ValueError('zip space')

        plan.before_zip.side_effect = reject_zip
        c2e.space_plan = plan
        with ExitStack() as stack:
            work, images, processing, zipped = self.book_mocks(stack, events)
            with self.assertRaisesRegex(ValueError, 'zip space'):
                c2e.makeBook(str(self.source))
        self.assertEqual(events, ['images check', 'process', 'processed check', 'build', 'zip check'])
        plan.before_images.assert_called_once_with(str(images))
        plan.after_images.assert_called_once_with([str(work)])
        plan.before_zip.assert_called_once_with(str(work), str(self.root / 'book.epub'), sevenzip=True)
        zipped.assert_not_called()
        self.assertTrue(images.exists())

    def test_image_check_failure_stops_workers(self):
        c2e.space_plan = Mock()
        c2e.space_plan.before_images.side_effect = ValueError('image space')
        with ExitStack() as stack:
            _, images, processing, zipped = self.book_mocks(stack, [])
            with self.assertRaisesRegex(ValueError, 'image space'):
                c2e.makeBook(str(self.source))
        processing.assert_not_called()
        zipped.assert_not_called()
        self.assertTrue((images / 'page.jpg').exists())

    def test_processed_check_failure_stops_before_epub_build_and_zip(self):
        c2e.space_plan = Mock()
        c2e.space_plan.after_images.side_effect = ValueError('remaining space')
        self.options.targetsize = 1
        events = []
        with ExitStack() as stack:
            work, images, processing, zipped = self.book_mocks(stack, events)
            with self.assertRaisesRegex(ValueError, 'remaining space'):
                c2e.makeBook(str(self.source))
        c2e.space_plan.after_images.assert_called_once_with([str(work)])
        processing.assert_called_once()
        self.assertEqual(events, ['process'])
        zipped.assert_not_called()
        self.assertTrue((images / 'page.jpg').exists())


if __name__ == '__main__':
    unittest.main()
