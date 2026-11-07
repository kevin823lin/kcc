import json
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from PIL import Image

from kindlecomicconverter import comic2ebook as c2e
from kindlecomicconverter.epub_helpers import COVER_ID, NS, OPF, XHTML, finalize_epub


class PairingEpubTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ('options', 'pairing_plan', 'pairing_output', 'GUI', 'space_plan'):
            if hasattr(c2e, name):
                self.addCleanup(setattr, c2e, name, getattr(c2e, name))
        c2e.space_plan = None
        c2e.options = SimpleNamespace(targetsize=1, batchsplit=1, webtoon=False)
        c2e.pairing_plan = dict(groups=[dict(id=1, source_ids=[1], mode='single'),
                                       dict(id=2, source_ids=[2, 3], mode='paired')])
        c2e.pairing_output = {
            f'kcc-{i:04d}-kcc-x.jpg': dict(source_id=i, group_id=1 if i == 1 else 2, geometry=None)
            for i in (1, 2, 3)}

    def image_tree(self):
        work = self.root / 'work'
        images = work / 'OEBPS' / 'Images'
        images.mkdir(parents=True)
        return work, images

    def write_page(self, parent, source_id, size):
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / f'kcc-{source_id:04d}-kcc-x.jpg'
        path.write_bytes(b'x' * size)

    def test_flat_and_oversized_chapter_split_keep_every_pair_together(self):
        for nested in (False, True):
            with self.subTest(nested=nested), tempfile.TemporaryDirectory(dir=self.root) as temporary:
                work = Path(temporary)
                images = work / 'OEBPS/Images'
                chapter = images / 'chapter' if nested else images
                for source_id, size in ((1, 400000), (2, 400000), (3, 400000)):
                    self.write_page(chapter, source_id, size)
                tomes = c2e.chunk_directory(str(work), {'chapter': '來源章名'} if nested else {})
                self.assertEqual(len(tomes), 2)
                sources = [[path.name for path in sorted((Path(tome) / 'OEBPS/Images').rglob('*.jpg'))] for tome in tomes]
                self.assertEqual(sources, [['kcc-0001-kcc-x.jpg'], ['kcc-0002-kcc-x.jpg', 'kcc-0003-kcc-x.jpg']])
                if nested:
                    self.assertTrue((Path(tomes[1]) / 'OEBPS/Images/chapter').is_dir())

    def test_pair_over_budget_rejected_in_both_page_split_paths(self):
        for nested, batchsplit in ((False, 1), (True, 1), (True, 2)):
            with self.subTest(nested=nested, batchsplit=batchsplit), tempfile.TemporaryDirectory(dir=self.root) as temporary:
                c2e.options.batchsplit = batchsplit
                work = Path(temporary)
                images = work / 'OEBPS/Images'
                chapter = images / 'chapter' if nested else images
                self.write_page(chapter, 2, 600000)
                self.write_page(chapter, 3, 600000)
                with patch.object(c2e, 'createNewTome') as create:
                    with self.assertRaisesRegex(UserWarning, 'paired group'):
                        c2e.chunk_directory(str(work), {'chapter': '來源章名'} if nested else {})
                create.assert_not_called()

    def test_native_above_below_groups_are_preserved_without_pairing(self):
        c2e.pairing_plan = None
        c2e.pairing_output = {}
        self.assertEqual(c2e.processedPageGroups(['a-above.jpg', 'a-below.jpg', 'b.jpg']),
                         [['a-above.jpg', 'a-below.jpg'], ['b.jpg']])

    def make_epub(self, direction):
        work, images = self.image_tree()
        text = work / 'OEBPS/Text'
        text.mkdir()
        Image.new('RGB', (100, 160), 'white').save(images / 'cover.jpg')
        manifest = ['<item id="cover" href="Images/cover.jpg" media-type="image/jpeg" properties="cover-image"/>']
        references = []
        processed = {}
        for source_id in range(1, 6):
            basename = f'kcc-{source_id:04d}-kcc-x'
            width = {1: 96, 2: 64}.get(source_id, 100)
            Image.new('RGB', (width, 160), 'gray').save(images / (basename + '.jpg'))
            (text / (basename + '.xhtml')).write_text(
                f'<html xmlns="{XHTML}"><head><meta name="viewport" content="width={width}, height=160"/></head>'
                f'<body><img src="../Images/{basename}.jpg"/></body></html>', encoding='utf-8')
            manifest.append(f'<item id="p{source_id}" href="Text/{basename}.xhtml" media-type="application/xhtml+xml"/>')
            references.append(f'<itemref idref="p{source_id}" properties="rendition:page-spread-left custom:keep"/>')
            group_id = 1 if source_id < 3 else source_id - 1
            processed[basename + '.jpg'] = dict(source_id=source_id, group_id=group_id,
                geometry=dict(crop_box=[0, 0, width, 160], rendered_size=[width, 160]) if source_id < 3 else None)
        opf = work / 'OEBPS/content.opf'
        opf.write_text(f'<package xmlns="{OPF}" xmlns:custom="urn:custom"><metadata/>'
                       f'<manifest>{"".join(manifest)}</manifest><spine page-progression-direction="{direction}">'
                       f'{"".join(references)}</spine></package>', encoding='utf-8')
        plan = dict(direction=direction, groups=[dict(id=1, source_ids=[1, 2], mode='paired'),
                                                dict(id=2, source_ids=[3], mode='single'),
                                                dict(id=3, source_ids=[4], mode='native'),
                                                dict(id=4, source_ids=[5], mode='native')])
        return work, opf, plan, processed

    def test_actual_spine_report_sides_order_paths_viewports_and_native_properties(self):
        for direction in ('rtl', 'ltr'):
            with self.subTest(direction=direction), tempfile.TemporaryDirectory(dir=self.root) as temporary:
                original = self.root
                self.root = Path(temporary)
                try:
                    work, opf, plan, processed = self.make_epub(direction)
                finally:
                    self.root = original
                before = ET.parse(opf).find('opf:spine', NS)
                native = [dict(reference.attrib) for reference in before][-2:]
                cover_bytes = (work / 'OEBPS/Images/cover.jpg').read_bytes()
                pages = finalize_epub(work, '系列', 2, pairing_plan=plan, pairing_output=processed)
                spine = ET.parse(opf).find('opf:spine', NS)
                self.assertEqual(spine[0].get('idref'), COVER_ID)
                self.assertEqual([dict(reference.attrib) for reference in spine][-2:], native)
                expected = ['right', 'left'] if direction == 'rtl' else ['left', 'right']
                self.assertEqual([page['side'] for page in pages], expected + ['center', 'left', 'left'])
                self.assertEqual([page['source_id'] for page in pages], [1, 2, 3, 4, 5])
                self.assertEqual([page['tome'] for page in pages], [2] * 5)
                self.assertEqual([page['image_size'] for page in pages], [[96, 160], [64, 160]] + [[100, 160]] * 3)
                self.assertTrue(all(page['image_size'] == page['viewport'] for page in pages))
                self.assertEqual(pages[0]['image'], 'OEBPS/Images/kcc-0001-kcc-x.jpg')
                self.assertEqual(pages[0]['xhtml'], 'OEBPS/Text/kcc-0001-kcc-x.xhtml')
                self.assertTrue(all('custom:keep' in reference.get('properties', '') for reference in list(spine)[1:]))
                self.assertEqual((work / 'OEBPS/Images/cover.jpg').read_bytes(), cover_bytes)
                self.assertEqual(len(pages), 5)

    def test_parent_report_writes_atomically_and_requires_absolute_path(self):
        c2e.pairing_plan = dict(version=2, state='analyzing', pages=[])
        report = self.root / 'pairing.json'
        with patch.dict(os.environ, {'CBZ2KEPUB_PAIRING_REPORT': str(report)}), \
                patch.object(c2e.os, 'replace', wraps=os.replace) as replace:
            c2e.writePairingReport()
        replace.assert_called_once_with(report.with_name('pairing.json.tmp'), report)
        self.assertEqual(json.loads(report.read_text(encoding='utf-8')), c2e.pairing_plan)
        self.assertFalse(report.with_name('pairing.json.tmp').exists())
        with patch.dict(os.environ, {'CBZ2KEPUB_PAIRING_REPORT': 'pairing.json'}):
            with self.assertRaisesRegex(ValueError, 'absolute'):
                c2e.writePairingReport()

    def test_report_capacity_hook_receives_actual_encoded_bytes_before_write(self):
        c2e.pairing_plan = dict(version=2, state='analyzing', chapters=[dict(title='測試')])
        path = self.root / 'pairing.json'
        observed = []
        def check(destination, size):
            self.assertEqual(destination, path)
            self.assertFalse(path.with_name('pairing.json.tmp').exists())
            observed.append(size)
        c2e.space_plan = SimpleNamespace(before_pairing_report=check)
        with patch.dict(os.environ, {'CBZ2KEPUB_PAIRING_REPORT': str(path)}):
            c2e.writePairingReport()
        self.assertEqual(observed, [path.stat().st_size])

    def test_failure_before_processing_replaces_stale_completed_report_with_analyzing(self):
        c2e.options = c2e.checkOptions(c2e.makeParser().parse_args(['-p', 'KoLC', '--chapter-pairing']))
        report = self.root / 'pairing.json'
        report.write_text('{"state":"completed"}', encoding='utf-8')
        with patch.dict(os.environ, {'CBZ2KEPUB_PAIRING_REPORT': str(report)}), \
                patch.object(c2e, 'checkTools'), patch.object(c2e, 'checkPre'), \
                patch.object(c2e, 'getWorkFolder', side_effect=RuntimeError('extract failed')):
            with self.assertRaisesRegex(RuntimeError, 'extract failed'):
                c2e.makeBook(str(self.root / 'chapter.cbz'))
        self.assertEqual(json.loads(report.read_text(encoding='utf-8'))['state'], 'analyzing')

    def test_second_tome_zip_failure_never_marks_report_completed(self):
        c2e.options = c2e.checkOptions(c2e.makeParser().parse_args([
            '-p', 'KoLC', '--chapter-pairing', '--targetsize', '1']))
        work, images = self.image_tree()
        Image.new('RGB', (100, 160), 'gray').save(images / 'kcc-0001.png')
        other = self.root / 'second'
        other.mkdir()
        report = self.root / 'pairing.json'
        zipped = []
        def fail_second(*args):
            self.assertEqual(json.loads(report.read_text(encoding='utf-8'))['state'], 'analyzing')
            zipped.append(args)
            if len(zipped) == 2:
                raise RuntimeError('second ZIP failed')
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {'CBZ2KEPUB_PAIRING_REPORT': str(report)}))
            for name in ('checkTools', 'checkPre', 'getMetadata', 'removeNonImages', 'detectSuboptimalProcessing',
                         'imgDirectoryProcessing', 'buildEPUB'):
                stack.enter_context(patch.object(c2e, name))
            stack.enter_context(patch.object(c2e, 'getWorkFolder', return_value=str(work)))
            stack.enter_context(patch.object(c2e, 'sanitizeTree', return_value=({}, images / 'kcc-0001.png')))
            stack.enter_context(patch.object(c2e.image, 'Cover', return_value=None))
            stack.enter_context(patch.object(c2e, 'chunk_directory', return_value=[str(work), str(other)]))
            stack.enter_context(patch.object(c2e, 'getOutputFilename', return_value=str(self.root / 'out.epub')))
            stack.enter_context(patch.object(c2e, 'makeZIP', side_effect=fail_second))
            with self.assertRaisesRegex(RuntimeError, 'second ZIP failed'):
                c2e.makeBook(str(self.root / 'chapter.cbz'))
        self.assertEqual(len(zipped), 2)
        self.assertEqual(c2e.pairing_plan['state'], 'analyzing')
        self.assertEqual(json.loads(report.read_text(encoding='utf-8'))['state'], 'analyzing')


if __name__ == '__main__':
    unittest.main()
