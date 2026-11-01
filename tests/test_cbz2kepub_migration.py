import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kindlecomicconverter import comic2ebook as c2e


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.options = SimpleNamespace(tempdir=True, targetsize=1, webtoon=False, batchsplit=1,
                                       language='zh-TW', uuid='test', comicinfo_chapters=[])
        self.original_options = getattr(c2e, 'options', None)
        c2e.options = self.options
        self.addCleanup(setattr, c2e, 'options', self.original_options)

    def image_tree(self):
        work = self.root / 'KCC-test'
        images = work / 'OEBPS' / 'Images'
        images.mkdir(parents=True)
        return work, images

    def write_page(self, directory, name, size):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(b'x' * size)

    def test_fusion_carries_book_xml_after_chapter_sanitizing(self):
        sources = [self.root / '1.cbz', self.root / '2.cbz']
        for source in sources:
            source.touch()
        metadata = self.root / 'ComicInfo.xml'
        original = b'<ComicInfo><Series>Book</Series></ComicInfo>'
        metadata.write_bytes(original)

        def extract(source, options, target):
            directory = Path(target)
            self.write_page(directory, '1.jpg', 10)
            return target

        with patch.object(c2e, 'getWorkFolder', side_effect=extract), \
                patch.object(c2e, 'sanitizeTree') as sanitize, \
                patch.object(c2e, 'flattenTree'):
            fused, _ = c2e.makeFusion(list(map(str, sources)))
        self.assertEqual(sanitize.call_count, 2)
        self.assertEqual((Path(fused) / 'ComicInfo.xml').read_bytes(), original)
        self.assertEqual(metadata.read_bytes(), original)

    def test_single_cbz_overlays_only_provided_book_fields_before_native_reader(self):
        source = self.root / '1.cbz'
        source.touch()
        external = self.root / 'ComicInfo.xml'
        external.write_text('<ComicInfo><Series>Book</Series><Summary>New</Summary></ComicInfo>', encoding='utf-8')
        original = external.read_bytes()
        work, images = self.image_tree()
        (images / 'ComicInfo.xml').write_text(
            '<ComicInfo><Series>Chapter</Series><Writer>Embedded</Writer><Summary>Old</Summary></ComicInfo>')
        self.options.filefusion = False
        self.options.lightnovel = False
        with patch.object(c2e, 'checkTools'), patch.object(c2e, 'checkPre'), \
                patch.object(c2e, 'getWorkFolder', return_value=str(work)), \
                patch.object(c2e, 'getMetadata', side_effect=RuntimeError('reader reached')):
            with self.assertRaisesRegex(RuntimeError, 'reader reached'):
                c2e.makeBook(str(source))
        merged = ET.parse(images / 'ComicInfo.xml').getroot()
        self.assertEqual(merged.findtext('Series'), 'Book')
        self.assertEqual(merged.findtext('Writer'), 'Embedded')
        self.assertEqual(merged.findtext('Summary'), 'New')
        self.assertEqual(external.read_bytes(), original)

    def test_malformed_embedded_xml_falls_back_to_valid_book_xml(self):
        source = self.root / '1.cbz'
        source.touch()
        external = self.root / 'ComicInfo.xml'
        external.write_text('<ComicInfo><Series>Book</Series><Summary>New</Summary></ComicInfo>', encoding='utf-8')
        original = external.read_bytes()
        work, images = self.image_tree()
        embedded = images / 'ComicInfo.xml'
        self.options.filefusion = False
        self.options.lightnovel = False
        for invalid_xml in ('<ComicInfo><Summary>broken', ''):
            embedded.write_text(invalid_xml, encoding='utf-8')
            with self.subTest(invalid_xml=invalid_xml), \
                    patch.object(c2e, 'checkTools'), patch.object(c2e, 'checkPre'), \
                    patch.object(c2e, 'getWorkFolder', return_value=str(work)), \
                    patch.object(c2e, 'getMetadata', side_effect=RuntimeError('reader reached')):
                with self.assertRaisesRegex(RuntimeError, 'reader reached'):
                    c2e.makeBook(str(source))
            self.assertIn('<Series>Book</Series>', embedded.read_text(encoding='utf-8'))
            self.assertEqual(external.read_bytes(), original)

    def test_malformed_external_xml_still_fails_before_native_reader(self):
        source = self.root / '1.cbz'
        source.touch()
        (self.root / 'ComicInfo.xml').write_text('<ComicInfo><Series>broken', encoding='utf-8')
        work, images = self.image_tree()
        (images / 'ComicInfo.xml').write_text('<ComicInfo><Series>Embedded</Series></ComicInfo>', encoding='utf-8')
        self.options.filefusion = False
        self.options.lightnovel = False
        with patch.object(c2e, 'checkTools'), patch.object(c2e, 'checkPre'), \
                patch.object(c2e, 'getWorkFolder', return_value=str(work)), \
                patch.object(c2e, 'getMetadata') as reader:
            with self.assertRaises(ET.ParseError):
                c2e.makeBook(str(source))
        reader.assert_not_called()

    def test_owned_fusion_is_renamed_with_metadata(self):
        source = self.root / 'long title [fused]'
        self.write_page(source / 'chapter', '1.jpg', 10)
        (source / 'ComicInfo.xml').write_text('<ComicInfo/>')
        with patch.object(c2e, 'copytree', side_effect=AssertionError('must not copy')):
            work = Path(c2e.getWorkFolder(str(source), self.options, owned_fusion=True))
        self.assertFalse(source.exists())
        self.assertEqual(work.parent, self.root)
        self.assertTrue(work.name.startswith('KCC-'))
        self.assertEqual((work / 'OEBPS/Images/chapter/1.jpg').read_bytes(), b'x' * 10)
        self.assertTrue((work / 'OEBPS/Images/ComicInfo.xml').is_file())

    def test_fused_suffix_does_not_grant_ownership(self):
        source = self.root / 'user data [fused]'
        self.write_page(source, '1.jpg', 10)
        work = Path(c2e.getWorkFolder(str(source), self.options))
        self.assertTrue((source / '1.jpg').is_file())
        self.assertTrue((work / 'OEBPS/Images/1.jpg').is_file())

    def test_default_owned_workspace_is_outside_long_fusion_parent(self):
        source = self.root / 'LLL-owned' / 'a long title [fused]'
        self.write_page(source, '1.jpg', 10)
        self.options.tempdir = False
        with patch.object(c2e, 'gettempdir', return_value=str(self.root)):
            work = Path(c2e.getWorkFolder(str(source), self.options, owned_fusion=True))
        self.assertEqual(work.parent, self.root)
        self.assertTrue((work / 'OEBPS/Images/1.jpg').is_file())

    def test_owned_fusion_uses_lower_space_check_and_never_copy_fallback(self):
        source = self.root / 'fused'
        self.write_page(source, '1.jpg', 100)
        with patch.object(c2e, 'disk_usage', return_value=(1000, 850, 150)):
            work = Path(c2e.getWorkFolder(str(source), self.options, owned_fusion=True))
        self.assertTrue((work / 'OEBPS/Images/1.jpg').is_file())
        source.mkdir()
        self.write_page(source, '1.jpg', 100)
        with patch.object(c2e, 'disk_usage', return_value=(1000, 890, 110)):
            with self.assertRaises(UserWarning):
                c2e.getWorkFolder(str(source), self.options, owned_fusion=True)
        self.assertTrue((source / '1.jpg').is_file())
        with patch.object(c2e.os, 'rename', side_effect=OSError('different volume')):
            with self.assertRaises(UserWarning):
                c2e.getWorkFolder(str(source), self.options, owned_fusion=True)
        self.assertTrue((source / '1.jpg').is_file())

    def test_native_main_explicitly_marks_its_fusion(self):
        sources = [self.root / '1.cbz', self.root / '2.cbz']
        for source in sources:
            source.touch()
        fused = str(self.root / 'owned [fused]')
        with patch.object(c2e, 'makeFusion', return_value=(fused, None)), \
                patch.object(c2e, 'checkOptions', side_effect=lambda options: options), \
                patch.object(c2e, 'makeBook') as make_book:
            self.assertEqual(c2e.main(['--filefusion', *map(str, sources)]), 0)
        make_book.assert_called_once_with(fused, None, owned_fusion=True)

    def test_decimal_and_integer_page_order(self):
        source = self.root / 'pages'
        source.mkdir()
        names = ['24.1.jpg', '10.jpg', '2.jpg', '24.05.jpg', '1.jpg']
        for name in names:
            (source / name).write_text(name)
        self.options.format = 'EPUB'
        c2e.sanitizeTree(str(source), self.options)
        actual = [p.read_text() for p in sorted(source.iterdir())]
        self.assertEqual(actual, ['1.jpg', '2.jpg', '10.jpg', '24.05.jpg', '24.1.jpg'])

    @unittest.skipUnless(os.name == 'nt', 'Windows path policy')
    def test_long_transliterated_slug_is_shortened_before_rename_and_keeps_order(self):
        work, images = self.image_tree()
        names = ['a-short', 'b-' + '長章節名稱' * 12, 'c0001']
        self.options.format = 'EPUB'
        self.assertGreater(len(str(images / c2e.slugify(names[1], self.options, True))), 260)
        for index, name in enumerate(names):
            directory = images / name
            directory.mkdir()
            (directory / 'page.jpg').write_text(str(index))
        chapters, cover = c2e.sanitizeTree(str(images), self.options)
        folders = sorted(images.iterdir())
        self.assertEqual([chapters[p.name] for p in folders], names)
        self.assertEqual([next(p.iterdir()).read_text() for p in folders], ['0', '1', '2'])
        self.assertTrue(all(len(str(p)) <= 220 for p in images.rglob('*.jpg')))
        self.assertEqual(Path(cover).read_text(), '0')

    def test_oversized_chapters_keep_navigation_and_all_pages(self):
        work, images = self.image_tree()
        chapters = {'c1': '第一話', 'c2': '第二話', 'c3': '第三話'}
        for chapter, pages in [('c1', [(1, 600000), (2, 600000)]),
                               ('c2', [(3, 300000)]),
                               ('c3', [(4, 600000), (5, 600000)])]:
            for number, size in pages:
                self.write_page(images / chapter, f'kcc-{number:04}-kcc-x.jpg', size)
        tomes = c2e.chunk_directory(str(work), chapters)
        expected = [['第一話'], ['第一話', '第二話'], ['第三話'], ['第三話']]
        seen = []
        for tome, expected_titles in zip(tomes, expected):
            image_root = Path(tome) / 'OEBPS/Images'
            pages = sorted(image_root.rglob('*.jpg'))
            self.assertTrue(pages)
            self.assertLessEqual(sum(p.stat().st_size for p in pages), 1048576)
            seen.extend(p.name for p in pages)
            entries = []
            for directory in sorted(image_root.iterdir()):
                first = sorted(directory.iterdir())[0]
                entries.append((str(directory).replace('Images', 'Text'), first.name))
            c2e.buildNAV(tome, '作品', entries, chapters)
            c2e.buildNCX(tome, '作品', entries, chapters)
            nav = ET.parse(Path(tome) / 'OEBPS/nav.xhtml')
            titles = [a.text for a in nav.findall('.//{*}nav')[0].findall('.//{*}a')]
            self.assertEqual(titles, expected_titles)
            ncx = ET.parse(Path(tome) / 'OEBPS/toc.ncx')
            self.assertEqual([e.text for e in ncx.findall('.//{*}navLabel/{*}text')], expected_titles)
        self.assertEqual(len(tomes), len(expected))
        self.assertEqual(sorted(seen), [f'kcc-{n:04}-kcc-x.jpg' for n in range(1, 6)])

    def test_above_below_pair_stays_in_one_tome(self):
        work, images = self.image_tree()
        chapter = images / 'chapter'
        self.write_page(chapter, 'kcc-0001-kcc-x.jpg', 300000)
        self.write_page(chapter, 'kcc-0002-kcc-x-above.jpg', 400000)
        self.write_page(chapter, 'kcc-0002-kcc-x-below.jpg', 400000)
        tomes = c2e.chunk_directory(str(work), {'chapter': '章節'})
        self.assertEqual(len(tomes), 2)
        pairs = list((Path(tomes[1]) / 'OEBPS/Images/chapter').iterdir())
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all('0002' in p.name for p in pairs))

    def test_single_page_over_budget_errors_without_empty_tome(self):
        work, images = self.image_tree()
        self.write_page(images / 'chapter', 'kcc-0001-kcc-x.jpg', 1048577)
        with patch.object(c2e, 'createNewTome') as create_tome:
            with self.assertRaises(UserWarning):
                c2e.chunk_directory(str(work), {'chapter': '章節'})
        create_tome.assert_not_called()

    def test_flat_above_below_pair_cannot_be_split_to_meet_budget(self):
        work, images = self.image_tree()
        self.write_page(images, 'kcc-0001-kcc-x-above.jpg', 600000)
        self.write_page(images, 'kcc-0001-kcc-x-below.jpg', 600000)
        with patch.object(c2e, 'createNewTome') as create_tome:
            with self.assertRaises(UserWarning):
                c2e.chunk_directory(str(work), {})
        create_tome.assert_not_called()

    @unittest.skipUnless(os.name == 'nt', 'Windows path policy')
    def test_windows_long_chapter_names_are_shortened_without_losing_titles(self):
        for length in (220, 221):
            with self.subTest(length=length), tempfile.TemporaryDirectory(dir=self.root) as temporary:
                work = Path(temporary) / 'KCC-test'
                images = work / 'OEBPS/Images'
                name = 'a' * (length - len(str(images)) - len('\\page.jpg') - 1)
                self.write_page(images / name, 'page.jpg', 10)
                self.write_page(images / 'c0001', 'other.jpg', 10)
                chapters = {name: '長中文章名', 'c0001': '原本就叫 c0001'}
                self.assertEqual(len(str(images / name / 'page.jpg')), length)
                tomes = c2e.chunk_directory(str(work), chapters)
                self.assertEqual(tomes, [str(work)])
                folders = sorted(p for p in images.iterdir() if p.is_dir())
                self.assertEqual([chapters[p.name] for p in folders], ['長中文章名', '原本就叫 c0001'])
                self.assertTrue(all(len(str(p)) <= 220 for p in images.rglob('*.jpg')))
                self.assertEqual((images / name).exists(), length == 220)


if __name__ == '__main__':
    unittest.main()
