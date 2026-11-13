import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kindlecomicconverter import comic2ebook as c2e


class OutputFilenameTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / '1.cbz'
        self.source.touch()
        self.options = SimpleNamespace(folder_output=False, profile='KoLC', format='EPUB',
                                       noKepub=False, output=str(self.root), title='漫畫:測試')
        options = patch.object(c2e, 'options', self.options, create=True)
        options.start()
        self.addCleanup(options.stop)

    def output(self, tome=''):
        path = Path(c2e.getOutputFilename(str(self.source), str(self.root), '.epub', tome))
        self.assertLessEqual(len(path.name.encode('utf-16-le')) // 2, 255)
        if sys.platform.startswith('win'):
            self.assertLessEqual(len(str(path).encode('utf-16-le')) // 2, 259)
        else:
            self.assertLessEqual(len(path.name.encode('utf-8')), 255)
        path.name.encode('utf-8')
        return path

    def test_short_title_keeps_upstream_naming(self):
        for title, name in (('漫畫:測試', '漫畫_測試'), ('Book.', 'Book.'), ('Book ', 'Book ')):
            with self.subTest(title=title):
                self.options.title = title
                self.assertEqual(self.output(' 2').name, name + ' 2.kepub.epub')
                self.assertEqual(self.options.title, title)

    def test_long_unicode_title_preserves_metadata_and_suffixes(self):
        title = '漫' + '😀' * 140
        self.options.title = title
        for no_kepub in (False, True):
            self.options.noKepub = no_kepub
            for tome in ('', ' 12'):
                with self.subTest(no_kepub=no_kepub, tome=tome):
                    path = self.output(tome)
                    extension = '.epub' if no_kepub else '.kepub.epub'
                    self.assertTrue(path.name.endswith(tome + extension))
                    self.assertIn('😀', path.name)
                    self.assertTrue(title.startswith(path.name[:-len(tome + extension)]))
                    path.write_bytes(b'epub')
                    self.assertEqual(self.options.title, title)

    def test_long_title_collisions_keep_all_outputs_through_two_digit_counter(self):
        self.options.title = 'A' * 246
        outputs = []
        for index in range(12):
            path = self.output(' 12')
            self.assertTrue(path.name.endswith(('' if index == 0 else f'_kcc{index - 1}') + '.kepub.epub'))
            self.assertIn(' 12', path.name)
            path.write_bytes(bytes([index]))
            outputs.append(path)
        self.assertEqual(len(set(outputs)), 12)
        self.assertEqual([path.read_bytes() for path in outputs], [bytes([i]) for i in range(12)])

    def test_long_source_name_is_bounded_when_title_is_default(self):
        length = min(250, 259 - len(str(self.root)) - 1 - len('.cbz')) if sys.platform.startswith('win') else 250
        self.source = self.root / ('A' * length + '.cbz')
        self.source.touch()
        self.options.title = 'defaulttitle'
        self.output().touch()

    def test_directory_without_filename_room_fails_clearly(self):
        self.options.output = str(self.root / ('A' * 250))
        with patch.object(c2e.sys, 'platform', 'win32'), patch.object(c2e.os.path, 'exists', return_value=True):
            with self.assertRaisesRegex(ValueError, 'Output directory leaves no room'):
                self.output()

    def test_mobi_and_epub_collisions_are_both_preserved(self):
        self.options.profile = 'KPW5'
        self.options.format = 'MOBI'
        self.options.title = 'A' * 251
        first = self.output()
        mobi = first.with_suffix('.mobi')
        mobi.write_bytes(b'old mobi')
        second = self.output()
        self.assertTrue(second.name.endswith('_kcc0.epub'))
        second.write_bytes(b'old epub')
        third = self.output()
        self.assertTrue(third.name.endswith('_kcc1.epub'))
        third.touch()
        self.assertEqual(mobi.read_bytes(), b'old mobi')
        self.assertEqual(second.read_bytes(), b'old epub')
