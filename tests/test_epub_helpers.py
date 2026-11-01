from io import BytesIO
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from PIL import Image

from kindlecomicconverter.epub_helpers import COVER_ID, NS, OPF, finalize_epub


class FinalizeEpubTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.opf = self.root / 'OEBPS/content.opf'
        self.opf.parent.mkdir()
        self.image = self.root / 'OEBPS/Images/cover.jpg'
        self.image.parent.mkdir()
        Image.new('RGB', (123, 234), 'red').save(self.image)
        self.body = self.root / 'OEBPS/Text/page.xhtml'
        self.body.parent.mkdir()
        self.body.write_bytes(b'original body bytes')
        self.cover = self.body.with_name('cbz2kepub-cover.xhtml')
        self.original = f'''<package xmlns="{OPF}" version="3.0"
          xmlns:dc="http://purl.org/dc/elements/1.1/"
          xmlns:rendition="http://www.idpf.org/vocab/rendition/#"
          xmlns:custom="urn:custom">
          <metadata>
            <dc:title id="title">Native title</dc:title>
            <dc:creator id="creator">Native author</dc:creator>
            <dc:description>Native description</dc:description>
            <meta property="custom:value">custom:term</meta>
            <meta property="rendition:layout">pre-paginated</meta>
            <meta refines="#creator" property="role">aut</meta>
            <meta name="cover" content="cover"/>
            <meta property="belongs-to-collection" id="old-series">Old</meta>
            <meta refines="#old-series" property="collection-type">series</meta>
            <meta refines="#old-series" property="group-position">99</meta>
            <meta name="calibre:series" content="Old"/>
            <meta name="calibre:series_index" content="99"/>
          </metadata>
          <manifest>
            <item id="cover" href="Images/cover.jpg" media-type="image/jpeg" properties="cover-image"/>
            <item id="body" href="Text/page.xhtml" media-type="application/xhtml+xml"/>
          </manifest>
          <spine page-progression-direction="rtl" toc="ncx">
            <itemref idref="body" linear="yes" properties="rendition:page-spread-left"/>
            <itemref idref="body2" properties="rendition:page-spread-right"/>
          </spine>
        </package>'''
        self.opf.write_text(self.original, encoding='utf-8')

    def test_cover_first_centered_and_body_metadata_images_unchanged(self):
        before = ET.parse(self.opf)
        body_refs = [dict(node.attrib) for node in before.find('opf:spine', NS)]
        metadata = before.find('opf:metadata', NS)
        untouched = [ET.tostring(node) for node in list(metadata)[:7]]
        image_bytes = self.image.read_bytes()
        finalize_epub(self.root, 'New series & title', 2)
        package = ET.parse(self.opf)
        spine = package.find('opf:spine', NS)
        self.assertEqual(spine[0].attrib, {'idref': COVER_ID, 'properties': 'rendition:page-spread-center'})
        self.assertEqual([dict(node.attrib) for node in spine][1:], body_refs)
        self.assertEqual(spine.attrib, before.find('opf:spine', NS).attrib)
        self.assertEqual(self.body.read_bytes(), b'original body bytes')
        self.assertEqual(self.image.read_bytes(), image_bytes)
        self.assertEqual([ET.tostring(node) for node in list(package.find('opf:metadata', NS))[:7]], untouched)
        html = ET.parse(self.cover)
        self.assertEqual(html.find('.//{*}meta').get('content'), 'width=123, height=234')
        self.assertEqual(html.find('.//{*}image').get('href'), '../Images/cover.jpg')
        self.assertEqual(html.find('.//{*}svg').get('viewBox'), '0 0 123 234')
        declarations = dict(value for event, value in ET.iterparse(BytesIO(self.opf.read_bytes()), events=('start-ns',)))
        self.assertEqual(declarations['rendition'], 'http://www.idpf.org/vocab/rendition/#')
        self.assertEqual(declarations['custom'], 'urn:custom')
        metadata = package.find('opf:metadata', NS)
        self.assertFalse(any(node.get('name', '').startswith('calibre:series') or node.get('refines') == '#old-series' for node in metadata))
        collection = metadata.find("opf:meta[@property='belongs-to-collection']", NS)
        self.assertEqual(collection.text, 'New series & title')
        self.assertEqual(metadata.find("opf:meta[@property='group-position']", NS).text, '2')
        self.assertEqual(metadata.find("opf:meta[@property='collection-type']", NS).text, 'series')

    def test_non_kcc_cover_is_rejected_before_writes(self):
        for old, new in (('Images/cover.jpg', '../Images/cover.jpg'),
                         ('Images/cover.jpg', '../../outside.jpg'),
                         ('Images/cover.jpg', 'https://example.com/cover.jpg'),
                         ('id="cover"', 'id="other-cover"'),
                         ('image/jpeg', 'image/png'),
                         (' properties="cover-image"', '')):
            with self.subTest(old=old, new=new):
                self.opf.write_text(self.original.replace(old, new), encoding='utf-8')
                original = self.opf.read_bytes()
                with self.assertRaises(ValueError):
                    finalize_epub(self.root, 'Series', 1)
                self.assertEqual(self.opf.read_bytes(), original)
                self.assertFalse(self.cover.exists())

    def test_repeat_call_is_rejected_without_changing_output(self):
        finalize_epub(self.root, 'Series', 1)
        original = self.opf.read_bytes(), self.cover.read_bytes()
        with self.assertRaises(ValueError):
            finalize_epub(self.root, 'Series', 1)
        self.assertEqual((self.opf.read_bytes(), self.cover.read_bytes()), original)

    def test_collisions_fail_without_writes(self):
        for extra in (
            '<item id="cbz2kepub-cover" href="Text/page.xhtml" media-type="application/xhtml+xml"/>',
            '<item id="cbz2kepub-cover" href="Text/cbz2kepub-cover.xhtml" media-type="image/png"/>',
            '<item id="other" href="Text/cbz2kepub-cover.xhtml" media-type="application/xhtml+xml"/>',
            '<item id="cbz2kepub-series" href="Text/page.xhtml"/>',
        ):
            with self.subTest(extra=extra):
                self.opf.write_text(self.original.replace('</manifest>', extra + '</manifest>'), encoding='utf-8')
                original = self.opf.read_bytes()
                with self.assertRaises(ValueError):
                    finalize_epub(self.root, 'Series', 1)
                self.assertEqual(self.opf.read_bytes(), original)
                self.assertFalse(self.cover.exists())
        self.opf.write_text(self.original, encoding='utf-8')
        self.cover.write_bytes(b'unrelated existing file')
        with self.assertRaises(ValueError):
            finalize_epub(self.root, 'Series', 1)
        self.assertEqual(self.cover.read_bytes(), b'unrelated existing file')

    def test_required_nodes_cover_and_series_arguments(self):
        for name in ('metadata', 'manifest', 'spine'):
            package = ET.fromstring(self.original)
            package.remove(package.find('opf:' + name, NS))
            self.opf.write_bytes(ET.tostring(package))
            with self.subTest(name=name), self.assertRaises(ValueError):
                finalize_epub(self.root, 'Series', 1)
        self.opf.write_text(self.original.replace(' properties="cover-image"', ''), encoding='utf-8')
        with self.assertRaises(ValueError):
            finalize_epub(self.root, 'Series', 1)
        for name, index in (('', 1), (' ', 1), (None, 1), ('Series', 0), ('Series', -1), ('Series', True), ('Series', '2')):
            with self.subTest(name=name, index=index), self.assertRaises(ValueError):
                finalize_epub(self.root, name, index)


if __name__ == '__main__':
    unittest.main()
