"""Finish unpacked EPUB metadata and cover before KCC creates the ZIP."""

from io import BytesIO
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from PIL import Image


OPF = "http://www.idpf.org/2007/opf"
XHTML = "http://www.w3.org/1999/xhtml"
EPUB = "http://www.idpf.org/2007/ops"
SVG = "http://www.w3.org/2000/svg"
COVER_ID = "cbz2kepub-cover"
COVER_HREF = "Text/cbz2kepub-cover.xhtml"
SERIES_ID = "cbz2kepub-series"
NS = {"opf": OPF}
ET.register_namespace("", OPF)
ET.register_namespace("dc", "http://purl.org/dc/elements/1.1/")
ET.register_namespace("epub", EPUB)


def finalize_epub(path, series_name, series_index, *, pairing_plan=None, pairing_output=None):
    """Finish a freshly generated KCC EPUB before its first ZIP packaging."""
    if not isinstance(series_name, str) or not series_name.strip() or type(series_index) is not int or series_index < 1:
        raise ValueError("A series name and positive integer series index are required")
    root = Path(path).resolve()
    opf_path = root / 'OEBPS/content.opf'
    source = opf_path.read_bytes()
    declarations = []
    for event, value in ET.iterparse(BytesIO(source), events=("start-ns", "start")):
        if event == "start":
            break
        declarations.append(value)
    package = ET.fromstring(source)
    if package.tag != f"{{{OPF}}}package":
        raise ValueError("Invalid OPF package document")
    for name in ("metadata", "manifest", "spine"):
        if package.find(f"opf:{name}", NS) is None:
            raise ValueError(f"OPF is missing {name}")
    metadata = package.find("opf:metadata", NS)
    manifest = package.find("opf:manifest", NS)
    spine = package.find("opf:spine", NS)
    covers = manifest.findall("opf:item[@id='cover']", NS)
    if (len(covers) != 1 or covers[0].get('href') != 'Images/cover.jpg'
            or covers[0].get('media-type') != 'image/jpeg'
            or 'cover-image' not in covers[0].get('properties', '').split()):
        raise ValueError('Unexpected KCC cover image')
    image_path = opf_path.parent / 'Images/cover.jpg'
    with Image.open(image_path) as image:
        width, height = image.size
        image.verify()
    if not width or not height:
        raise ValueError("Cover image has invalid dimensions")

    cover_path = opf_path.parent / COVER_HREF
    if cover_path.exists() or any(item.get('href') == COVER_HREF for item in manifest):
        raise ValueError("Cover page path is already occupied")
    if any(node.get('id') in (COVER_ID, SERIES_ID) for node in package.iter()):
        raise ValueError('Cover page or series ID is already occupied')

    collections = {node.get("id") for node in metadata
                   if node.get("property") == "belongs-to-collection" and node.get("id")}
    removed = [node for node in metadata if (node.get("property") == "belongs-to-collection"
               or node.get("refines", "").removeprefix("#") in collections
               or node.get("name") in ("calibre:series", "calibre:series_index"))]
    for node in removed:
        metadata.remove(node)
    for attributes, text in (
        ({"property": "belongs-to-collection", "id": SERIES_ID}, series_name),
        ({"property": "collection-type", "refines": f"#{SERIES_ID}"}, "series"),
        ({"property": "group-position", "refines": f"#{SERIES_ID}"}, str(series_index)),
    ):
        ET.SubElement(metadata, f"{{{OPF}}}meta", attributes).text = text
    ET.SubElement(manifest, f"{{{OPF}}}item", {
        "id": COVER_ID, "href": COVER_HREF, "media-type": "application/xhtml+xml"})
    spine.insert(0, ET.Element(f"{{{OPF}}}itemref", {
        "idref": COVER_ID, "properties": "rendition:page-spread-center"}))

    html = ET.Element(f"{{{XHTML}}}html")
    head = ET.SubElement(html, f"{{{XHTML}}}head")
    ET.SubElement(head, f"{{{XHTML}}}title").text = "Cover"
    ET.SubElement(head, f"{{{XHTML}}}meta", {"name": "viewport", "content": f"width={width}, height={height}"})
    ET.SubElement(head, f"{{{XHTML}}}style").text = "html,body{margin:0;padding:0;width:100%;height:100%;}svg{display:block;width:100%;height:100%;}"
    body = ET.SubElement(html, f"{{{XHTML}}}body", {f"{{{EPUB}}}type": "cover"})
    svg = ET.SubElement(body, f"{{{SVG}}}svg", {
        "viewBox": f"0 0 {width} {height}", "preserveAspectRatio": "xMidYMid meet"})
    ET.SubElement(svg, f"{{{SVG}}}image", {
        "width": str(width), "height": str(height), "preserveAspectRatio": "xMidYMid meet",
        "href": '../Images/cover.jpg', "{http://www.w3.org/1999/xlink}href": '../Images/cover.jpg'})

    pages = []
    if pairing_plan is not None:
        pages = finalize_pairing(root, package, pairing_plan, pairing_output, series_index)

    # ElementTree drops declarations used only inside property values or text.
    serialized = ET.tostring(package, encoding="utf-8")
    emitted = dict(value for event, value in ET.iterparse(BytesIO(serialized), events=("start-ns",)))
    for prefix, uri in declarations:
        if prefix not in emitted:
            package.set("xmlns" + (":" + prefix if prefix else ""), uri)
    opf_bytes = ET.tostring(package, encoding="utf-8", xml_declaration=True)
    cover_bytes = ET.tostring(html, encoding="utf-8", xml_declaration=True)
    cover_path.parent.mkdir(parents=True, exist_ok=True)
    cover_path.write_bytes(cover_bytes)
    opf_path.write_bytes(opf_bytes)
    return pages


def finalize_pairing(root, package, plan, processed, tome):
    """Read actual spine resources and change only paired-chapter page sides."""
    opf_root = root / 'OEBPS'
    manifest = {item.get('id'): item for item in package.find('opf:manifest', NS)}
    groups = {group['id']: group for group in plan['groups']}
    pages = []
    for reference in package.find('opf:spine', NS):
        if reference.get('idref') == COVER_ID:
            continue
        item = manifest[reference.get('idref')]
        xhtml_path = (opf_root / item.get('href')).resolve()
        html = ET.parse(xhtml_path).getroot()
        viewport = None
        for meta in html.findall(f'.//{{{XHTML}}}meta'):
            if meta.get('name') == 'viewport':
                values = dict(re.findall(r'(width|height)\s*=\s*(\d+)', meta.get('content', '')))
                if 'width' in values and 'height' in values:
                    viewport = [int(values['width']), int(values['height'])]
        images = []
        for node in html.iter():
            if node.tag == f'{{{XHTML}}}img':
                href = node.get('src')
            elif node.tag == f'{{{SVG}}}image':
                href = node.get('href') or node.get('{http://www.w3.org/1999/xlink}href')
            else:
                continue
            image_path = (xhtml_path.parent / href).resolve()
            metadata = processed.get(image_path.name)
            if metadata is None:
                raise ValueError('Missing chapter pairing output for ' + image_path.name)
            images.append((image_path, metadata))
        if not images:
            raise ValueError('Chapter pairing body XHTML has no images')
        group_ids = {metadata['group_id'] for _, metadata in images}
        if len(group_ids) != 1:
            raise ValueError('Body XHTML crosses chapter pairing groups')
        group = groups[next(iter(group_ids))]
        if group['mode'] != 'native':
            if len(images) != 1:
                raise ValueError('Paired chapter pages require one image per XHTML')
            if group['mode'] == 'single':
                side = 'center'
            else:
                first = images[0][1]['source_id'] == group['source_ids'][0]
                side = 'right' if first == (plan['direction'] == 'rtl') else 'left'
            properties = [value for value in reference.get('properties', '').split()
                          if not value.removeprefix('rendition:').startswith('page-spread-')]
            reference.set('properties', ' '.join([*properties, 'rendition:page-spread-' + side]))
        sides = [side for side in ('left', 'right', 'center')
                 if any(value.removeprefix('rendition:') == 'page-spread-' + side
                        for value in reference.get('properties', '').split())]
        if len(sides) != 1:
            raise ValueError('Body page requires one final page-spread side')
        for image_path, metadata in images:
            with Image.open(image_path) as image:
                image_size = list(image.size)
            pages.append(dict(
                source_id=metadata['source_id'], group_id=metadata['group_id'],
                image=image_path.relative_to(root).as_posix(), xhtml=xhtml_path.relative_to(root).as_posix(),
                tome=tome, side=sides[0], image_size=image_size, viewport=viewport,
                geometry=metadata['geometry']))
    return pages
