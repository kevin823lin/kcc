"""Finish unpacked EPUB metadata and cover before KCC creates the ZIP."""

from io import BytesIO
from pathlib import Path
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


def finalize_epub(path, series_name, series_index):
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
