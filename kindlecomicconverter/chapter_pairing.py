"""Chapter-local spread detection and common geometry, without output side effects.

Sampling/scoring adapted from Mihon Desktop SharedSpreadMatching at
3d7652e30698508aa0c17798104037708d879b02 (Apache-2.0).
Modified: Python/Pillow sampling, chapter first-match policy and geometry.
See third_party/mihon-desktop-NOTICE.txt and Apache-2.0.txt.
"""

from fractions import Fraction
from math import ceil, floor

from PIL import Image


SAMPLE_POINTS = 50


def sample_edges(image):
    """Return 50 white-composited RGB samples per edge in source orientation."""
    width, height = image.size
    edges = []
    for x in (min(2, width - 1), max(0, width - 3)):
        if 'I;16' in image.mode:
            edges.append([(image.getpixel((x, i * height // SAMPLE_POINTS)) // 256,) * 3
                          for i in range(SAMPLE_POINTS)])
            continue
        with image.crop((x, 0, x + 1, height)).convert('RGBA') as column:
            edge = []
            for i in range(SAMPLE_POINTS):
                *rgb, alpha = column.getpixel((0, i * height // SAMPLE_POINTS))
                edge.append(tuple((c * alpha + 255 * (255 - alpha) + 127) // 255 for c in rgb))
            edges.append(edge)
    return tuple(edges)


def score_edges(earlier, later, direction):
    """Return RGB distance (<30), or None for an unqualified adjacent edge."""
    if direction not in ('rtl', 'ltr'):
        raise ValueError('Pairing direction must be rtl or ltr')
    first, second = (earlier[0], later[1]) if direction == 'rtl' else (earlier[1], later[0])
    if len(first) != SAMPLE_POINTS or len(second) != SAMPLE_POINTS:
        raise ValueError('Pairing requires 50 samples per edge')
    total = 0
    for a, b in zip(first, second):
        total += sum(abs(x - y) for x, y in zip(a, b))
        if total >= 30 * SAMPLE_POINTS:
            return None
    if all(sum(min(rgb) > 220 for rgb in edge) >= 40 for edge in (first, second)):
        return None
    if all(sum(max(rgb) < 50 for rgb in edge) >= 40 for edge in (first, second)):
        return None
    def variance(edge):
        values = [.299 * r + .587 * g + .114 * b for r, g, b in edge]
        mean = sum(values) / SAMPLE_POINTS
        return sum((value - mean) ** 2 for value in values) / SAMPLE_POINTS
    if variance(first) < 100 and variance(second) < 100:
        return None
    return total / SAMPLE_POINTS


def analyze_chapters(chapters, target, direction, spreadshift=False):
    """Build an analyzing v1 report from (title, [(source_id, path), ...]).

    Caller excludes the independent cover and supplies CBZ chapter boundaries.
    Only headers and the current adjacent pair's small samples are retained.
    """
    if direction not in ('rtl', 'ltr'):
        raise ValueError('Pairing direction must be rtl or ltr')
    ordered = [(title, sorted(pages)) for title, pages in chapters]
    if any(not pages for _, pages in ordered):
        raise ValueError('Pairing chapters must contain pages')
    ordered.sort(key=lambda chapter: chapter[1][0][0])
    source_ids = [source_id for _, pages in ordered for source_id, _ in pages]
    if any(type(source_id) is not int or source_id < 1 for source_id in source_ids):
        raise ValueError('Source IDs must be positive integers')
    if source_ids != sorted(set(source_ids)):
        raise ValueError('Chapter source ranges must be ordered and disjoint')
    report = dict(version=1, state='analyzing', target=list(target), direction=direction,
                  chapters=[], groups=[], pages=[])
    for chapter_id, (title, pages) in enumerate(ordered, 1):
        wide = False
        for _, path in pages:
            with Image.open(path) as image:
                wide |= image.width >= image.height
        shift, anchor = bool(spreadshift), None
        if not wide:
            previous = None
            for index, (source_id, path) in enumerate(pages):
                with Image.open(path) as image:
                    current = sample_edges(image)
                if previous is not None:
                    score = score_edges(previous, current, direction)
                    if score is not None:
                        shift = bool((index - 1) % 2)
                        anchor = dict(source_ids=[pages[index - 1][0], source_id], score=score)
                        break
                previous = current
        report['chapters'].append(dict(
            id=chapter_id, title=title, source_ids=[page[0] for page in pages],
            mode='native' if wide else 'paired', spreadshift=shift,
            reason='wide_image' if wide else 'detected' if anchor else 'spreadshift', anchor=anchor))
        index = 0
        while index < len(pages):
            paired = not wide and not (shift and index == 0) and index + 1 < len(pages)
            count = 2 if paired else 1
            report['groups'].append(dict(
                id=len(report['groups']) + 1, chapter_id=chapter_id,
                source_ids=[page[0] for page in pages[index:index + count]],
                mode='native' if wide else 'paired' if paired else 'single'))
            index += count
    return report


def pair_geometry(sizes, crop_boxes, target, direction, upscale=True):
    """Return two v1 geometry dictionaries from accepted native crop candidates.

    Native crop mode, first-color exemption, margin and area limits must already
    be applied by the caller; this function preserves both horizontal candidates.
    """
    if direction not in ('rtl', 'ltr') or len(sizes) != 2 or len(crop_boxes) != 2:
        raise ValueError('Common geometry requires two pages and rtl/ltr direction')
    width, height = target
    if width < 1 or height < 1:
        raise ValueError('Target dimensions must be positive')
    boxes = []
    for (w, h), box in zip(sizes, crop_boxes):
        l, t, r, b = [Fraction(str(value)) for value in box]
        if not (0 <= l < r <= w and 0 <= t < b <= h):
            raise ValueError('Crop candidate must lie within the source image')
        boxes.append((l, t, r, b))
    top = min(box[1] / size[1] for box, size in zip(boxes, sizes))
    bottom = max(box[3] / size[1] for box, size in zip(boxes, sizes))
    crops = [[floor(box[0]), floor(top * h), ceil(box[2]), ceil(bottom * h)]
             for box, (_, h) in zip(boxes, sizes)]
    ratios = [Fraction(r - l, b - t) for l, t, r, b in crops]
    content_height = floor(min(height, width / max(ratios)))
    if not upscale:
        content_height = min(content_height, *(b - t for _, t, _, b in crops))
    if content_height < 1:
        raise ValueError('Pair cannot fit the target at a positive common height')
    geometries = []
    for index, (crop, ratio) in enumerate(zip(crops, ratios)):
        rendered_width = max(1, floor(ratio * content_height))
        left = (index == 0) == (direction == 'rtl')
        geometries.append(dict(crop_box=crop, rendered_size=[rendered_width, content_height],
                               offset=[0 if left else width - rendered_width,
                                       (height - content_height) // 2]))
    return geometries


def resize_method(crop_box, rendered_size):
    """Use LANCZOS if either dimension shrinks, otherwise BICUBIC."""
    l, t, r, b = crop_box
    return (Image.Resampling.LANCZOS if rendered_size[0] < r - l or rendered_size[1] < b - t
            else Image.Resampling.BICUBIC)
