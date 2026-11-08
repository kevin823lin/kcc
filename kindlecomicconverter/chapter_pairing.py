"""Chapter-local spread detection and common geometry, without output side effects.

Sampling/scoring adapted from Mihon Desktop SharedSpreadMatching at
3d7652e30698508aa0c17798104037708d879b02 (Apache-2.0).
Modified: Python/Pillow sampling, independent opening/body detection and geometry.
See third_party/mihon-desktop-NOTICE.txt and Apache-2.0.txt.
"""

from fractions import Fraction
from math import ceil, floor
from statistics import correlation, pvariance

from PIL import Image


SAMPLE_POINTS = 50
STRIP_WIDTH = 8


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


def sample_strips(image, *, width=STRIP_WIDTH):
    """Return 50 mean grayscale samples from each inner edge strip."""
    image_width, height = image.size
    strips = []
    for box in ((0, 0, min(width, image_width), height),
                (max(0, image_width - width), 0, image_width, height)):
        strip = image.crop(box)
        if 'I;16' in image.mode:
            reduced = strip.resize((1, SAMPLE_POINTS), Image.Resampling.BOX)
            samples = [reduced.getpixel((0, y)) // 256 for y in range(SAMPLE_POINTS)]
        else:
            rgba = strip.convert('RGBA')
            white = Image.new('RGBA', rgba.size, 'white')
            reduced = Image.alpha_composite(white, rgba).convert('L').resize(
                (1, SAMPLE_POINTS), Image.Resampling.BOX)
            samples = [reduced.getpixel((0, y)) for y in range(SAMPLE_POINTS)]
        strips.append(samples)
    return tuple(strips)


def score_strips(earlier, later, direction, *, fallback=False):
    """Return full-strip Pearson correlation when non-margin bins corroborate it."""
    if direction not in ('rtl', 'ltr'):
        raise ValueError('Pairing direction must be rtl or ltr')
    first, second = (earlier[0], later[1]) if direction == 'rtl' else (earlier[1], later[0])
    if len(first) != SAMPLE_POINTS or len(second) != SAMPLE_POINTS:
        raise ValueError('Pairing requires 50 samples per strip')
    if (all(sum(v > 220 for v in strip) >= 40 for strip in (first, second))
            or all(sum(v < 50 for v in strip) >= 40 for strip in (first, second))):
        return None
    pairs = [(a, b) for a, b in zip(first, second) if 20 <= a <= 235 and 20 <= b <= 235]
    if len(pairs) < 15:
        return None

    if sum(abs(a - b) for a, b in zip(first, second)) > (25 if fallback else 16) * SAMPLE_POINTS:
        return None
    a, b = zip(*pairs)
    if pvariance(a) < 100 or pvariance(b) < 100 or correlation(a, b) < (.60 if fallback else .80):
        return None
    score = correlation(first, second)
    return min(1, score) if score >= (.82 if fallback else .93) else None


def score_edge_profiles(earlier, later, direction):
    """Match narrow contours only when their brightness changes corroborate them."""
    if direction not in ('rtl', 'ltr'):
        raise ValueError('Pairing direction must be rtl or ltr')
    first, second = (earlier[0], later[1]) if direction == 'rtl' else (earlier[1], later[0])
    if len(first) != SAMPLE_POINTS or len(second) != SAMPLE_POINTS:
        raise ValueError('Pairing requires 50 samples per edge profile')
    if sum(abs(a - b) for a, b in zip(first, second)) > 16 * SAMPLE_POINTS:
        return None
    changes = [[b - a for a, b in zip(profile, profile[1:])] for profile in (first, second)]
    active = [(index, a > 0) for index, (a, b) in enumerate(zip(*changes))
              if abs(a) >= 10 and abs(b) >= 10 and a * b > 0]
    # A single shared border or horizontal band is not independent artwork.
    if (len(active) < 4 or active[-1][0] - active[0][0] < SAMPLE_POINTS // 4
            or sum(a[1] != b[1] for a, b in zip(active, active[1:])) < 2):
        return None
    if min(pvariance(profile) for profile in (first, second, *changes)) < 100:
        return None
    if correlation(*changes) < .85:
        return None
    score = correlation(first, second)
    return min(1, score) if score >= .95 else None


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


def _find_anchor(pages, direction):
    """Find the first candidate in the strongest tier across these pages."""
    if len(pages) < 2:
        return None, None
    anchor = anchor_index = None
    previous = strip_previous = strip_anchor = strip_anchor_index = None
    fallback_anchor = fallback_anchor_index = None
    profile_previous = profile_anchor = profile_anchor_index = None
    for index, (source_id, path) in enumerate(pages):
        with Image.open(path) as image:
            current = sample_edges(image)
            current_strip = sample_strips(image) if strip_anchor is None else None
            current_profile = (sample_strips(image, width=1)
                               if strip_anchor is None and fallback_anchor is None
                               and profile_anchor is None else None)
        if previous is not None:
            score = score_edges(previous, current, direction)
            if score is not None:
                anchor_index = index - 1
                anchor = dict(source_ids=[pages[index - 1][0], source_id], score=score)
                break
            if strip_anchor is None:
                strip_score = score_strips(strip_previous, current_strip, direction)
                if strip_score is not None:
                    strip_anchor = dict(source_ids=[pages[index - 1][0], source_id],
                                        score=strip_score)
                    strip_anchor_index = index - 1
                elif fallback_anchor is None:
                    fallback_score = score_strips(strip_previous, current_strip, direction, fallback=True)
                    if fallback_score is not None:
                        fallback_anchor = dict(source_ids=[pages[index - 1][0], source_id],
                                               score=fallback_score, method='strip_fallback')
                        fallback_anchor_index = index - 1
            if strip_anchor is None and fallback_anchor is None and profile_anchor is None:
                profile_score = score_edge_profiles(profile_previous, current_profile, direction)
                if profile_score is not None:
                    profile_anchor = dict(source_ids=[pages[index - 1][0], source_id],
                                          score=profile_score, method='edge_profile')
                    profile_anchor_index = index - 1
        previous = current
        profile_previous = current_profile
        if strip_anchor is None:
            strip_previous = current_strip
    if anchor is None and strip_anchor is not None:
        anchor_index = strip_anchor_index
        anchor = dict(strip_anchor, method='strip')
    elif anchor is None and fallback_anchor is not None:
        anchor_index = fallback_anchor_index
        anchor = fallback_anchor
    elif anchor is None and profile_anchor is not None:
        anchor_index = profile_anchor_index
        anchor = profile_anchor
    return anchor, anchor_index


def analyze_chapters(chapters, target, direction, spreadshift=False):
    """Build an analyzing v3 report from (title, [(source_id, path), ...]).

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
    report = dict(version=3, state='analyzing', target=list(target), direction=direction,
                  chapters=[], groups=[], pages=[])
    for chapter_id, (title, pages) in enumerate(ordered, 1):
        wide = False
        for _, path in pages:
            with Image.open(path) as image:
                wide |= image.width >= image.height
        shift, anchor, opening_anchor = bool(spreadshift), None, None
        if not wide:
            opening_anchor, _ = _find_anchor(pages[:2], direction)
            body_start = 2 if opening_anchor else 1
            # shortcut: unmatched p2 is sampled twice; cache it if profiling shows a cost.
            anchor, anchor_index = _find_anchor(pages[body_start:], direction)
            if anchor is not None:
                shift = bool((body_start + anchor_index) % 2)
            elif opening_anchor is not None:
                shift = False
        report['chapters'].append(dict(
            id=chapter_id, title=title, source_ids=[page[0] for page in pages],
            mode='native' if wide else 'paired', spreadshift=shift,
            reason='wide_image' if wide else 'detected' if anchor else 'opening' if opening_anchor else 'spreadshift',
            opening_anchor=opening_anchor, anchor=anchor))
        index = 0
        while index < len(pages):
            paired = (not wide and not (shift and index == (2 if opening_anchor else 0))
                      and index + 1 < len(pages))
            count = 2 if paired else 1
            report['groups'].append(dict(
                id=len(report['groups']) + 1, chapter_id=chapter_id,
                source_ids=[page[0] for page in pages[index:index + count]],
                mode='native' if wide else 'paired' if paired else 'single'))
            index += count
    return report


def pair_geometry(sizes, crop_boxes, target, upscale=True):
    """Return two v2 geometry dictionaries from accepted native crop candidates.

    Native crop mode, first-color exemption, margin and area limits must already
    be applied by the caller; this function preserves both horizontal candidates.
    """
    if len(sizes) != 2 or len(crop_boxes) != 2:
        raise ValueError('Common geometry requires two pages')
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
    for crop, ratio in zip(crops, ratios):
        rendered_width = max(1, floor(ratio * content_height))
        geometries.append(dict(crop_box=crop, rendered_size=[rendered_width, content_height]))
    return geometries


def resize_method(crop_box, rendered_size):
    """Use LANCZOS if either dimension shrinks, otherwise BICUBIC."""
    l, t, r, b = crop_box
    return (Image.Resampling.LANCZOS if rendered_size[0] < r - l or rendered_size[1] < b - t
            else Image.Resampling.BICUBIC)
