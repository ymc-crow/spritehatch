"""Eyes, painted back after the shrink.

Every sprite pixel takes one color by vote over its block of the big drawing.
A 48px face is about 12x8 pixels and an eye 2x3, so the vote melts the
model's eyes into dark smudges: a highlight goes, an eye joins the dark hair
or helmet rim beside it, two eyes come out different sizes. The big drawing
has them clean, so find them there (dark blobs where the pose put the eyes)
and paint each one as a small pixel eye: pupil, iris along the bottom when the
eye has a colored iris, a highlight when it has one, and skin around it where
the vote let hair or rim run into it.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .pixelate import LUMA, _components
from .pose import keypoints

EYES = (14, 15)


def face_box(view: str, size: tuple[int, int], body: str = "chibi"):
    """Where to look for the eyes in a cell of `size` (w, h), and how many to
    find; None for a view without a face. The model puts the eyes a little
    lower and wider apart than the pose asks, so the box reaches well past
    the pose's eye points, mostly downward."""
    w, h = size
    pts = [p for k, p in enumerate(keypoints(view, body)) if k in EYES and p is not None]
    if not pts:
        return None
    xs = [p[0] * w for p in pts]
    ey = sum(p[1] for p in pts) / len(pts) * h
    gap = abs(xs[0] - xs[1]) if len(xs) == 2 else 0.1 * w
    box = (min(xs) - 1.5 * gap, ey - 0.4 * gap, max(xs) + 1.5 * gap, ey + 1.8 * gap)
    x0, y0, x1, y1 = (round(float(v)) for v in box)
    return (max(0, x0), max(0, y0), min(w, x1), min(h, y1)), len(pts)


def find_eyes(rgb: np.ndarray, box, count: int, min_size: int = 12) -> list:
    """Boxes (x0, y0, x1, y1) of `count` eyes inside `box` of the big drawing:
    blobs much darker than the skin around them that do not touch the box
    (those are hair, hood or helmet edges). Two eyes sit side by side at
    about the same height; the pair with the most dark pixels wins."""
    x0, y0, x1, y1 = box
    lum = rgb[y0:y1, x0:x1].astype(float) @ LUMA
    if lum.size == 0:
        return []
    dark = lum < 0.4 * np.median(lum)
    label, sizes = _components(dark)
    h, w = dark.shape
    blobs = []
    for k in range(1, len(sizes)):
        if sizes[k] < min_size:
            continue
        ys, xs = np.nonzero(label == k)
        if ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1:
            continue
        if ys.max() - ys.min() + 1 < 0.7 * (xs.max() - xs.min() + 1):
            continue                      # flat: a brow, a rim, not an eye
        blobs.append((int(sizes[k]), (xs.min() + x0, ys.min() + y0,
                                      xs.max() + 1 + x0, ys.max() + 1 + y0)))
    if count == 1:
        return [max(blobs)[1]] if blobs else []
    best = None
    for i, (sa, a) in enumerate(blobs):
        for sb, b in blobs[i + 1:]:
            level = abs((a[1] + a[3]) - (b[1] + b[3])) / 2 < 0.25 * h
            apart = a[2] <= b[0] or b[2] <= a[0]
            if level and apart and (best is None or sa + sb > best[0]):
                best = (sa + sb, sorted([a, b]))
    return [] if best is None else best[1]


def _eye_colors(rgb: np.ndarray, eye, skin_lum: float):
    """Pupil (the darkest fifth), iris (a strong color among the eye's own
    dark pixels, None for a plain dark eye: the soft brown where a black eye
    meets the skin is not an iris) and whether it has a white highlight."""
    x0, y0, x1, y1 = eye
    c = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(int)
    lum = c @ LUMA
    low = np.percentile(lum, 20)
    pupil = c[lum <= low].mean(0)
    sat = c.max(1) - c.min(1)
    colored = (lum > low) & (lum < 0.6 * skin_lum) & (sat > 50)
    iris = c[colored].mean(0) if colored.sum() >= 0.15 * len(c) else None
    shine = int((lum > 200).sum()) >= 3
    return pupil, iris, shine


def paint_eyes(sprite: Image.Image, rgb: np.ndarray, eyes: list, box) -> Image.Image:
    """Paint the eyes found in the big drawing `rgb` onto the sprite cut from
    it (sprite.info["place"] maps cell points onto the sprite)."""
    scale, x0, y0, left, top = sprite.info["place"]
    out = np.asarray(sprite.convert("RGBA")).copy()

    def to_sprite(x, y):
        return (x - x0) * scale + left, (y - y0) * scale + top

    # Skin: the most common sprite color in the face box, and its brightness
    # in the big drawing, to tell where the vote let a dark edge run in.
    bx0, by0 = to_sprite(box[0], box[1])
    bx1, by1 = to_sprite(box[2], box[3])
    face = out[int(by0):int(np.ceil(by1)), int(bx0):int(np.ceil(bx1))]
    solid = face[face[..., 3] > 0][:, :3]
    if not len(solid):
        return sprite
    # The skin is what lies between the two eyes (the bridge of the nose) or,
    # for one eye, just below it: the face box also holds hair and helmet,
    # often more of them than skin.
    a, b = eyes[0], eyes[-1]
    if len(eyes) == 2:
        patch = rgb[a[1]:a[3], a[2]:b[0]]
    else:
        patch = rgb[a[3]:a[3] + (a[3] - a[1]), a[0]:a[2]]
    if not patch.size:
        return sprite
    raw_skin = np.median(patch.reshape(-1, 3), axis=0)
    keys = np.unique(solid, axis=0)
    skin = keys[((keys.astype(float) - raw_skin) ** 2).sum(1).argmin()]
    skin_lum = float(raw_skin @ LUMA)

    # Both eyes are painted the same: one size, one set of colors (the model
    # draws a pair; small differences in the drawing would read as a squint).
    looks = [_eye_colors(rgb, eye, skin_lum) for eye in eyes]
    pupil = np.mean([p for p, _, _ in looks], axis=0).round().astype(np.uint8)
    irises = [i for _, i, _ in looks if i is not None]
    iris = (np.mean(irises, axis=0).round().astype(np.uint8)
            if len(irises) == len(looks) else None)
    shine = all(s for _, _, s in looks)
    ew = int(np.clip(round(np.mean([(e[2] - e[0]) * scale for e in eyes])), 1, 2))
    eh = int(np.clip(round(np.mean([(e[3] - e[1]) * scale for e in eyes])), 2, 3))
    # A pair seen from the front gets the pixel-art eye hand-made sprites use
    # (a copy of the model's dark oval reads as two dots): 2x3, a dark lash
    # row on top, a white catch light and the iris below. A plain black eye
    # gets a dark warm brown iris.
    centers = [((e[0] + e[2]) / 2 - x0) * scale for e in eyes]
    styled = len(eyes) == 2 and abs(centers[1] - centers[0]) >= 5
    if styled:
        ew, eh = 2, 3
        if iris is None:
            iris = (skin.astype(float) * 0.38).round().astype(np.uint8)
    # First wipe the smudges the vote made of the eyes: one left 2 pixels wide
    # beside a new 1-pixel eye made the pair look mismatched.
    for eye in eyes:
        sx0, sy0 = to_sprite(eye[0], eye[1])
        sx1, sy1 = to_sprite(eye[2], eye[3])
        ys, xs = slice(int(sy0), int(np.ceil(sy1))), slice(int(sx0), int(np.ceil(sx1)))
        spot = out[ys, xs]
        dark = (spot[..., 3] > 0) & (spot[..., :3] @ LUMA < 0.5 * (skin @ LUMA))
        spot[dark, :3] = skin
    for eye in eyes:
        sx0, sy0 = to_sprite(eye[0], eye[1])
        sx1, sy1 = to_sprite(eye[2], eye[3])
        cx = round(float((sx0 + sx1) / 2 - ew / 2))
        cy = round(float((sy0 + sy1) / 2 - eh / 2))
        # Skin back beside and under the eye where the big drawing has skin.
        for y in range(cy - 1, cy + eh + 1):
            for x in (cx - 1, cx + ew):
                _skin_back(out, rgb, x, y, scale, x0, y0, left, top, skin, skin_lum)
        for x in range(cx, cx + ew):
            _skin_back(out, rgb, x, cy + eh, scale, x0, y0, left, top, skin, skin_lum)
        out[cy:cy + eh, cx:cx + ew, :3] = pupil
        out[cy:cy + eh, cx:cx + ew, 3] = 255
        if styled:
            out[cy + 1:cy + 3, cx:cx + 2, :3] = iris
            out[cy + 1, cx, :3] = 255
            continue
        if iris is not None and eh >= 3:
            out[cy + eh - 1, cx:cx + ew, :3] = iris
        if shine and ew >= 2:
            out[cy, cx, :3] = 255
    mouth = find_mouth(rgb, box, eyes, skin_lum) if len(eyes) == 2 else None
    if mouth is not None:
        (mx0, my0, mx1, my1), color = mouth
        sx0, sy0 = to_sprite(mx0, my0)
        sx1, sy1 = to_sprite(mx1, my1)
        mw = int(np.clip(round((mx1 - mx0) * scale), 1, 2))
        mx = round(float((sx0 + sx1) / 2 - mw / 2))
        my = int((sy0 + sy1) / 2)
        if (out[my, mx:mx + mw, 3] > 0).all():
            out[my, mx:mx + mw, :3] = color
    result = Image.fromarray(out)
    result.info["place"] = sprite.info["place"]
    return result


def find_mouth(rgb: np.ndarray, box, eyes: list, skin_lum: float):
    """The mouth under and between the two eyes: a small blob darker or
    redder than the skin. Returns (its box, the color to paint it 1px tall:
    darkened until it shows against the skin) or None (a scarf or mask covers
    it, or the model drew none)."""
    (a, b) = sorted(eyes)
    x0, x1 = a[2], b[0]
    y0 = max(a[3], b[3]) + (max(a[3], b[3]) - min(a[1], b[1])) // 4
    y1 = box[3]
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    c = rgb[y0:y1, x0:x1].astype(int)
    lum = c @ LUMA
    # Against the skin right there (the face box also holds hair and hood):
    # a mouth is a little darker and redder, or much darker.
    redness = c[..., 0] - c[..., 1]
    local, skin_red = np.median(lum), np.median(redness)
    mark = ((lum < 0.92 * local) & (redness > skin_red + 25)) | (lum < 0.7 * local)
    label, sizes = _components(mark)
    h, w = mark.shape
    best = None
    for k in range(1, len(sizes)):
        ys, xs = np.nonzero(label == k)
        if sizes[k] < 6 or sizes[k] > 0.15 * mark.size:
            continue
        if xs.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1:
            continue
        off = abs((xs.min() + xs.max()) / 2 - w / 2)
        if best is None or off < best[0]:
            best = (off, k, (xs.min() + x0, ys.min() + y0, xs.max() + 1 + x0, ys.max() + 1 + y0))
    if best is None:
        return None
    color = c[label == best[1]].mean(0)
    while color @ LUMA > local - 60 and color.max() > 10:
        color = color * 0.85
    return best[2], color.round().astype(np.uint8)


def _skin_back(out, rgb, x, y, scale, x0, y0, left, top, skin, skin_lum) -> None:
    """Sprite pixel (x, y) is dark but its block of the big drawing is mostly
    skin: paint it skin."""
    h, w = out.shape[:2]
    if not (0 <= x < w and 0 <= y < h) or out[y, x, 3] == 0:
        return
    if out[y, x, :3] @ LUMA >= 0.5 * (skin @ LUMA):
        return
    bx0, by0 = int((x - left) / scale + x0), int((y - top) / scale + y0)
    bx1, by1 = int((x + 1 - left) / scale + x0), int((y + 1 - top) / scale + y0)
    block = rgb[max(by0, 0):by1, max(bx0, 0):bx1].reshape(-1, 3).astype(float) @ LUMA
    if len(block) and (block > 0.6 * skin_lum).mean() > 0.5:
        out[y, x, :3] = skin


def fix_faces(cells: list[Image.Image], sprites: list[Image.Image], views: list[str],
              body: str = "chibi") -> list[Image.Image]:
    """Paint the eyes back on every view that shows a face."""
    out = []
    for cell, sprite, view in zip(cells, sprites, views):
        found = face_box(view, cell.size, body)
        if found is None:
            out.append(sprite)
            continue
        box, count = found
        rgb = np.asarray(cell.convert("RGB"))
        eyes = find_eyes(rgb, box, count)
        out.append(paint_eyes(sprite, rgb, eyes, box) if len(eyes) == count else sprite)
    return out
