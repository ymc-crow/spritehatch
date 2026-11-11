"""Motions for creatures (spritehatch --creature): no skeleton to bend, so the
whole sprite squashes, stretches, hops and tips over, the way slimes and
small monsters move in hand-made games.

idle:   slow squash and stretch
walk:   a hop: squash, spring up, land
attack: lean back, lunge forward with a flash, come back
death:  hit flash, squash, tip over, lie gray

Winged creatures (the top half much wider than the body below) also beat
their wings in idle and walk (flap), and profiles lift their tail (wag).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import effects as fx
from .moves import canvas_size

FORWARD = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1)}
WHITE = (255, 255, 255)


def _place(still: Image.Image, size: int, sx: float = 1.0, sy: float = 1.0,
           dx: float = 0.0, dy: float = 0.0) -> Image.Image:
    """The still scaled by (sx, sy) around its feet and moved by (dx, dy),
    on a size x size canvas with the feet where they were."""
    x0, y0, x1, y1 = fx.box(still)
    body = still.crop((x0, y0, x1, y1))
    w, h = max(1, round(body.width * sx)), max(1, round(body.height * sy))
    body = body.resize((w, h), Image.NEAREST)
    pad_x, pad_y = (size - still.width) // 2, (size - still.height) // 2
    feet_x, feet_y = pad_x + (x0 + x1) / 2, pad_y + y1
    out = Image.new("RGBA", (size, size))
    out.alpha_composite(body, (round(feet_x - w / 2 + dx), round(feet_y - h + dy)))
    return out


def _waist(solid: np.ndarray):
    """(top row, height, narrowest lower-body row index, its first and last
    column): the body the wings and tail stick out of."""
    ys, _ = np.nonzero(solid)
    top, bottom = ys.min(), ys.max()
    h = bottom - top + 1
    rows = [(y, np.nonzero(solid[y])[0]) for y in range(top + int(0.6 * h), top + int(0.9 * h))]
    rows = [(y, r) for y, r in rows if len(r)]
    if not rows:
        return None
    y, r = min(rows, key=lambda yr: yr[1][-1] - yr[1][0])
    return top, h, y, r[0], r[-1]


def _wings(solid: np.ndarray):
    """(center x, half body width, shoulder row) when the top half spreads much
    wider than the narrowest row of the lower body (wings), else None."""
    found = _waist(solid)
    if found is None:
        return None
    top, h, _, x0, x1 = found
    body = x1 - x0 + 1
    upper = [np.ptp(np.nonzero(solid[y])[0]) + 1 if solid[y].any() else 0
             for y in range(top, top + h // 2)]
    if max(upper) < 1.3 * body:
        return None
    shoulder = top + max(i for i, w in enumerate(upper) if w >= max(upper) - 1)
    return (x0 + x1 + 1) / 2, body / 2, shoulder


def _lift(a: np.ndarray, rows: range, cx: float, half: float, view: str,
          amount: float) -> np.ndarray:
    """Slide every column outside the body (|x - cx| > half) up by amount x
    (distance past the body)^2 / span, within `rows`: whole columns move, so
    no holes open. Profiles only move the side behind the back (the head is
    on the other side)."""
    out = a.copy()
    span = max(cx, a.shape[1] - cx) - half
    idx = np.arange(rows.start, rows.stop)
    for x in range(a.shape[1]):
        side = x + 0.5 - cx
        if (view == "right" and side > 0) or (view == "left" and side < 0):
            continue
        o = abs(side) - half
        d = round(amount * o * o / span) if o > 0 else 0
        if d:
            src = idx + d
            ok = (src >= rows.start) & (src < rows.stop)
            col = a[np.clip(src, rows.start, rows.stop - 1), x].copy()
            col[~ok] = 0
            out[rows.start:rows.stop, x] = col
    return out


def flap(still: Image.Image, view: str, amount: float) -> Image.Image:
    """Beat the wings: every column outside the body, above the shoulder row,
    slides down (amount < 0, folding) or up, more toward the wing tips."""
    a = np.asarray(still.convert("RGBA"))
    found = _wings(a[..., 3] > 0)
    if found is None or not amount:
        return still
    cx, half, shoulder = found
    return Image.fromarray(_lift(a, range(shoulder + 1), cx, half, _profile(still, view),
                                 amount))


def wag(still: Image.Image, view: str, amount: float) -> Image.Image:
    """Lift the tail in a profile: what sticks out behind the body below the
    waist rises, more toward the tip (it rests on the floor, so only up).
    Front and back views keep it still."""
    a = np.asarray(still.convert("RGBA"))
    found = _waist(a[..., 3] > 0)
    if view not in ("left", "right") or found is None or not amount:
        return still
    _, _, waist, x0, x1 = found
    return Image.fromarray(_lift(a, range(waist, a.shape[0]), (x0 + x1 + 1) / 2,
                                 (x1 - x0 + 1) / 2, _profile(still, view), amount))


def _profile(still: Image.Image, view: str) -> str:
    """The view a profile really shows ("right" when the head is on the
    right), so wings and tail move on the back side, never the head."""
    if view not in ("left", "right"):
        return view
    return "right" if facing(still, view)[0] > 0 else "left"


def facing(still: Image.Image, view: str) -> tuple[int, int]:
    """Which way a view moves forward (x, y). A profile goes where the head
    is, not where the view's name says: the model sometimes draws the right
    view facing left, and the lunge went out of its back. The head side has
    more of the top half (head, ears, horns), the back side more of the
    bottom (tail, hind legs); a round body that shows neither keeps the
    view's own way."""
    fx_, fy = FORWARD[view]
    if not fx_:
        return fx_, fy
    a = np.asarray(still.convert("RGBA"))[..., 3] > 0
    ys, xs = np.nonzero(a)
    if not len(xs):
        return fx_, fy
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w, mid = y1 - y0 + 1, x1 - x0 + 1, (x0 + x1) / 2
    top = xs[ys < y0 + h * 0.5].mean() - mid
    bottom = xs[ys >= y0 + h * 0.7].mean() - mid
    lean = (top - bottom) / w
    return (fx_ if abs(lean) < 0.02 else (1 if lean > 0 else -1)), 0


def idle(still: Image.Image, view: str) -> list[Image.Image]:
    s = still.width
    beat = (0, -0.1, -0.2, -0.1)            # wings sink a little as it breathes
    tail = (0, 0.1, 0.2, 0.1)
    return [_place(wag(flap(still, view, b), view, t), s, *k)
            for b, t, k in zip(beat, tail, ((1, 1), (1.04, 0.96), (1, 1), (0.97, 1.03)))]


def walk(still: Image.Image, view: str) -> list[Image.Image]:
    s = canvas_size(still.width)          # room to hop
    fx_, fy = facing(still, view)
    k = still.width / 48                  # hop sizes are for a 48px frame
    hop = [  # (sx, sy, forward, up)
        (1.08, 0.9, 0, 0),
        (0.94, 1.08, 1, 3),
        (1.0, 1.0, 2, 5),
        (0.96, 1.05, 1, 2),
    ]
    beat = (-0.3, 0.1, 0.15, -0.1)          # wings fold on landing, open in the air
    tail = (0, 0.2, 0.25, 0.1)              # the tail swings up in the air
    return [_place(wag(flap(still, view, b), view, t), s, sx, sy, fx_ * f * k,
                   (fy * f - up) * k)
            for b, t, (sx, sy, f, up) in zip(beat, tail, hop)]


def attack(still: Image.Image, view: str) -> list[Image.Image]:
    s = canvas_size(still.width)
    fx_, fy = facing(still, view)
    reach = still.width // 8

    def at(sx, sy, f):
        return _place(still, s, sx, sy, fx_ * f, fy * f)
    lunge = at(1.1, 0.92, reach) if fx_ else at(1.06, 1.06, reach)
    return [
        at(1.0, 1.0, 0),
        at(1.06, 0.92, -reach / 3),       # lean back
        lunge,
        fx.tint(lunge, WHITE, 0.5),       # the hit
        at(1.0, 1.0, reach / 2),
        at(1.0, 1.0, 0),
    ]


def death(still: Image.Image, view: str) -> list[Image.Image]:
    s = canvas_size(still.width)
    fx_, _ = facing(still, view)
    back = -fx_ if fx_ else 1             # falls away from where it faces
    stand = _place(still, s)
    floor = fx.box(stand)[3]
    x0, _, x1, _ = fx.box(stand)
    pivot = ((x0 + x1) / 2, floor)
    out = [fx.tint(stand, WHITE, 0.8), _place(still, s, 1.1, 0.85, -back * 2 * still.width / 48)]
    # Turned on a canvas twice as big, so the body lying flat is not cut off,
    # then put back on the floor in the middle.
    m = s // 2
    big = Image.new("RGBA", (2 * s, 2 * s))
    big.alpha_composite(stand, (m, m))
    for deg in (35, 70, 90):
        # Positive turns counterclockwise: the head goes left.
        lying = fx.ground(fx.rotate(big, -deg * back, (pivot[0] + m, pivot[1] + m)),
                          floor + m, s)
        out.append(lying.crop((m, m, m + s, m + s)))
    out.append(fx.darken(out[-1], 0.5))
    return out


def _legs(solid: np.ndarray):
    """(top row, floor row, first and last column) of the legs: going up from
    the feet, the rows where the body splits into separate runs, limited to
    the width of the body just above (a tail lying low is not a leg)."""
    ys = np.nonzero(solid.any(1))[0]
    floor, top = ys.max(), ys.max()

    def runs(row):
        return int((np.diff(np.concatenate([[0], row.astype(int), [0]])) == 1).sum())
    while top > ys.min() and runs(solid[top - 1]) >= 2:
        top -= 1
    if floor - top < 3:
        return None
    above = np.nonzero(solid[max(top - 1, 0)])[0]
    return top, floor, above.min(), above.max()


def stride(still: Image.Image, view: str, amount: float) -> Image.Image:
    """Move the legs of a four-legged creature. Profiles: front and hind legs
    swing opposite ways, the feet most (amount in pixels at the feet). Front
    and back views: one side's feet lift (amount > 0 the left side)."""
    a = np.asarray(still.convert("RGBA"))
    found = _legs(a[..., 3] > 0)
    if found is None or not amount:
        return still
    top, floor, x0, x1 = found
    out = a.copy()
    zone = a[top:floor + 1, x0:x1 + 1].copy()
    out[top:floor + 1, x0:x1 + 1] = 0
    mid = (x1 - x0) / 2
    for r in range(zone.shape[0]):
        k = (r + 1) / zone.shape[0]                 # feet move most, hips stay
        for c in np.nonzero(zone[r, :, 3])[0]:
            y, x = top + r, x0 + c
            if view in ("left", "right"):
                x += round(amount * k * (1 if c > mid else -1))
            elif (c < mid) == (amount > 0):
                y -= round(abs(amount) * k)
            if 0 <= x < a.shape[1] and 0 <= y < a.shape[0]:
                out[y, x] = zone[r, c]
    return Image.fromarray(out)


def trot(still: Image.Image, view: str) -> list[Image.Image]:
    """Four-legged walk: legs swing (profiles) or step left and right (front
    and back) and the body bobs a little, instead of hopping whole."""
    s = canvas_size(still.width)
    k = still.width / 48
    steps = (3, 0, -3, 0)                           # pixels at the feet, 48px frame
    bob = (0, 1, 0, 1)                              # up while the legs pass
    beat = (-0.1, 0.05, -0.1, 0.05)
    tail = (0.1, 0.2, 0.1, 0.2)
    return [_place(stride(wag(flap(still, view, f), view, t), view, st * k), s,
                   dy=-b * round(k))
            for st, b, f, t in zip(steps, bob, beat, tail)]


MOTIONS = {
    "idle": (idle, 220, False),
    "walk": (walk, 110, True),
    "attack": (attack, 80, True),
    "death": (death, 120, True),
}


def render(still: Image.Image, direction: str, motion: str,
           quadruped: bool = False) -> list[Image.Image]:
    """Frames of one motion; four-legged creatures walk on their legs."""
    if motion == "walk" and quadruped:
        return trot(still.convert("RGBA"), direction)
    return MOTIONS[motion][0](still.convert("RGBA"), direction)
