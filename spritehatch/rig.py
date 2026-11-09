"""A 2D bone rig for one pixel sprite.

The sprite is cut into body parts (head, torso, upper and lower arms and legs)
using the pose skeleton the model was asked to follow. A pose rotates the
parts around their joints, parents first, the same way 2D game tools
(Spine, DragonBones) do it, and the parts are drawn back with plain nearest
pixel sampling so the art stays crisp.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# name, parent, start point (the joint it turns around), end point, thickness
# as a share of body height. Numbers are OpenPose keypoints; strings are points
# worked out below.
BONES = [
    ("hips", None, "hip", "hip", 0.09),
    ("torso", "hips", "hip", 1, 0.15),
    ("head", "torso", 1, "top", 0.24),
    ("arm_r", "torso", 2, 3, 0.06),
    ("hand_r", "arm_r", 3, "wrist_r", 0.06),
    ("arm_l", "torso", 5, 6, 0.06),
    ("hand_l", "arm_l", 6, "wrist_l", 0.06),
    ("leg_r", "hips", 8, 9, 0.065),
    ("foot_r", "leg_r", 9, "sole_r", 0.07),
    ("leg_l", "hips", 11, 12, 0.065),
    ("foot_l", "leg_l", 12, "sole_l", 0.07),
]
NAMES = [b[0] for b in BONES]
CORE = ("hips", "torso", "head")
# Pixels farther from an arm or leg than this many of its thicknesses are
# clothes (a cape, a wide skirt) and stay on the body instead of swinging
# with the limb. None: every pixel goes to its nearest bone.
LIMB_REACH: float | None = 1.5
# Arm and leg thickness per body type: a boss has massive arms, and with chibi
# thickness most of each arm stays on the torso while the bone swings away.
LIMB = {"chibi": 1.0, "boss": 1.6}
# Torso thickness per body type: a boss's chest is broad, and with chibi
# thickness the strip an arm covered is not filled, so raising the arms
# leaves ragged gaps down both sides.
TORSO = {"chibi": 1.0, "boss": 1.85}
# The pelvis is a band across both hip joints, not a single point.
SEGMENT = {"hips": (8, 11)}

# Back to front. Profiles face right: the character's right side is near us.
DRAW_ORDER = {
    "right": ["arm_l", "hand_l", "leg_l", "foot_l", "hips", "torso", "head",
              "leg_r", "foot_r", "arm_r", "hand_r"],
    "down": ["leg_r", "foot_r", "leg_l", "foot_l", "hips", "torso",
             "arm_r", "hand_r", "arm_l", "hand_l", "head"],
    "up": ["leg_r", "foot_r", "leg_l", "foot_l", "arm_r", "hand_r",
           "arm_l", "hand_l", "hips", "torso", "head"],
}


def _rot(deg: float, pivot) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    px, py = pivot
    return np.array([[c, -s, px - c * px + s * py],
                     [s, c, py - s * px - c * py],
                     [0, 0, 1]])


def _squash(sy: float, pivot) -> np.ndarray:
    """Scale height around a point, so a part gets shorter toward it."""
    return np.array([[1, 0, 0], [0, sy, pivot[1] * (1 - sy)], [0, 0, 1]], dtype=float)


def _move(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=float)


def _seg_dist(p: np.ndarray, a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-6), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _fill(tex: np.ndarray, holes: np.ndarray, steps: int = 12) -> None:
    """Paint `holes` with the colors of neighboring opaque pixels, in place."""
    todo = holes & (tex[..., 3] == 0)
    for _ in range(steps):
        if not todo.any():
            return
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            src = np.roll(tex, (dy, dx), axis=(0, 1))
            take = todo & (src[..., 3] > 0)
            tex[take] = src[take]
            todo &= ~take


class Rig:
    def __init__(self, sprite: Image.Image, skeleton, view: str, limb: float = 1.0,
                 torso: float = 1.0):
        """sprite: RGBA image; skeleton: 18 [x, y] points (or None) on it.

        view is the drawn view: right, down or up (left = mirrored right).
        limb scales how thick arms and legs are (a boss's are much thicker
        than a chibi's, see LIMB), torso the torso's (see TORSO).
        """
        self.view = view
        self.size = sprite.size
        a = np.asarray(sprite.convert("RGBA"))
        ys, xs = np.nonzero(a[..., 3] > 0)
        self.top, self.floor = ys.min(), ys.max()
        self.height = self.floor - self.top + 1
        self.pts = self._points(skeleton, xs)
        self.cx = self.pts["hip"][0]

        pix = np.stack([xs + 0.5, ys + 0.5], axis=1)
        scores = []
        below_hip = pix[:, 1] > self.pts["hip"][1]
        above_neck = pix[:, 1] < self.pts[1][1]
        for name, _, s, e, r in BONES:
            s, e = SEGMENT.get(name, (s, e))
            r = r * torso if name == "torso" else r if name in CORE else r * limb
            d = _seg_dist(pix, self.pts[s], self.pts[e]) / (r * self.height)
            if name == "head":
                d = np.where(above_neck, 0.0, np.inf)
            else:
                d = np.where(above_neck, np.inf, d)
            if name == "torso":
                # Below the belt it is hips or legs, so skirts move with them.
                d = np.where(below_hip, np.inf, d)
            elif name not in CORE and LIMB_REACH:
                d = np.where(d > LIMB_REACH, np.inf, d)
            scores.append(d)
        scores = np.stack(scores)
        label = scores.argmin(axis=0)

        self.parts = {}
        for i, name in enumerate(NAMES):
            tex = np.zeros_like(a)
            sel = label == i
            tex[ys[sel], xs[sel]] = a[ys[sel], xs[sel]]
            self.parts[name] = tex
        # Where a limb lay over the body, the body needs pixels of its own,
        # or a moving arm or leg would leave a hole behind.
        for core in ("torso", "hips"):
            ci = NAMES.index(core)
            inside = (scores[ci] < 1.0) & ~np.isin(label, [NAMES.index(n) for n in CORE])
            under = np.zeros(a.shape[:2], bool)
            under[ys[inside], xs[inside]] = True
            _fill(self.parts[core], under)
        self.alpha = a[..., 3] > 0
        self.joints = [(name, parent, s, r * self.height) for name, parent, s, _, r in BONES
                       if parent]

    def _points(self, skeleton, xs):
        p = {i: None if q is None else np.array(q, float) for i, q in enumerate(skeleton)}
        p["hip"] = (p[8] + p[11]) / 2
        face = [p[i] for i in (0, 14, 15, 16, 17) if p[i] is not None]
        head_x = np.mean([f[0] for f in face]) if face else p[1][0]
        p["top"] = np.array([head_x, self.top + (p[1][1] - self.top) * 0.3])
        for side, (e, w) in {"r": (3, 4), "l": (6, 7)}.items():
            p[f"wrist_{side}"] = p[w] + (p[w] - p[e]) * 0.4
        for side, (k, an) in {"r": (9, 10), "l": (12, 13)}.items():
            p[f"sole_{side}"] = np.array([p[an][0], max(p[an][1], self.floor)])
        return p

    # --- poses --------------------------------------------------------------

    def angles(self, pose: dict) -> dict:
        """Turn friendly pose numbers into screen rotations.

        Profiles: degrees toward the facing side (arms forward, torso leaning
        in). Front/back: degrees away from the body's middle (arms out).
        """
        out = {}
        for name, _, s, e, _ in BONES:
            v = pose.get(name, 0)
            if not v:
                continue
            start, end = self.pts[s], self.pts[e]
            down = end[1] > start[1]
            if name == "hips":
                out[name] = v
            elif self.view == "right":
                out[name] = -v if down else v
            elif name in CORE:
                out[name] = v
            else:
                left_side = (start[0] + end[0]) / 2 < self.cx
                out[name] = v * (1 if left_side else -1) * (1 if down else -1)
        return out

    def matrices(self, pose: dict) -> dict:
        rot = self.angles(pose)
        shift = dict(pose.get("shift", {}))
        # "lift" raises a bone by a share of the body height, so a step looks
        # the same at 48 and at 96 pixels.
        for name, share in pose.get("lift", {}).items():
            dx, dy = shift.get(name, (0, 0))
            shift[name] = (dx, dy - round(share * self.height))
        # "squash" shortens a bone toward its joint: a leg stepping toward or
        # away from the camera looks shorter, its foot higher, its top still
        # at the hip.
        squash = pose.get("squash", {})
        root = _move(*pose.get("root", (0, 0)))
        world = {}
        for name, parent, s, _, _ in BONES:
            local = (_rot(rot.get(name, 0), self.pts[s]) @ _move(*shift.get(name, (0, 0)))
                     @ _squash(squash.get(name, 1.0), self.pts[s]))
            world[name] = (world[parent] if parent else root) @ local
        if pose.get("ground", True):
            # Keep the lowest foot on the floor, like a real step.
            low = max(self._lowest(world, f) for f in ("foot_r", "foot_l"))
            fix = _move(0, self.floor - low)
            world = {k: fix @ m for k, m in world.items()}
        # The head rides on the neck but never turns: a 48px face turned even
        # 5-10 degrees breaks its eyes and outline into stairs. Whole pixels,
        # so it stays the exact drawing.
        neck = np.array([*self.pts[1], 1.0])
        x, y = (world["head"] @ neck)[:2]
        world["head"] = _move(round(x - neck[0]), round(y - neck[1]))
        return world

    def _lowest(self, world, part) -> float:
        ys, xs = np.nonzero(self.parts[part][..., 3] > 0)
        if len(ys) == 0:
            return self.floor
        pts = world[part] @ np.stack([xs + 0.5, ys + 0.5, np.ones(len(xs))])
        return pts[1].max() - 0.5

    def point(self, world: dict, bone: str, key) -> np.ndarray:
        x, y = self.pts[key]
        return (world[bone] @ np.array([x, y, 1.0]))[:2]

    def render(self, pose: dict):
        """Draw the pose. Returns the image and the bone matrices used."""
        world = self.matrices(pose)
        w, h = self.size
        out = np.zeros((h, w, 4), np.uint8)
        for name in DRAW_ORDER[self.view]:
            tex = self.parts[name]
            ys, xs = np.nonzero(tex[..., 3] > 0)
            if len(ys) == 0:
                continue
            m = world[name]
            corners = m @ np.array([[xs.min(), xs.max() + 1, xs.min(), xs.max() + 1],
                                    [ys.min(), ys.min(), ys.max() + 1, ys.max() + 1],
                                    [1, 1, 1, 1]])
            x0, y0 = np.floor(corners[:2].min(axis=1)).astype(int) - 1
            x1, y1 = np.ceil(corners[:2].max(axis=1)).astype(int) + 1
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            if x0 >= x1 or y0 >= y1:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            src = np.linalg.inv(m) @ np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
            sx, sy = np.floor(src[0]).astype(int), np.floor(src[1]).astype(int)
            ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
            tx, ty = gx.ravel()[ok].astype(int), gy.ravel()[ok].astype(int)
            color = tex[sy[ok], sx[ok]]
            vis = color[:, 3] > 0
            out[ty[vis], tx[vis]] = color[vis]
        self._close_joints(out, world)
        return Image.fromarray(out), world

    def _close_joints(self, out: np.ndarray, world: dict) -> None:
        """Fill the cracks that open where a bent part meets its parent.

        A crack is an empty pixel near a joint that, followed back through
        either of the two parts' movement, was body in the original sprite.
        """
        h, w = out.shape[:2]
        gy, gx = np.mgrid[0:h, 0:w]
        cx, cy = gx + 0.5, gy + 0.5
        crack = np.zeros((h, w), bool)
        empty = out[..., 3] == 0
        for name, parent, joint, radius in self.joints:
            x, y = self.point(world, name, joint)
            cand = empty & ((cx - x) ** 2 + (cy - y) ** 2 < (radius * 1.2) ** 2)
            if not cand.any():
                continue
            pts = np.stack([cx[cand], cy[cand], np.ones(cand.sum())])
            was_body = np.zeros(cand.sum(), bool)
            for m in (world[name], world[parent]):
                sx, sy = np.floor(np.linalg.inv(m) @ pts)[:2].astype(int)
                ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
                was_body[ok] |= self.alpha[sy[ok], sx[ok]]
            crack[gy[cand][was_body], gx[cand][was_body]] = True
        _fill(out, crack, steps=3)
