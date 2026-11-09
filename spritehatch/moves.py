"""Motions as joint poses, played on the bone rig.

A pose is a dict of bone -> degrees (see Rig.angles for what the numbers
mean), plus optional "shift" (bone -> pixels), "squash" (bone -> height
factor toward its joint), "lift" (bone -> share of the
body height to raise it), "root" and effect keys that
render() handles after the body is drawn:

    sword   draw the sword in the right hand
    blade   screen direction of that sword in degrees (0 right, -90 up);
            without it the blade follows the forearm
    trail   slash trail from the sword's previous position
    orb     (radius, travel) glowing orb between the hands, flying forward
    ring    magic ring at the feet with this alpha
    glow    blue tint on the body
    flash   white hit flash
    fall    tip the whole body over by this many degrees (clockwise; minus
            means backward for a profile)
    bounce  one pixel up, the body bouncing off the floor
    dark    grayed out (down for good)
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageOps

from . import effects as fx
from .rig import LIMB, TORSO, Rig

FORWARD = {"right": (1, 0), "down": (0, 1), "up": (0, -1)}
# Blade angle relative to the forearm, so a hanging arm holds it pointing ahead.
GRIP = {"right": -110, "down": 25, "up": -25}


def canvas_size(frame: int) -> int:
    """Room for a swung sword or a body lying flat."""
    return -(-int(frame * 1.5) // 8) * 8


def make_rig(still: Image.Image, skeleton, view: str, size: int, body: str = "chibi") -> Rig:
    dx, dy = (size - still.width) // 2, (size - still.height) // 2
    canvas = Image.new("RGBA", (size, size))
    canvas.alpha_composite(still.convert("RGBA"), (dx, dy))
    moved = [None if p is None else [p[0] + dx, p[1] + dy] for p in skeleton]
    return Rig(canvas, moved, view, LIMB[body], TORSO[body])


def _side(view: str) -> bool:
    return view == "right"


# --- poses ------------------------------------------------------------------

def idle(view: str) -> list[dict]:
    breath = [0, 1, 1, 0]
    arms = [0, 2, 4, 2]
    return [{"shift": {"torso": (0, b)}, "arm_r": a, "arm_l": a} for b, a in zip(breath, arms)]


def walk(view: str) -> list[dict]:
    poses = []
    for i in range(8):
        # Start on a full stride, so frames 0, 2, 4, 6 are the four key poses
        # of a 4-frame walk: right step, passing, left step, passing.
        t = 2 * math.pi * i / 8 + math.pi / 2
        s, c = math.sin(t), math.cos(t)
        if _side(view):
            poses.append({
                "leg_r": 28 * s, "leg_l": -28 * s,
                # the knee bends while that leg swings forward
                "foot_r": -40 * max(0.0, c), "foot_l": -40 * max(0.0, -c),
                "arm_r": -24 * s, "arm_l": 24 * s, "hand_r": 18, "hand_l": 18,
                # the pelvis rocks with the stride, the chest turns against it
                "hips": 5 * s, "torso": 4 - 4 * s, "head": -2 + 1.5 * s,
            })
        else:
            lift_r, lift_l = max(0.0, s), max(0.0, -s)
            poses.append({
                # Seen from the front, a leg stepping toward the camera looks
                # shorter: its foot rises while its top stays under the skirt.
                # (Lifting the whole leg hid it behind the hips, and turning
                # hips and chest a few degrees tore holes in a 48 px body.)
                "squash": {"leg_r": 1 - 0.25 * lift_r, "leg_l": 1 - 0.25 * lift_l},
                # small swings: a cape hanging behind the arms moves with them
                "arm_r": 2 + 4 * lift_l, "arm_l": 2 + 4 * lift_r,
                "hand_r": 6, "hand_l": 6,
                # the body dips a pixel when a foot lands and rises as the legs pass
                "shift": {"torso": (0, 1 if abs(s) > 0.9 else 0)},
            })
    return poses


def attack(view: str) -> list[dict]:
    if _side(view):
        stance = {"leg_r": 12, "leg_l": -12}
        # Overhead chop: the hand goes up over the head (not across the face),
        # the blade drops behind the back, then comes over the top and ends
        # pointing ahead and down. The blade direction is set per frame:
        # following the forearm, a raised arm points it backward.
        keys = [
            {**stance, "arm_r": 30, "hand_r": 30, "arm_l": 15, "hand_l": 20},
            {**stance, "torso": -4, "head": -2, "arm_r": 205, "hand_r": 15, "arm_l": -10,
             "blade": -135},
            {**stance, "torso": -7, "head": -3, "arm_r": 230, "hand_r": 25, "arm_l": -25,
             "leg_r": 18, "foot_r": -15, "blade": 160},
            {"torso": 6, "arm_r": 105, "hand_r": 10, "arm_l": 20, "leg_r": 30, "foot_r": -5,
             "leg_l": -25, "foot_l": -20, "trail": True, "blade": -30},
            {"torso": 10, "head": 3, "arm_r": 70, "hand_r": 0, "arm_l": 35, "leg_r": 34,
             "foot_r": -8, "leg_l": -30, "foot_l": -25, "trail": True, "blade": 25},
            {"torso": 8, "head": 2, "arm_r": 40, "hand_r": -10, "arm_l": 30, "leg_r": 30,
             "leg_l": -26, "foot_l": -20, "blade": 55},
            {"torso": 3, "arm_r": 30, "hand_r": 20, "arm_l": 20, "leg_r": 20, "leg_l": -15},
            {**stance, "arm_r": 30, "hand_r": 30, "arm_l": 15, "hand_l": 20},
        ]
    else:
        # The sword goes up beside the head, on the sword arm's side; raised
        # over the head it crossed the face.
        keys = [
            {"arm_r": 20, "hand_r": 20},
            {"arm_r": 100, "hand_r": 20, "torso": -4, "arm_l": 10},
            {"arm_r": 125, "hand_r": 30, "torso": -7, "head": -3, "arm_l": 18},
            {"arm_r": 70, "hand_r": 0, "torso": 4, "arm_l": 8, "trail": True},
            {"arm_r": -35, "hand_r": -20, "torso": 8, "head": 3, "arm_l": 4, "trail": True},
            {"arm_r": -45, "hand_r": -25, "torso": 6, "head": 2},
            {"arm_r": -5, "hand_r": 5, "torso": 2},
            {"arm_r": 20, "hand_r": 20},
        ]
    for k in keys:
        k["sword"] = True
    return keys


def cast(view: str) -> list[dict]:
    if _side(view):
        keys = [
            {},
            {"arm_r": 40, "arm_l": 35, "hand_r": 20, "hand_l": 20, "torso": -4,
             "leg_r": 10, "leg_l": -10, "ring": 120},
            {"arm_r": 150, "arm_l": 140, "hand_r": 10, "hand_l": 10, "torso": -4, "head": -6,
             "leg_r": 10, "leg_l": -10, "ring": 255, "orb": (2, 0), "glow": 0.06},
            {"arm_r": 165, "arm_l": 155, "hand_r": 5, "hand_l": 5, "torso": -5, "head": -7,
             "leg_r": 10, "leg_l": -10, "ring": 255, "orb": (4, 0), "glow": 0.14},
            {"arm_r": 175, "arm_l": 165, "torso": -5, "head": -7, "leg_r": 10, "leg_l": -10,
             "ring": 255, "orb": (5, 0), "glow": 0.2},
            {"arm_r": 90, "arm_l": 85, "torso": 7, "head": 4, "leg_r": 25, "leg_l": -20,
             "foot_l": -15, "ring": 180, "orb": (4, 14), "glow": 0.08},
            {"arm_r": 80, "arm_l": 75, "torso": 6, "head": 3, "leg_r": 22, "leg_l": -18,
             "ring": 90, "orb": (3, 28)},
            {"arm_r": 30, "arm_l": 25, "torso": 3, "leg_r": 8, "leg_l": -8},
        ]
    else:
        keys = [
            {},
            {"arm_r": 30, "arm_l": 30, "ring": 120},
            {"arm_r": 100, "arm_l": 100, "hand_r": -20, "hand_l": -20, "head": 0,
             "ring": 255, "orb": (2, 0), "glow": 0.06},
            {"arm_r": 150, "arm_l": 150, "hand_r": -30, "hand_l": -30,
             "ring": 255, "orb": (4, 0), "glow": 0.14},
            {"arm_r": 165, "arm_l": 165, "hand_r": -35, "hand_l": -35,
             "ring": 255, "orb": (5, 0), "glow": 0.2},
            {"arm_r": 40, "arm_l": 40, "hand_r": -40, "hand_l": -40,
             "ring": 180, "orb": (4, 14), "glow": 0.08},
            {"arm_r": 30, "arm_l": 30, "hand_r": -30, "hand_l": -30, "ring": 90, "orb": (3, 28)},
            {"arm_r": 10, "arm_l": 10},
        ]
    return keys


def death(view: str) -> list[dict]:
    if _side(view):
        # Hit from the front: thrown back, knees give, falls flat on the back.
        recoil = {"torso": -10, "head": -12, "arm_r": 45, "arm_l": 35, "hand_r": 20,
                  "hand_l": 20, "root": (-1, 0)}
        sag = {"torso": -6, "head": -8, "arm_r": 60, "arm_l": 50, "hand_r": 25, "hand_l": 25,
               "leg_r": 20, "foot_r": -35, "leg_l": -5, "foot_l": -30, "root": (-3, 0)}
        # Lying nearly straight: bent limbs on a tipped 48px body read as a lump.
        flat = {"arm_r": 10, "arm_l": 10, "head": -5, "root": (-3, 0)}
        return [
            {**recoil, "flash": True},
            {**recoil, "torso": -14, "head": -16, "root": (-2, 0)},
            sag,
            # tipping over straight, see the front/back fall below
            {"arm_r": 10, "arm_l": 10, "head": -5, "fall": -30},
            {"arm_r": 10, "arm_l": 10, "head": -5, "fall": -60},
            {**flat, "fall": -90},
            {**flat, "fall": -90, "bounce": True},
            {**flat, "fall": -90, "dark": True},
        ]
    recoil = {"torso": -5, "head": -6, "arm_r": 30, "arm_l": 30, "hand_r": 25, "hand_l": 25}
    slump = {"shift": {"torso": (0, 3), "leg_r": (0, -2), "leg_l": (0, -2)},
             "arm_r": 15, "arm_l": 15, "head": 10, "torso": 4, "ground": False}
    return [
        {**recoil, "flash": True},
        {**recoil, "arm_r": 40, "arm_l": 40, "torso": -8, "head": -10},
        {**slump, "shift": {"torso": (0, 2), "leg_r": (0, -1), "leg_l": (0, -1)}},
        slump,
        # Tip over straight: a slumped 48px body turned 30-65 degrees breaks
        # into specks, a straight one keeps its face and outline.
        {"arm_r": 10, "arm_l": 10, "fall": 30},
        {"arm_r": 10, "arm_l": 10, "fall": 65},
        {"arm_r": 10, "arm_l": 10, "fall": 90},
        {"arm_r": 10, "arm_l": 10, "fall": 90, "dark": True},
    ]


# name: (poses, ms per frame, big canvas)
MOTIONS = {
    "idle": (idle, 220, False),
    "walk": (walk, 100, False),
    "attack": (attack, 70, True),
    "cast": (cast, 100, True),
    "death": (death, 110, True),
}


# --- playing ------------------------------------------------------------------

def _angle(a, b) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _fx_scale(rig: Rig) -> int:
    """Swords and orbs are made for a 48px body; a 96px boss gets them doubled."""
    return max(1, round(rig.height / 45))


def _sword_layer(rig: Rig, world: dict, view: str, state: dict, pose: dict) -> Image.Image:
    layer = Image.new("RGBA", rig.size)
    draw = ImageDraw.Draw(layer)
    hand = rig.point(world, "hand_r", "wrist_r")
    elbow = rig.point(world, "hand_r", 3)
    shoulder = rig.point(world, "arm_r", 2)
    length = rig.height * 0.34
    # "blade" fixes the screen direction (0 ahead for a profile, -90 up);
    # otherwise the sword follows the forearm.
    deg = pose["blade"] if "blade" in pose else _angle(elbow, hand) + GRIP[view]
    ux, uy = fx.unit(deg)
    tip = (hand[0] + ux * length, hand[1] + uy * length)
    tip_deg = _angle(shoulder, tip)
    reach = math.dist(shoulder, tip)
    if pose.get("trail") and "tip" in state:
        prev = state["tip"]
        turn = (tip_deg - prev + 180) % 360 - 180
        start = tip_deg - turn
        if view == "right" and turn > 0:
            # A chop coming over the top: the slash starts straight up, so it
            # never sweeps across the head.
            start = max(start, -90)
        fx.trail(draw, shoulder, start, tip_deg, reach * 0.7, reach + 1)
    fx.sword(layer, hand, deg, length, scale=_fx_scale(rig))
    state["tip"] = tip_deg
    return layer


def _magic(rig: Rig, world: dict, view: str, pose: dict, i: int):
    back = Image.new("RGBA", rig.size)
    front = Image.new("RGBA", rig.size)
    if "ring" in pose:
        foot = (rig.cx, rig.floor + 1)
        fx.magic_ring(ImageDraw.Draw(back), foot, rig.height * 0.7, i, pose["ring"])
    if "orb" in pose:
        k = _fx_scale(rig)
        radius, travel = (v * k for v in pose["orb"])
        hands = (rig.point(world, "hand_r", "wrist_r") + rig.point(world, "hand_l", "wrist_l")) / 2
        fxd, fyd = FORWARD[view]
        # The orb grows clear of the head: between raised hands it sat on the face.
        mark = rig.point(world, "head", "top")
        head_top = mark[1] - (rig.pts["top"][1] - rig.top)
        above = head_top - radius - 2 * k
        if view == "down":
            # Thrown at the camera: it swells above the head instead of
            # flying down across the face and body.
            radius += travel / 14
            center, travel = (mark[0], max(radius + 1, head_top - radius - 2 * k)), 0
        elif travel:
            center = (hands[0] + fxd * (3 * k + travel), hands[1] + fyd * travel - radius)
        else:
            x = mark[0] if view == "up" else hands[0] + fxd * 3 * k
            center = (x, min(hands[1] - radius, above))
        d = ImageDraw.Draw(back if view == "up" else front)
        if travel:
            d.line([(center[0] - fxd * travel, center[1] - fyd * travel), center],
                   fill=fx.MAGIC[2], width=2 * k)
        fx.orb(d, center, radius)
        fx.sparkles(d, center, radius + 4 * k, seed=i)
    return back, front


# OpenPose keypoint -> the bone whose movement carries it.
KEYPOINT_BONE = {0: "head", 14: "head", 15: "head", 16: "head", 17: "head",
                 1: "torso", 2: "torso", 5: "torso", 3: "arm_r", 4: "hand_r",
                 6: "arm_l", 7: "hand_l", 8: "hips", 9: "leg_r", 10: "foot_r",
                 11: "hips", 12: "leg_l", 13: "foot_l"}


def _turn(p, deg: float, pivot, shift) -> tuple:
    """Where point p goes when the frame turns `deg` clockwise around pivot
    and then moves by shift (what fx.rotate + fx.ground do to the pixels)."""
    a = math.radians(deg)
    dx, dy = p[0] - pivot[0], p[1] - pivot[1]
    return (pivot[0] + dx * math.cos(a) - dy * math.sin(a) + shift[0],
            pivot[1] + dx * math.sin(a) + dy * math.cos(a) + shift[1])


def with_effects(body: Image.Image, layers: tuple) -> Image.Image:
    back, front = layers
    return Image.alpha_composite(Image.alpha_composite(back, body), front)


def render(still: Image.Image, skeleton, direction: str, motion: str,
           keypoints: list | None = None, effects: list | None = None,
           body: str = "chibi") -> list[Image.Image]:
    """All frames of one motion for one direction (left is a mirrored right).

    Pass a list as `keypoints` to get, per frame, the 18 posed OpenPose points
    in frame pixels, tipped over with the body when it falls. Not available
    for "left".

    Pass a list as `effects` to get the body alone; the list gets, per frame,
    the (behind, in front) layers with the sword, slash and magic, for
    with_effects(). --redraw paints the body and adds them back crisp.
    Only for motions that do not fall over.
    """
    view = "right" if direction == "left" else direction
    make_poses, _, big = MOTIONS[motion]
    size = canvas_size(still.width) if big else still.width
    rig = make_rig(still, skeleton, view, size, body)
    state: dict = {}
    frames = []
    for i, pose in enumerate(make_poses(view)):
        body, world = rig.render(pose)
        pts = [None if p is None else tuple(rig.point(world, KEYPOINT_BONE[k], k))
               for k, p in enumerate(rig.pts[j] for j in range(18))]
        if pose.get("glow"):
            body = fx.tint(body, fx.MAGIC[1], pose["glow"])
        if pose.get("flash"):
            body = fx.tint(body, (255, 255, 255), 1.0)
        back, front = _magic(rig, world, view, pose, i)
        if pose.get("sword"):
            layer = _sword_layer(rig, world, view, state, pose)
            if view == "up":
                back = Image.alpha_composite(back, layer)
            else:
                front = Image.alpha_composite(front, layer)
        if effects is not None:
            effects.append((back, front))
        else:
            body = Image.alpha_composite(Image.alpha_composite(back, body), front)
        if pose.get("fall"):
            # Profiles fall on their face; front and back views topple sideways.
            # Turn around the body's middle: around the feet, a long body
            # swings its head out of the canvas and gets cut off. ground()
            # puts it back on the floor afterwards anyway.
            x0, y0, x1, y1 = fx.box(body)
            pivot = ((x0 + x1) / 2, (y0 + y1) / 2)
            body = fx.rotate(body, -pose["fall"], pivot)
            # Slide toward the middle while falling so the body stays in frame.
            x0, y0, x1, _ = fx.box(body)
            k = min(1.0, abs(pose["fall"]) / 90)
            center = (x0 + x1) / 2 * (1 - k) + size / 2 * k
            floor = rig.floor - (1 if pose.get("bounce") else 0)
            body = fx.ground(body, floor, center)
            nx0, ny0, _, _ = fx.box(body)
            # The skeleton tips over and slides with the body.
            pts = [None if p is None else
                   _turn(p, pose["fall"], pivot, (nx0 - x0, ny0 - y0)) for p in pts]
        if keypoints is not None:
            keypoints.append(pts)
        if pose.get("dark"):
            body = fx.darken(body)
        frames.append(body)
    if direction == "left":
        frames = [ImageOps.mirror(f) for f in frames]
    return frames
