import numpy as np
from PIL import Image, ImageOps

from spritehatch.pixelate import background_mask, downscale_sprites, snap_to_grid, to_sprites
from spritehatch.pose import MODES, keypoints, pose_sheet


def test_every_drawn_view_has_a_skeleton():
    for spec in MODES.values():
        for view in spec["draw"]:
            assert len(keypoints(view)) == 18
        img = pose_sheet(640, 320, spec["draw"])
        assert np.asarray(img).max() > 0


def test_back_view_has_no_face():
    back = keypoints("up")
    assert back[0] is None and back[14] is None and back[15] is None


def _figure(height, top=10):
    """White cell with a dark box 'character' that has a white spot inside."""
    a = np.full((120, 80, 3), 255, np.uint8)
    a[top:top + height, 30:50] = (40, 40, 160)
    a[top + 5:top + 8, 38:42] = 255
    return Image.fromarray(a)


def test_background_mask_keeps_white_inside_character():
    mask = background_mask(_figure(60))
    assert mask[0, 0]
    assert not mask[16, 39]


def test_sprites_share_scale_and_feet_line():
    sprites = to_sprites([_figure(100), _figure(50, top=40)], size=32)
    alphas = [np.asarray(s)[..., 3] > 0 for s in sprites]
    heights = [np.nonzero(a.any(axis=1))[0] for a in alphas]
    assert heights[0][-1] == heights[1][-1]
    assert len(heights[0]) > len(heights[1])


def test_downscale_keeps_scale_feet_line_eyes_and_adds_outline():
    big = _figure(100)
    a = np.asarray(big).copy()
    a[30:34, 34:37] = (10, 10, 10)          # a small dark "eye" on the body
    sprites = downscale_sprites([Image.fromarray(a), _figure(50, top=40)], size=32, colors=8)
    assert all(s.size == (32, 32) for s in sprites)
    rgba = [np.asarray(s) for s in sprites]
    bottoms = [np.nonzero((r[..., 3] > 0).any(axis=1))[0][-1] for r in rgba]
    assert bottoms[0] == bottoms[1]
    first = rgba[0]
    solid = first[..., 3] > 0
    left_col = np.nonzero(solid.any(axis=0))[0][0]
    ys = np.nonzero(solid[:, left_col])[0]
    assert (first[ys, left_col, :3].sum(axis=1) < 150).all()   # dark outline on the left edge
    lum = first[..., :3].astype(int).sum(axis=2)
    right_col = np.nonzero(solid.any(axis=0))[0][-1]
    inside = ((lum < 100) & solid)[ys[0] + 1:ys[-1], left_col + 1:right_col]
    assert inside.any()                                        # the eye survived the shrink


def test_zimage_turnaround_is_text_to_image():
    from spritehatch.comfy import zimage_turnaround_workflow

    wf = zimage_turnaround_workflow("knight", "pose.png", ["front view"], seed=3)
    assert wf["sample"]["inputs"]["denoise"] == 1.0
    assert wf["enc"]["class_type"] == "EmptySD3LatentImage"
    assert "knight" in wf["pos"]["inputs"]["text"]


def test_snap_to_grid_recovers_fake_pixels():
    rng = np.random.default_rng(0)
    small = rng.integers(0, 255, (20, 30, 3), dtype=np.uint8)
    big = Image.fromarray(small).resize((240, 160), Image.NEAREST)
    snapped = np.asarray(snap_to_grid(big))
    assert snapped.shape[0] >= 18 and snapped.shape[1] >= 28



def _stick_figure(view="down", size=80):
    """A plain sprite drawn along the pose skeleton, plus that skeleton."""
    from PIL import ImageDraw

    from spritehatch.pose import LIMBS, keypoints

    pts = [None if p is None else [p[0] * size, p[1] * size] for p in keypoints(view)]
    img = Image.new("RGBA", (size, size))
    d = ImageDraw.Draw(img)
    for a, b in LIMBS[:12]:
        if pts[a - 1] and pts[b - 1]:
            d.line([tuple(pts[a - 1]), tuple(pts[b - 1])], fill=(200, 60, 60, 255), width=5)
    neck = pts[1]
    d.ellipse((neck[0] - 9, neck[1] - 20, neck[0] + 9, neck[1] - 2), fill=(90, 90, 200, 255))
    return img, pts


def test_rig_parts_cover_the_sprite_and_rest_pose_is_unchanged():
    from spritehatch.rig import Rig

    img, pts = _stick_figure()
    rig = Rig(img, pts, "down")
    alpha = np.asarray(img)[..., 3] > 0
    cover = sum((p[..., 3] > 0).astype(int) for n, p in rig.parts.items() if n != "torso")
    torso = rig.parts["torso"][..., 3] > 0
    assert ((cover + torso) >= alpha).all()
    still, _ = rig.render({})
    assert np.array_equal(np.asarray(still), np.asarray(img))


def test_rig_bends_joints_and_keeps_feet_on_floor():
    from spritehatch.rig import Rig

    img, pts = _stick_figure("right")
    rig = Rig(img, pts, "right")
    moved, _ = rig.render({"leg_r": 30, "foot_r": -40, "arm_r": 90})
    a = np.asarray(moved)[..., 3] > 0
    assert not np.array_equal(a, np.asarray(img)[..., 3] > 0)
    assert np.nonzero(a.any(axis=1))[0][-1] == rig.floor


def test_every_motion_renders_in_every_direction():
    from spritehatch.moves import MOTIONS, canvas_size, render

    img, pts = _stick_figure()
    side, side_pts = _stick_figure("right")
    for name, (_, _, big) in MOTIONS.items():
        for d in ("down", "left", "right", "up"):
            still, skel = (side, side_pts) if d in ("left", "right") else (img, pts)
            frames = render(still, skel, d, name)
            assert len(frames) >= 4
            size = canvas_size(80) if big else 80
            assert all(f.size == (size, size) for f in frames)
    lying = render(side, side_pts, "right", "death")[-1]
    ys, xs = np.nonzero(np.asarray(lying)[..., 3] > 0)
    assert np.ptp(xs) > np.ptp(ys)


def test_render_reports_posed_keypoints_that_follow_the_limbs():
    from spritehatch.moves import render

    side, side_pts = _stick_figure("right")
    pts: list = []
    frames = render(side, side_pts, "right", "walk", keypoints=pts)
    assert len(pts) == len(frames)
    ankles = {round(p[10][0]) for p in pts}
    assert len(ankles) > 1          # the right ankle moves through the stride
    assert all(len(p) == 18 for p in pts)
    dead: list = []
    render(side, side_pts, "right", "death", keypoints=dead)
    neck, ankle = dead[-1][1], dead[-1][10]
    assert abs(neck[1] - ankle[1]) < abs(neck[0] - ankle[0])   # the skeleton lies flat too


class _FakeComfy:
    """Stands in for ComfyUI: 'paints' by returning the draft unchanged."""

    def __init__(self):
        self.uploads, self.runs = {}, 0

    def upload_image(self, img, name):
        self.uploads[name] = img
        return name

    def run(self, wf):
        self.runs += 1
        assert wf["sample"]["inputs"]["latent_image"] == ["enc", 0]
        return [self.uploads[wf["init"]["inputs"]["image"]]]


def test_redraw_paints_four_frames_per_grid_and_keeps_size_and_palette():
    from spritehatch.moves import render
    from spritehatch.redraw import build_grids, cell_size, redraw

    side, side_pts = _stick_figure("right")
    pts: list = []
    frames = render(side, side_pts, "right", "walk", keypoints=pts)
    draft, pose = build_grids(frames[:4], pts[:4])
    assert draft.size == pose.size == (cell_size(80) * 2,) * 2
    assert np.asarray(pose).any()

    fake = _FakeComfy()
    out = redraw(frames, pts, still=side, character="knight", view="right",
                 motion="walk", client=fake)
    assert fake.runs == 2           # 8 frames -> two 2x2 grids
    assert [f.size for f in out] == [f.size for f in frames]
    src = {tuple(c) for c in np.asarray(side)[np.asarray(side)[..., 3] > 0][:, :3]}
    for f in out:
        a = np.asarray(f)
        used = {tuple(c) for c in a[a[..., 3] > 0][:, :3]}
        assert used and used <= src  # only colors of the original sprite


def test_every_nth_picks_evenly_spaced_frames():
    from spritehatch.redraw import every_nth

    assert every_nth(list(range(8)), 4) == [0, 2, 4, 6]
    assert every_nth([1, 2, 3], 4) == [1, 2, 3]
    assert every_nth(list(range(8)), 4, keep_last=True) == [0, 2, 5, 7]


def test_redraw_runs_on_every_model_workflow():
    from spritehatch.comfy import WORKFLOWS
    from spritehatch.moves import render
    from spritehatch.redraw import redraw

    side, side_pts = _stick_figure("right")
    pts: list = []
    frames = render(side, side_pts, "right", "idle", keypoints=pts)
    for model in WORKFLOWS:
        out = redraw(frames, pts, still=side, character="knight", view="right",
                     motion="idle", client=_FakeComfy(), model=model)
        assert len(out) == len(frames)


def test_mirror_keeps_the_side_view_that_drew_more():
    from spritehatch.cli import _pick_sides

    thin = Image.new("RGBA", (8, 8))
    thin.putpixel((1, 1), (255, 0, 0, 255))
    full = Image.new("RGBA", (8, 8), (0, 0, 255, 255))
    assert _pick_sides({"left": "right"}, {"left": thin, "right": full}) == {"left": "right"}
    assert _pick_sides({"left": "right"}, {"left": full, "right": thin}) == {"right": "left"}
    assert _pick_sides({"left": "right"}, {"right": thin}) == {"left": "right"}


def test_open_holes_frees_trapped_background_but_keeps_eye_whites():
    from spritehatch.pixelate import open_holes

    a = np.full((100, 100, 3), 255, np.uint8)
    a[20:80, 20:80] = (40, 40, 160)          # body
    a[30:60, 30:50] = 255                    # big gap caught inside
    a[25:27, 60:62] = 255                    # tiny eye white
    cell = Image.fromarray(a)
    fg = open_holes(cell, ~background_mask(cell))
    assert not fg[40, 40]
    assert fg[25, 60]
    assert fg[70, 70]


def test_redraw_keeps_rig_frames_it_cannot_guide_or_lost():
    from spritehatch.moves import render
    from spritehatch.redraw import redraw

    side, side_pts = _stick_figure("right")
    pts: list = []
    frames = render(side, side_pts, "right", "death", keypoints=pts)[-4:]
    pts = pts[-4:]

    class Blank(_FakeComfy):
        def run(self, wf):
            self.runs += 1
            img = self.uploads[wf["init"]["inputs"]["image"]]
            return [Image.new("RGB", img.size, "white")]    # paints nothing

    out = redraw(frames, pts, still=side, character="knight", view="right",
                 motion="death", client=Blank())
    assert [np.asarray(a).tobytes() for a in out] == [np.asarray(b).tobytes() for b in frames]


def test_animate_adds_motions_to_an_earlier_run(tmp_path):
    import json

    from spritehatch.animate import ORDER, main

    (tmp_path / "frames").mkdir()
    skeleton = {}
    for d in ORDER:
        img, pts = _stick_figure("right" if d == "left" else d)
        img.save(tmp_path / "frames" / f"{d}.png")
        skeleton[d] = pts
    (tmp_path / "sheet.json").write_text(json.dumps({"order": ORDER, "skeleton": skeleton}))
    main([str(tmp_path), "--motions", "idle"])
    main([str(tmp_path), "--motions", "walk"])
    motions = json.loads((tmp_path / "anim" / "anim.json").read_text())["motions"]
    assert set(motions) == {"idle", "walk"}


def test_cells_keep_the_body_the_same_size_in_bigger_frames():
    from spritehatch.redraw import GRID_MAX, cell_size

    walk, attack = cell_size(48, 48), cell_size(72, 48)
    assert walk % 48 == 0 and attack % 72 == 0
    assert attack / 72 >= cell_size(72) / 72   # never smaller than without the body hint
    assert 2 * attack <= GRID_MAX         # but the grid stays small enough to fit in memory


def test_showcase_has_a_row_per_motion(tmp_path):
    import json

    from spritehatch.animate import ORDER, main
    from spritehatch.showcase import showcase

    (tmp_path / "frames").mkdir()
    skeleton = {}
    for d in ORDER:
        img, pts = _stick_figure("right" if d == "left" else d)
        img.save(tmp_path / "frames" / f"{d}.png")
        skeleton[d] = pts
    (tmp_path / "sheet.json").write_text(json.dumps({"order": ORDER, "skeleton": skeleton}))
    main([str(tmp_path), "--motions", "idle,death"])
    frames = showcase(tmp_path / "anim", zoom=1)
    big = 120                                # death's frame, 1.5x the 80px sprite
    assert frames[0].size == (big * 3, big * 2)


def test_scale2x_doubles_and_smooths_a_diagonal():
    from spritehatch.effects import scale2x

    a = np.zeros((2, 2, 4), np.uint8)
    a[0, 0] = a[1, 1] = (255, 0, 0, 255)       # a diagonal line
    b = scale2x(np.ascontiguousarray(a))
    assert b.shape == (4, 4, 4)
    red = b[..., 0] == 255
    assert red[1, 2] and red[2, 1]             # the gap between the blocks is filled
    assert not red[0, 3] and not red[3, 0]     # the far corners stay empty


def test_zimage_uses_no_lora_unless_asked():
    from spritehatch.comfy import zimage_redraw_workflow

    plain = zimage_redraw_workflow("knight", "a.png", "b.png")
    assert "lora" not in plain and plain["shift"]["inputs"]["model"] == ["unet", 0]
    styled = zimage_redraw_workflow("knight", "a.png", "b.png", lora="pixel.safetensors")
    assert styled["shift"]["inputs"]["model"] == ["lora", 0]


def test_zimage_turnaround_passes_style_lora():
    from spritehatch.comfy import zimage_turnaround_workflow

    plain = zimage_turnaround_workflow("knight", "p.png", ["front"])
    assert "lora" not in plain
    styled = zimage_turnaround_workflow("knight", "p.png", ["front"], lora="s.safetensors")
    assert styled["lora"]["inputs"]["strength_model"] == 0.5
    assert styled["shift"]["inputs"]["model"] == ["lora", 0]


def test_comfy_run_keeps_waiting_when_a_poll_times_out(monkeypatch):
    from spritehatch.comfy import ComfyClient

    client = ComfyClient()
    replies = iter([TimeoutError("busy loading a model"),
                    b'{"p1": {"status": {}, "outputs": {}}}'])

    def fake_get(path):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(client, "_post", lambda *a: {"prompt_id": "p1"})
    monkeypatch.setattr(client, "_get", fake_get)
    monkeypatch.setattr("spritehatch.comfy.time.sleep", lambda s: None)
    assert client.run({}) == []


def test_flatten_merges_close_shades_but_keeps_far_colors():
    from spritehatch.pixelate import _flatten

    colors = np.array([[150, 150, 150], [170, 170, 170], [10, 10, 10]], np.uint8)
    idx = np.zeros((5, 5), int)
    idx[2, 2] = 1                      # a lighter speck in grey metal: flattened
    idx[1, 1] = 2                      # a dark eye: kept
    out = _flatten(idx, np.ones((5, 5), bool), colors)
    assert out[2, 2] == 0 and out[1, 1] == 2


def test_merge_rare_folds_a_lone_shade_but_keeps_a_lone_eye():
    from spritehatch.pixelate import _merge_rare

    colors = np.array([[150, 150, 150], [175, 175, 175], [10, 10, 10]], np.uint8)
    idx = np.zeros((5, 5), int)
    idx[0, 0] = 1                      # one lighter metal pixel: merged
    idx[3, 3] = 2                      # one dark eye: kept
    out = _merge_rare(idx, np.ones((5, 5), bool), colors)
    assert out[0, 0] == 0 and out[3, 3] == 2


def test_squash_shortens_a_leg_toward_the_hip():
    from spritehatch.rig import _squash

    m = _squash(0.5, (0, 10))
    assert np.allclose(m @ [3, 10, 1], [3, 10, 1])   # the joint stays
    assert np.allclose(m @ [3, 20, 1], [3, 15, 1])   # the foot comes up halfway


def test_far_cape_pixels_stay_on_the_body():
    import spritehatch.rig as R
    from spritehatch.pose import keypoints

    img = Image.new("RGBA", (64, 64))
    sk = [None if p is None else [p[0] * 64, p[1] * 64] for p in keypoints("down")]
    a = np.zeros((64, 64, 4), np.uint8)
    a[4:60, 16:48] = (200, 200, 200, 255)
    a[50:60, 2:62] = (0, 0, 255, 255)          # a wide cape hem far from any limb
    img = Image.fromarray(a)
    rig = R.Rig(img, sk, "down")
    arms = sum(rig.parts[n][..., 3].astype(bool) for n in ("arm_r", "hand_r", "arm_l", "hand_l"))
    assert not arms[50:60, 2:5].any() and not arms[50:60, 59:62].any()


def test_a_body_falling_flat_is_not_cut_off():
    from spritehatch.moves import render
    from spritehatch.pose import keypoints

    a = np.zeros((48, 48, 4), np.uint8)
    a[2:47, 18:30] = (200, 200, 200, 255)
    a[2:12, 14:34] = (255, 0, 0, 255)                  # a big red helmet on top
    sk = [None if p is None else [p[0] * 48, p[1] * 48] for p in keypoints("right")]
    lying = render(Image.fromarray(a), sk, "right", "death")[-2]
    red = (np.asarray(lying)[..., :3] == (255, 0, 0)).all(2)
    assert red.sum() > 0.8 * 200                       # the whole helmet is still there


def test_lying_frames_are_told_apart():
    from spritehatch.redraw import _lying

    a = np.zeros((72, 72, 4), np.uint8)
    a[50:62, 10:60, 3] = 255
    assert _lying(Image.fromarray(a))
    assert not _lying(Image.fromarray(a.transpose(1, 0, 2)))


def test_redrawn_walk_keeps_the_rig_head():
    from spritehatch.redraw import _keep_head

    rig = np.zeros((10, 10, 4), np.uint8)
    rig[:, :] = (255, 0, 0, 255)
    new = np.zeros((10, 10, 4), np.uint8)
    new[:, :] = (0, 0, 255, 255)
    pts = [None] * 18
    pts[1] = (5, 4)                                    # neck at y = 4
    pts[8], pts[11] = (4, 8), (6, 8)                   # hips straight below
    out = np.asarray(_keep_head(Image.fromarray(rig), Image.fromarray(new), pts))
    assert (out[:4, :, 0] == 255).all() and (out[4:, :, 2] == 255).all()


def test_a_tipped_body_keeps_the_painted_head():
    from spritehatch.redraw import _keep_head

    new = Image.new("RGBA", (10, 10), (0, 0, 255, 255))
    pts = [None] * 18
    pts[1], pts[8], pts[11] = (2, 5), (8, 5), (8, 6)   # lying: hips beside the neck
    assert _keep_head(Image.new("RGBA", (10, 10)), new, pts) is new


def test_body_and_effects_put_back_together_match_the_normal_frames():
    from spritehatch.moves import render, with_effects
    from spritehatch.pose import keypoints

    a = np.zeros((48, 48, 4), np.uint8)
    a[2:47, 18:30] = (200, 200, 200, 255)
    for view in ("down", "right", "up"):
        sk = [None if p is None else [p[0] * 48, p[1] * 48] for p in keypoints(view)]
        for motion in ("attack", "cast"):
            layers: list = []
            bodies = render(Image.fromarray(a), sk, view, motion, effects=layers)
            normal = render(Image.fromarray(a), sk, view, motion)
            for b, lay, n in zip(bodies, layers, normal):
                assert np.array_equal(np.asarray(with_effects(b, lay)), np.asarray(n))


def test_sword_is_drawn_at_the_hand_in_any_direction():
    from spritehatch.effects import BLADE, sword

    for deg in (0, 37, 90, 200):
        layer = Image.new("RGBA", (64, 64))
        sword(layer, (32, 32), deg, 16)
        a = np.asarray(layer)
        assert a[30:35, 30:35, 3].any()                        # held at the hand
        assert (a[..., :3] == BLADE[:3]).all(2).sum() >= 4     # a visible blade


def test_sword_doubles_for_a_big_body_and_stays_at_the_hand():
    from spritehatch.effects import sword

    small, big = Image.new("RGBA", (96, 96)), Image.new("RGBA", (96, 96))
    sword(small, (40, 48), 0, 30)
    sword(big, (40, 48), 0, 30, scale=2)
    rows = lambda im: np.nonzero(np.asarray(im)[..., 3].any(1))[0]
    assert len(rows(big)) == 2 * len(rows(small))          # twice as thick
    assert np.asarray(big)[44:52, 36:44, 3].any()           # still in the hand
    assert np.asarray(big)[..., 3].any(0).sum() <= 30 + 12  # same length


def test_black_clothes_are_lifted_off_the_outline():
    from spritehatch.pixelate import LUMA, lift_darks

    colors = np.array([[6, 6, 8], [20, 18, 24], [30, 28, 34], [200, 40, 40]], np.uint8)
    out = lift_darks(colors)
    lum = out.astype(float) @ LUMA
    assert (out[0] == colors[0]).all() and (out[3] == colors[3]).all()  # outline, red stay
    assert lum[1] >= 40 and lum[2] > lum[1]                            # cloth lifted, in order


def test_redraw_patches_see_through_spots_the_rig_has_solid():
    from spritehatch.redraw import _patch_holes

    rig = np.zeros((20, 20, 4), np.uint8)
    rig[2:18, 4:16] = (90, 90, 200, 255)
    rig[8:12, 9:11] = (0, 0, 0, 0)  # a real gap the rig has too
    new = rig.copy()
    new[4:7, 6:9] = 0  # see-through spot inside the body
    new[3, 5:10] = new[7, 5:10] = (10, 10, 10, 255)  # outline ring around it
    out = np.asarray(_patch_holes(Image.fromarray(rig), Image.fromarray(new)))
    assert (out[4:7, 6:9] == rig[4:7, 6:9]).all()
    assert (out[3, 6:9] == rig[3, 6:9]).all()
    assert (out[8:12, 9:11, 3] == 0).all()


def test_open_holes_opens_gaps_but_keeps_shine():
    from spritehatch.pixelate import open_holes

    a = np.full((100, 100, 3), 255, np.uint8)
    a[10:90, 10:40] = (90, 60, 140)    # robe with a gap of background in it
    a[30:50, 18:32] = 255
    a[10:90, 60:90] = (190, 190, 200)  # bright metal with white shine on it
    a[30:50, 68:82] = 250
    cell = Image.fromarray(a)
    fg = open_holes(cell, ~background_mask(cell))
    assert not fg[40, 25]
    assert fg[40, 75]


def test_selout_shades_the_outline_with_the_color_inside():
    from spritehatch.pixelate import LUMA, selout

    a = np.zeros((12, 12, 4), np.uint8)
    a[2:10, 2:10] = (20, 20, 20, 255)          # black outline
    a[3:9, 3:6] = (40, 90, 230, 255)           # blue cape
    a[3:9, 6:9] = (230, 140, 30, 255)          # orange armor
    out = np.asarray(selout(Image.fromarray(a)))
    left, right = out[5, 2, :3].astype(int), out[5, 9, :3].astype(int)
    assert left[2] > left[0] and right[0] > right[2]       # each keeps its hue
    assert left @ LUMA <= 57 and right @ LUMA <= 57        # and stays dark
    assert (out[3:9, 3:9] == a[3:9, 3:9]).all()            # inside untouched
    assert (out[..., 3] == a[..., 3]).all()


def test_swap_sides_trades_the_left_and_right_views(tmp_path):
    from spritehatch.cli import build

    cells = []
    for _ in range(4):
        a = np.full((320, 192, 3), 255, np.uint8)
        a[60:300, 60:130] = (40, 40, 160)
        a[80:120, 110:150] = (200, 30, 30)          # a red snout: only one side
        cells.append(a)
    raw = Image.fromarray(np.concatenate(cells, axis=1))
    build(raw, "4", 48, 8, tmp_path / "a", "x")
    build(raw, "4", 48, 8, tmp_path / "b", "x", swap_sides=True)
    frame = lambda d, v: np.asarray(Image.open(tmp_path / d / "frames" / f"{v}.png"))
    assert (frame("b", "right") == frame("a", "left")).all()
    assert (frame("b", "left") == frame("a", "right")).all()


def test_creature_prompt_drops_the_chibi_body():
    from spritehatch.comfy import zimage_turnaround_workflow

    def text(wf):
        return " ".join(str(n["inputs"].get("text", "")) for n in wf.values())

    person = zimage_turnaround_workflow("slime", "p.png", ["front"])
    creature = zimage_turnaround_workflow("slime", "p.png", ["front"], creature=True)
    assert "heads tall" in text(person) and "heads tall" not in text(creature)


def test_main_body_drops_a_neighbors_wing_tip_at_the_cell_edge():
    from spritehatch.pixelate import main_body

    fg = np.zeros((40, 40), bool)
    fg[5:35, 15:25] = True              # the character
    fg[10:20, 0:4] = True               # a wing tip over the left edge
    fg[30:33, 30:33] = True             # a loose part inside the cell
    out = main_body(fg)
    assert out[20, 20] and not out[15, 1] and out[31, 31]


def test_creature_motions_stay_inside_their_frames():
    from spritehatch import creature

    a = np.zeros((48, 48, 4), np.uint8)
    a[2:47, 10:38] = (90, 200, 90, 255)             # a blob as tall as the frame
    still = Image.fromarray(a)
    for name in creature.MOTIONS:
        for view in ("down", "left", "right", "up"):
            for f in creature.render(still, view, name):
                solid = np.asarray(f)[..., 3] > 0
                assert solid.any()
                # nothing cut by the frame edge (the hop and the fall have room)
                assert not (solid[0].any() or solid[:, 0].any() or solid[:, -1].any())


def test_creature_sheet_is_marked_for_animate(tmp_path):
    import json

    from spritehatch.cli import build

    raw = Image.fromarray(np.concatenate([np.asarray(_figure(80).resize((192, 320)))] * 4, axis=1))
    build(raw, "4", 48, 8, tmp_path, "slime", creature=True)
    assert json.loads((tmp_path / "sheet.json").read_text())["creature"] is True


def test_boss_skeleton_has_a_smaller_head_and_wider_body():
    chibi, boss = keypoints("down"), keypoints("down", "boss")
    # neck sits higher (smaller head), shoulders spread wider
    assert boss[1][1] < chibi[1][1]
    assert abs(boss[2][0] - boss[5][0]) > abs(chibi[2][0] - chibi[5][0])
    for p in boss:
        assert p is None or (0 < p[0] < 1 and 0 < p[1] < 1)


def test_boss_prompt_drops_the_chibi_body():
    from spritehatch.comfy import zimage_turnaround_workflow

    def text(wf):
        return " ".join(str(n["inputs"].get("text", "")) for n in wf.values())

    boss = text(zimage_turnaround_workflow("demon", "p.png", ["front"], boss=True))
    assert "chibi" not in boss.lower() and "menacing" in boss


def test_boss_sheet_is_96px_and_marked_for_animate(tmp_path):
    import json

    from spritehatch.cli import build

    raw = Image.fromarray(np.concatenate([np.asarray(_figure(80).resize((384, 640)))] * 4, axis=1))
    build(raw, "4", 0, 8, tmp_path, "demon", boss=True)
    meta = json.loads((tmp_path / "sheet.json").read_text())
    assert meta["frame"] == 96 and meta["body"] == "boss"


def test_boss_redraw_prompt_drops_the_chibi_body():
    from spritehatch.moves import render
    from spritehatch.redraw import redraw

    side, side_pts = _stick_figure("right")
    pts: list = []
    frames = render(side, side_pts, "right", "idle", keypoints=pts)
    texts = []

    class Spy(_FakeComfy):
        def run(self, wf):
            texts.extend(str(n["inputs"].get("text", "")) for n in wf.values())
            return super().run(wf)

    redraw(frames, pts, still=side, character="demon", view="right", motion="idle",
           client=Spy(), body="boss")
    assert texts and not any("chibi" in t for t in texts)


def test_glow_keeps_a_thin_lava_crack_but_not_a_gold_plate():
    from spritehatch.pixelate import paint_small

    a = np.full((120, 120, 3), (120, 20, 15), np.uint8)    # dark red armor
    a[:, 58:61] = (250, 170, 20)                            # 3px glowing crack
    a[:, 90:] = (240, 190, 30)                              # solid gold plate
    a[:, 104:106] = (40, 20, 10)                            # dark seam in the gold
    fg = np.ones((120, 120), bool)
    colors = np.array([(120, 20, 15), (250, 170, 20), (240, 190, 30), (40, 20, 10)], np.uint8)
    plain = paint_small(a, fg, (20, 20), colors)
    lit = paint_small(a, fg, (20, 20), colors, glow=True)
    crack = lit[5:15, 9:11, 0]
    assert (plain[5:15, 9:11, 0] == 120).all()              # the vote loses it
    assert (crack > 200).any(axis=1).all()                  # glow brings it back
    assert (lit[:, 15:] == plain[:, 15:]).all()             # gold and seam untouched


def test_boss_rig_keeps_a_thick_arm_on_the_arm():
    from spritehatch.rig import Rig

    side, pts = _stick_figure("down")
    a = np.asarray(side).copy()
    # thicken the right arm well past chibi thickness
    p, q = np.array(pts[2]), np.array(pts[4])
    ys, xs = np.mgrid[:a.shape[0], :a.shape[1]]
    t = np.clip(((xs - p[0]) * (q - p)[0] + (ys - p[1]) * (q - p)[1]) / ((q - p) ** 2).sum(), 0, 1)
    near = np.hypot(xs - p[0] - t * (q - p)[0], ys - p[1] - t * (q - p)[1]) < 0.14 * a.shape[0]
    a[near & (a[..., 3] == 0)] = (200, 60, 40, 255)
    img = Image.fromarray(a)
    arm = lambda r: sum((np.asarray(r.parts[n])[..., 3] > 0).sum() for n in ("arm_r", "hand_r"))
    assert arm(Rig(img, pts, "down", 1.6)) > arm(Rig(img, pts, "down"))


def test_glow_leaves_bright_green_skin_and_its_shadow_alone():
    from spritehatch.pixelate import paint_small

    a = np.full((120, 120, 3), (58, 85, 22), np.uint8)     # skin shadow
    a[:, 58:61] = (147, 204, 54)                            # thin lit skin edge
    fg = np.ones((120, 120), bool)
    colors = np.array([(58, 85, 22), (147, 204, 54)], np.uint8)
    assert (paint_small(a, fg, (20, 20), colors, glow=True)
            == paint_small(a, fg, (20, 20), colors)).all()


def test_cast_orb_doubles_for_a_big_body():
    from spritehatch.moves import render

    def orb_size(size):
        side, pts = _stick_figure("down", size)
        layers: list = []
        render(side, pts, "down", "cast", effects=layers)
        _, front = layers[2]
        return int((np.asarray(front)[..., 3] > 0).sum())

    assert orb_size(144) > 1.6 * orb_size(48)       # a body about 90px tall


def test_creature_boss_prompt_has_no_human_body():
    from spritehatch.comfy import zimage_turnaround_workflow

    def text(wf):
        return " ".join(str(n["inputs"].get("text", "")) for n in wf.values())

    dragon = text(zimage_turnaround_workflow("dragon", "p.png", ["front"], boss=True,
                                             creature=True))
    assert "menacing" in dragon
    assert not any(w in dragon for w in ("heads tall", "shoulders", "legs", "chibi", "Cute"))


def test_creature_hop_grows_with_the_frame():
    from spritehatch import creature

    def hop(size):
        a = np.zeros((size, size, 4), np.uint8)
        a[size // 3:size - 1, size // 4:3 * size // 4] = (90, 200, 90, 255)
        frames = creature.render(Image.fromarray(a), "down", "walk")
        bottoms = [np.nonzero(np.asarray(f)[..., 3].any(1))[0].max() for f in frames]
        return max(bottoms) - min(bottoms)

    assert hop(96) >= 2 * hop(48) - 1


def _winged(size=96):
    a = np.zeros((size, size, 4), np.uint8)
    a[40:94, 38:58] = (200, 40, 40, 255)        # body
    a[10:40, 42:54] = (200, 40, 40, 255)        # head
    a[8:44, 6:38] = (150, 20, 20, 255)          # left wing
    a[8:44, 58:90] = (150, 20, 20, 255)         # right wing
    return Image.fromarray(a)


def test_flap_folds_the_wings_and_leaves_the_body():
    from spritehatch.creature import flap

    still = _winged()
    folded = np.asarray(flap(still, "down", -0.3))
    a = np.asarray(still)
    assert (folded[:, 38:58] == a[:, 38:58]).all()          # body and head stay
    tip = lambda img, x: np.nonzero(img[:, x, 3])[0].min()
    assert tip(folded, 8) > tip(a, 8) and tip(folded, 88) > tip(a, 88)


def test_flap_leaves_wingless_creatures_and_the_face_side_alone():
    from spritehatch.creature import flap

    blob = np.zeros((48, 48, 4), np.uint8)
    blob[10:47, 8:40] = (90, 200, 90, 255)
    assert (np.asarray(flap(Image.fromarray(blob), "down", -0.3)) == blob).all()
    a = np.asarray(_winged())
    side = np.asarray(flap(_winged(), "right", -0.3))
    assert (side[:, 58:] == a[:, 58:]).all()                 # the face side stays
    assert not (side[:, :38] == a[:, :38]).all()            # the back wing moves


def test_boss_torso_is_filled_where_the_arms_covered_it():
    from spritehatch.rig import Rig

    a = np.zeros((96, 96, 4), np.uint8)
    a[30:70, 28:68] = (200, 60, 40, 255)        # broad chest, arms painted over it
    a[8:30, 38:58] = (230, 180, 150, 255)       # head
    a[70:94, 34:62] = (90, 60, 40, 255)         # legs
    pts = [[48, 20], [48, 30], [34, 32], [30, 48], [30, 62], [62, 32], [66, 48], [66, 62],
           [42, 70], [42, 82], [42, 93], [54, 70], [54, 82], [54, 93],
           [45, 18], [51, 18], [40, 20], [56, 20]]
    chest = lambda r: (r.parts["torso"][30:70, 28:68, 3] > 0).sum()
    assert chest(Rig(Image.fromarray(a), pts, "down", 1.6, 1.85)) > \
        chest(Rig(Image.fromarray(a), pts, "down", 1.6))


def test_wag_lifts_the_tail_behind_a_profile_only():
    from spritehatch.creature import wag

    a = np.zeros((96, 96, 4), np.uint8)
    a[10:94, 46:70] = (200, 40, 40, 255)        # body, facing right
    a[60:66, 50:52] = 0                          # (a waist to find)
    a[80:94, 6:46] = (160, 30, 30, 255)          # tail along the floor, behind
    still = Image.fromarray(a)
    lifted = np.asarray(wag(still, "right", 0.3))
    low = lambda img, x: np.nonzero(img[:, x, 3])[0].max()
    assert low(lifted, 8) < low(a, 8)                        # the tip rises
    assert (lifted[:, 46:] == a[:, 46:]).all()               # body untouched
    assert (np.asarray(wag(still, "down", 0.3)) == a).all()  # front view: no tail


def test_four_legged_creatures_are_told_to_stand_on_all_fours():
    from spritehatch.cli import is_quadruped
    from spritehatch.comfy import zimage_turnaround_workflow

    assert is_quadruped("giant red dragon, four legs, long tail")
    assert is_quadruped("a four-legged beast") and is_quadruped("Quadruped lizard")
    assert not is_quadruped("cute purple bat") and not is_quadruped(None)

    def text(wf):
        return " ".join(str(n["inputs"].get("text", "")) for n in wf.values())

    wf = zimage_turnaround_workflow("wolf", "p.png", ["front"], creature=True, quadruped=True)
    assert "all four legs" in text(wf)
    assert "all four legs" not in text(zimage_turnaround_workflow("bat", "p.png", ["front"],
                                                                  creature=True))


def test_quadruped_guide_draws_a_low_body_in_every_view():
    from spritehatch.pose import quadruped_sheet

    img = np.asarray(quadruped_sheet(800, 320, ["down", "left", "right", "up"]).convert("L"))
    for i in range(4):
        cell = img[:, i * 200:(i + 1) * 200] > 0
        ys, _ = np.nonzero(cell)
        assert ys.min() > 0.3 * 320             # nothing up high: not standing
        assert ys.max() > 0.9 * 320             # feet on the ground
    side = img[:, 400:600] > 0
    assert np.nonzero(side)[1].max() > 0.85 * 200   # the right view's head reaches right


def test_four_legged_profiles_are_turned_to_face_their_side():
    from spritehatch.cli import face_profiles

    a = np.full((200, 160, 3), 255, np.uint8)
    a[110:150, 40:120] = (90, 90, 90)           # body
    a[150:184, 50:58] = a[150:184, 104:112] = (90, 90, 90)   # legs
    a[80:110, 110:140] = (90, 90, 90)           # head, right
    a[140:180, 4:40] = (90, 90, 90)             # tail, low on the left
    faces_right = Image.fromarray(a)
    left, right = face_profiles([faces_right, faces_right], ["left", "right"])
    assert np.array_equal(np.asarray(left), np.asarray(ImageOps.mirror(faces_right)))
    assert np.array_equal(np.asarray(right), a)
    front = face_profiles([faces_right], ["down"])[0]
    assert np.array_equal(np.asarray(front), a)


def test_second_pass_paints_front_and_back_in_the_profile_colors():
    from spritehatch.cli import refill_ends
    from spritehatch.pose import quadruped_sheet

    views = ["down", "left", "right", "up"]
    a = np.full((320, 800, 3), 255, np.uint8)
    a[200:280, 230:370] = (200, 30, 30)          # red profile in the left cell
    a[200:280, 430:570] = (200, 30, 30)          # and the right one
    a[60:300, 40:160] = (30, 30, 200)            # a blue figure standing in front
    out = np.asarray(refill_ends(Image.fromarray(a), quadruped_sheet(800, 320, views), views))
    assert (out[:, 200:600] == a[:, 200:600]).all()          # profiles kept
    front = out[:, :200][(out[:, :200] < 250).any(2)]
    assert len(front) and (front[:, 0] > front[:, 2]).all()  # reddish silhouette, not blue


def test_lifting_darks_keeps_a_near_black_gray_not_blue():
    from spritehatch.pixelate import lift_darks

    out = lift_darks(np.array([(5, 5, 6), (12, 11, 23), (200, 200, 200)], np.uint8))
    r, _, b = out[1].astype(int)
    assert b - r <= 12 and r > 25          # lighter, still about gray


def test_four_legged_dragons_get_a_dinosaur_body():
    from spritehatch.comfy import zimage_turnaround_workflow

    def text(wf):
        return " ".join(str(n["inputs"].get("text", "")) for n in wf.values())

    kw = {"creature": True, "quadruped": True}
    assert "dinosaur" in text(zimage_turnaround_workflow("red Dragon", "p.png", ["front"], **kw))
    assert "dinosaur" not in text(zimage_turnaround_workflow("wolf", "p.png", ["front"], **kw))
    assert "dinosaur" not in text(zimage_turnaround_workflow("dragon", "p.png", ["front"],
                                                             creature=True))


def test_bulky_beasts_get_the_heavy_four_legged_guide():
    from spritehatch.cli import quadruped_body
    from spritehatch.pose import QUAD_BODIES, quadruped_sheet

    assert quadruped_body("giant red Dragon, four legs") == "heavy"
    assert quadruped_body("dire wolf, four legs") == "lean"
    for body in QUAD_BODIES:
        assert np.asarray(quadruped_sheet(800, 320, ["down", "left", "right", "up"], body)).any()


def test_the_profile_closer_to_the_guide_is_kept():
    from spritehatch.cli import _pick_sides, guide_match

    guide = np.zeros((100, 200, 3), np.uint8)
    guide[40:90, 10:90] = guide[40:90, 110:190] = 200          # both profiles: a wide body
    good = np.full((100, 100, 3), 255, np.uint8)
    good[40:90, 10:90] = 80
    headless = np.full((100, 100, 3), 255, np.uint8)
    headless[40:90, 30:70] = 80
    cells = [Image.fromarray(good), Image.fromarray(headless)]
    scores = guide_match(cells, Image.fromarray(guide), ["left", "right"])
    assert scores["left"] > scores["right"]
    big = {"left": Image.new("RGBA", (4, 4)), "right": Image.new("RGBA", (4, 4), (1, 1, 1, 255))}
    assert _pick_sides({"left": "right"}, big, scores) == {"right": "left"}


def test_profiles_are_turned_to_fit_their_guide():
    from spritehatch.cli import face_profiles

    guide = np.zeros((100, 200, 3), np.uint8)
    guide[50:90, 20:80] = 200                    # left view: body
    guide[30:50, 10:30] = 200                    # head up on the left
    cell = np.full((100, 100, 3), 255, np.uint8)
    cell[50:90, 20:80] = 80
    cell[30:50, 70:90] = 80                      # drawn with the head on the right
    blank = np.full((100, 100, 3), 255, np.uint8)
    left, _ = face_profiles([Image.fromarray(cell), Image.fromarray(blank)], ["left", "right"],
                            Image.fromarray(guide))
    assert (np.asarray(left) == np.asarray(ImageOps.mirror(Image.fromarray(cell)))).all()


def _four_legs(size=96):
    a = np.zeros((size, size, 4), np.uint8)
    a[40:70, 16:80] = (180, 50, 40, 255)                     # body
    a[30:50, 76:90] = (180, 50, 40, 255)                     # head
    for x in (20, 32, 62, 72):
        a[70:95, x:x + 6] = (150, 40, 30, 255)               # four legs
    return Image.fromarray(a)


def test_four_legged_walk_moves_the_legs_not_the_body():
    from spritehatch.creature import render, stride

    still = _four_legs()
    a = np.asarray(still)
    moved = np.asarray(stride(still, "right", 4))
    assert (moved[:70] == a[:70]).all()                      # body untouched
    assert not (moved[70:] == a[70:]).all()                  # legs swing
    feet = lambda img: np.nonzero(img[94, :, 3])[0]
    assert feet(moved).min() < feet(a).min()                 # hind feet back
    assert feet(moved).max() > feet(a).max()                 # front feet forward
    lifted = np.asarray(stride(still, "down", 3))
    assert (lifted[94, :48, 3] == 0).all() and lifted[94, 48:, 3].any()
    hop = render(still, "right", "walk")
    trot = render(still, "right", "walk", quadruped=True)
    assert len(trot) == len(hop) and all(f.size == hop[0].size for f in trot)


def _face_hits(view, motion, size=80):
    """Per frame: effect pixels drawn in front of the face (the stick figure's
    blue head below its top third; in a profile only its front half)."""
    from spritehatch.moves import render

    still, pts = _stick_figure(view, size)
    layers: list = []
    bodies = render(still, pts, view, motion, effects=layers)
    hits = []
    for body, (_, front) in zip(bodies, layers):
        b = np.asarray(body)
        head = (b[..., 3] > 0) & (b[..., 2] > 150) & (b[..., 0] < 120)
        ys = np.nonzero(head.any(axis=1))[0]
        head[: ys.min() + (ys.max() - ys.min()) // 3] = False
        if view == "right":
            xs = np.nonzero(head.any(axis=0))[0]
            head[:, : (xs.min() + xs.max()) // 2] = False
        hits.append(int(((np.asarray(front)[..., 3] > 0) & head).sum()))
    return hits


def test_sword_and_magic_stay_off_the_face():
    for view in ("down", "right"):
        for motion in ("attack", "cast"):
            assert max(_face_hits(view, motion)) <= 3, (view, motion, _face_hits(view, motion))


def test_side_attack_strikes_forward():
    """The frames with a slash trail point the blade ahead, not behind the back."""
    from spritehatch.moves import _sword_layer, attack, canvas_size, make_rig

    still, pts = _stick_figure("right", 80)
    rig = make_rig(still, pts, "right", canvas_size(80))
    state: dict = {}
    for pose in attack("right"):
        _, world = rig.render(pose)
        layer = np.asarray(_sword_layer(rig, world, "right", state, pose))
        if pose.get("trail"):
            shoulder = rig.point(world, "arm_r", 2)
            blade = (layer[..., 3] > 0) & (layer[..., 0] > 200) & (layer[..., 1] > 200)
            xs = np.nonzero(blade.any(axis=0))[0]
            assert xs.max() > shoulder[0] + 8 and xs.mean() > shoulder[0]


def test_head_moves_with_the_neck_but_never_turns():
    """A turned 48px face breaks into stairs, so the head is only moved,
    by whole pixels, to wherever the tilted neck goes."""
    from spritehatch.rig import Rig

    img, pts = _stick_figure("down")
    rig = Rig(img, pts, "down")
    world = rig.matrices({"torso": -10, "head": 12})
    head = world["head"]
    assert np.allclose(head[:2, :2], np.eye(2))              # not turned
    assert np.allclose(head[:2, 2], np.round(head[:2, 2]))   # whole pixels
    neck = world["torso"] @ np.array([*rig.pts[1], 1.0])
    assert np.abs(rig.point(world, "head", 1) - neck[:2]).max() <= 0.5

