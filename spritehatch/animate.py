"""spritehatch-animate: add motions to a sheet made by spritehatch.

The standing sprite is cut into body parts along the pose skeleton saved in
sheet.json, and every motion bends those parts at the joints (see rig.py and
moves.py). Sword, slash trail and magic are drawn on top as pixels.

By default nothing is redrawn, so every frame keeps the exact look and colors
of the original. --redraw lets the model paint over those frames (redraw.py)
so they look hand drawn; it needs ComfyUI.

idle:   breathing, arms sway a little
walk:   8 frames, legs swing with bending knees, arms swing opposite
attack: sword wind-up, swing with a slash trail, follow-through
cast:   arms rise, an orb grows over the head and is thrown forward
death:  hit flash, staggers back, falls flat, lies still

Attack, cast and death use a 1.5x bigger frame to fit the sword and the fall.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps

from . import creature as beast
from .comfy import WORKFLOWS, ComfyClient
from .moves import MOTIONS, key_frames, render, with_effects
from .pixelate import preview, selout, sheet
from .redraw import GROUP, every_nth, redraw

ORDER = ["down", "left", "right", "up"]
# Played once, so they must keep their first and last frame when thinned.
ONE_SHOT = {"attack", "cast", "death"}
# Last frame of these stays up longer in the preview GIFs.
HOLD = {"death": 1200}


def stack(rows: list[list[Image.Image]]) -> Image.Image:
    strips = [sheet(r) for r in rows]
    out = Image.new("RGBA", (max(s.width for s in strips), sum(s.height for s in strips)))
    y = 0
    for s in strips:
        out.paste(s, (0, y))
        y += s.height
    return out


def save_gif(frames: list[Image.Image], path: Path, ms: int, zoom: int = 6,
             hold: int | None = None) -> None:
    big = [preview(f, zoom).convert("RGB") for f in frames]
    durations = [ms] * len(big)
    if hold:
        durations[-1] = hold
    big[0].save(path, save_all=True, append_images=big[1:], duration=durations, loop=0)


def _save_motion(out: Path, result: dict, name: str, rows: list, ms: int) -> None:
    stack(rows).save(out / f"{name}.png")
    for d, frames in zip(ORDER, rows):
        save_gif(frames, out / f"{name}_{d}.gif", ms, hold=HOLD.get(name))
    result["motions"][name] = {"file": f"{name}.png", "frame": rows[0][0].width,
                               "frames": len(rows[0]), "ms": ms}


def _creature(stills: dict, args, out: Path) -> None:
    """Creatures have no skeleton: they squash, hop and tip over whole."""
    names = args.motions.split(",") if args.motions != ",".join(MOTIONS) else list(beast.MOTIONS)
    index = out / "anim.json"
    result = json.loads(index.read_text()) if index.exists() else {"rows": ORDER, "motions": {}}
    for name in names:
        if name not in beast.MOTIONS:
            raise SystemExit(f"creatures have: {', '.join(beast.MOTIONS)}")
        rows = [[selout(f) for f in beast.render(stills[d], d, name, args.quadruped)]
                for d in ORDER]
        _save_motion(out, result, name, rows, beast.MOTIONS[name][1])
    index.write_text(json.dumps(result, indent=2))
    print(f"done: {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="spritehatch-animate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet_dir", type=Path, help="output folder of spritehatch (-d 4)")
    ap.add_argument("--motions", default=",".join(MOTIONS),
                    help="comma separated: " + ", ".join(MOTIONS))
    ap.add_argument("--redraw", action="store_true",
                    help="let the model paint over the rig frames, 4 per motion (needs ComfyUI, "
                         "about 2-3 min per motion and direction on an M1 Pro)")
    ap.add_argument("--denoise", type=float, default=0.6,
                    help="how much --redraw may change (0.6 keeps the look; higher drifts)")
    ap.add_argument("--model", choices=sorted(WORKFLOWS), default="zimage",
                    help="model --redraw paints with: zimage (Z-Image-Turbo, best) or sdxl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None,
                    help="sampling steps per --redraw grid (zimage: 6; 4 is twice as fast "
                         "for a quick look, small marks melt)")
    ap.add_argument("--style-lora", default=None,
                    help="a LoRA in ComfyUI's models/loras for --redraw (as in spritehatch)")
    ap.add_argument("--style-strength", type=float, default=0.5, help="strength of --style-lora")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    args = ap.parse_args(argv)

    src = args.sheet_dir
    meta = json.loads((src / "sheet.json").read_text())
    body = meta.get("body", "chibi")

    if meta.get("order") != ORDER:
        ap.error("only 4-direction sheets are supported")
    if "skeleton" not in meta:
        ap.error("sheet.json has no skeleton; re-cut it with spritehatch --from-raw")
    out = src / "anim"
    out.mkdir(exist_ok=True)
    stills = {d: Image.open(src / "frames" / f"{d}.png").convert("RGBA") for d in ORDER}
    args.quadruped = bool(meta.get("quadruped"))
    if meta.get("creature"):
        _creature(stills, args, out)
        return
    client = ComfyClient(args.server) if args.redraw else None

    # Add to what an earlier run made, so motions can be made (or redrawn) a
    # few at a time.
    index = out / "anim.json"
    result = json.loads(index.read_text()) if index.exists() else {"rows": ORDER, "motions": {}}
    for name in args.motions.split(","):
        if name not in MOTIONS:
            ap.error(f"unknown motion: {name}")
        _, ms, _ = MOTIONS[name]
        # Left is drawn from the right-facing sprite and flipped.
        drawn = {}
        for d in ("down", "right", "up"):
            pts: list = []
            if not client:
                drawn[d] = render(stills[d], meta["skeleton"][d], d, name, keypoints=pts,
                                  body=body)
                continue
            # The model paints the body only; sword and magic go back on top
            # afterwards, crisp, and out of the way of the face it keeps.
            layers: list = []
            frames = render(stills[d], meta["skeleton"][d], d, name, keypoints=pts,
                            effects=layers, body=body)
            # One 2x2 grid holds the whole motion, so all frames get painted
            # together and stay the same character (two grids drift apart).
            keys = key_frames(name, d)
            if keys:
                frames, pts, layers = ([x[k] for k in keys] for x in (frames, pts, layers))
            else:
                last = name in ONE_SHOT
                frames, pts, layers = (every_nth(x, GROUP, last) for x in (frames, pts, layers))
            frames = redraw(frames, pts, still=stills[d], character=meta.get("character", ""),
                            view=d, motion=name, client=client,
                            denoise=args.denoise, seed=args.seed, model=args.model,
                            steps=args.steps, lora=args.style_lora,
                            lora_strength=args.style_strength,
                            body=body)
            drawn[d] = [with_effects(f, lay) for f, lay in zip(frames, layers)]
        drawn["left"] = [ImageOps.mirror(f) for f in drawn["right"]]
        rows = [[selout(f) for f in drawn[d]] for d in ORDER]
        # Fewer frames play slower, so the motion keeps its length.
        ms = round(ms * len(MOTIONS[name][0]("right")) / len(rows[0]))
        _save_motion(out, result, name, rows, ms)
        # Saved after every motion, so a run that fails later (ComfyUI timing
        # out) still leaves anim.json matching the sheets on disk.
        index.write_text(json.dumps(result, indent=2))

    print(f"done: {out}")


if __name__ == "__main__":
    main()
