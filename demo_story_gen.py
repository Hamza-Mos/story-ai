#!/usr/bin/env python3
"""
 demo_story_gen.py  ·  v4  — sharp manga stills + fluid narration
 -----------------------------------------------------------------
 * Switches to **cjwbw/waifu‑diffusion** on Replicate with quality tags
 * Images: 512 px, 75 steps, guidance 7 (crisper lines)
 * Single ElevenLabs call → one continuous MP3 (no clip gaps)
 * Frames auto‑split by equal duration (simple & smooth)

 Quick install
   pip install openai elevenlabs python-dotenv requests tqdm replicate moviepy

 .env (minimum)
   OPENAI_API_KEY=<sk‑…>
   ELEVEN_API_KEY=<eleven‑…>
   REPLICATE_API_TOKEN=<r8_…>
   # optional voice id
   ELEVEN_VOICE_ID=21m00Tcm4TlvDq8ikWAM
"""

import argparse, json, os, tempfile, time
from pathlib import Path
from typing import List

import openai
from dotenv import load_dotenv
from tqdm import tqdm

IMAGE_RES = 512
MODEL_REPLICATE = "cjwbw/waifu-diffusion"
QUALITY_PREFIX = ["best quality", "masterpiece", "high detail"]

# ── GPT helpers ─────────────────────────────────────────────────────────────

def scene_beats(narration: str, n_force: int | None):
    """Return list[{tags, subtitle}] using GPT‑4o‑mini."""
    if n_force is None:
        sys = (
            "You are an anime storyboarder. Break the user's narration into 4‑12 beats. "
            "Return JSON list [{tags, subtitle}]. 'tags' must be 15‑25 Danbooru tags, lowercase." )
    else:
        sys = f"You are an anime storyboarder. Split into exactly {n_force} beats. Same JSON spec."

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=900,
        messages=[{"role":"system","content":sys}, {"role":"user","content":narration}],
        response_format={"type":"json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    for key, value in data.items():
        data = data[key]
        break
    return data


def cohesive_story(subtitles: List[str]):
    joined = " ".join(subtitles)
    sys = "Rewrite these beat captions into a flowing first‑person story (150‑200 words)."
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=300,
        messages=[{"role":"system","content":sys}, {"role":"user","content":joined}],
    )
    return resp.choices[0].message.content.strip()

# ── Replicate image call ────────────────────────────────────────────────────

def gen_image(tags: str) -> bytes:
    import replicate, requests as rq, os

    if not os.getenv("_WAIFU_VER"):
        os.environ["_WAIFU_VER"] = replicate.models.get(MODEL_REPLICATE).latest_version.id
    ver = os.environ["_WAIFU_VER"]

    print('tags', tags)

    out = replicate.run(
        f"{MODEL_REPLICATE}:{ver}",
        input={
            "prompt": ", ".join(QUALITY_PREFIX + tags),
            "width": IMAGE_RES,
            "height": IMAGE_RES,
            "num_inference_steps": 75,
            "guidance_scale": 7,
            "scheduler": "k_euler",
            "negative_prompt": "lowres, bad anatomy, bad hands, text",
        },
    )
    url = out[0] if isinstance(out, list) else out
    return rq.get(url).content

# ── ElevenLabs single‑call narration ────────────────────────────────────────

def whole_narration(text: str, out_path: Path):
    from elevenlabs import generate, save
    audio = generate(
        api_key=os.getenv("ELEVEN_API_KEY"),
        model="eleven_multilingual_v1",
        voice=os.getenv("ELEVEN_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        text=text,
    )
    save(audio, out_path)

# ── Video assembly (equal split) ────────────────────────────────────────────

def build_video(images: List[Path], audio_path: Path, out: Path):
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
    audio = AudioFileClip(str(audio_path))
    per = audio.duration / len(images)
    clips = [ImageClip(str(p)).set_duration(per) for p in images]
    concatenate_videoclips(clips, method="compose").set_audio(audio)\
        .write_videofile(str(out), fps=30, codec="libx264", audio_codec="aac", logger=None)

# ── main ────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("-t", "--text")
    g.add_argument("-a", "--audio")
    ap.add_argument("--scenes", type=int)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("-o", "--out", default="out")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True)
    if args.audio:
        with open(args.audio, "rb") as f:
            prompt = openai.audio.transcriptions.create(model="whisper-1", file=f, response_format="text").text
    else:
        prompt = args.text
    print("Prompt:", prompt)

    scenes = scene_beats(prompt, args.scenes)
    print("GPT scenes:", len(scenes))

    subtitles = [s["subtitle"] for s in scenes]
    story_text = cohesive_story(subtitles)

    # --- TTS once ---
    narration_mp3 = out_dir / "story.mp3"
    whole_narration(story_text, narration_mp3)

    # --- Images ---
    img_paths = []
    for i, s in enumerate(tqdm(scenes, desc="Images")):
        png = gen_image(s["tags"])
        p = out_dir / f"scene-{i}.png"; p.write_bytes(png)
        img_paths.append(p)

    if args.video:
        build_video(img_paths, narration_mp3, out_dir / "story.mp4")
        print("Video saved →", out_dir / "story.mp4")
    else:
        print("Done. Check", out_dir)


if __name__ == "__main__":
    main()
