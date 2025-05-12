#!/usr/bin/env python3
"""
 demo_story_gen.py  ·  *v3*  — high‑quality manga stills only
 -------------------------------------------------------------
 Fixes from feedback  ❯❯❯
   1. Use **cjwbw/waifu‑diffusion** model on Replicate for sharper manga art.
   2. GPT now returns a **Danbooru‑style tag prompt** (15‑25 tags) per scene so
     every image call is richly described.
   3.  (Optional) You can still build a video, but the default `--stills` flag
      dumps only the PNGs and narration MP3s.

 Quick install:
   pip install openai elevenlabs python-dotenv requests tqdm replicate moviepy

 .env (3 keys – same as before):
   OPENAI_API_KEY=sk‑…
   ELEVEN_API_KEY=elev‑…
   REPLICATE_API_TOKEN=r8_…

 Example  –  just export the stills (no video build):
   python demo_story_gen.py -t "A cat‑girl ninja rescues a pizza chef on the moon" \
      --stills

 Example  –  full narrated video:
   python demo_story_gen.py -t "A shy dragon learns violin" --video
"""

import argparse, json, os, tempfile, time
from pathlib import Path
from typing import List

import openai, requests
from dotenv import load_dotenv
from tqdm import tqdm

# ----- constants ------------------------------------------------------------
MODEL_REPLICATE = "cjwbw/waifu-diffusion"  # sharper manga look
IMAGE_RES = 512
TAG_COUNT = (15, 25)                       # min, max Danbooru tags per scene

# ----- helpers --------------------------------------------------------------

def usd(x: float) -> str:
    return f"${x:.4f}"

# ----- GPT: scene → Danbooru tag prompt -------------------------------------

def story_to_scenes(prompt: str, n: int | None) -> List[dict]:
    if n is None:
        system = (
            "You are a manga storyboarder. Split the user prompt into an ordered "
            "list of short scenes (4‑10). For each scene output JSON with: "
            "'tags' (comma‑separated Danbooru tags, 15‑25 items) and 'subtitle' (≤20 words)."
        )
    else:
        system = (
            f"You are a manga storyboarder. Split the user prompt into exactly {n} scenes. "
            "Return JSON list of {'tags', 'subtitle'} as above."
        )
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=700,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

# ----- Replicate call --------------------------------------------------------

def gen_image_tags(tags: str) -> bytes:
    import replicate, requests as rq

    # grab the model’s latest version id on the fly
    version = replicate.models.get("cjwbw/waifu-diffusion").latest_version.id

    out = replicate.run(
        f"cjwbw/waifu-diffusion:{version}",
        input={
            "prompt": tags,
            "width": IMAGE_RES,
            "height": IMAGE_RES,
            "num_inference_steps": 50,
            "guidance_scale": 6,
            "scheduler": "k_euler",
            "negative_prompt": "lowres, bad anatomy, bad hands, text",
        },
    )
    url = out[0] if isinstance(out, list) else out
    return rq.get(url).content


# ----- ElevenLabs TTS per scene ---------------------------------------------

def tts_clip(text: str, tmp: Path) -> Path:
    from elevenlabs import generate, save

    audio = generate(
        api_key=os.getenv("ELEVEN_API_KEY"),
        model="eleven_multilingual_v1",
        voice=os.getenv("ELEVEN_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        text=text,
    )
    out = tmp / f"{abs(hash(text))}.mp3"
    save(audio, out)
    return out

# ----- Optional video combine ----------------------------------------------

def build_video(images: List[Path], audios: List[Path], outfile: Path):
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_audioclips, concatenate_videoclips

    aud_clips = [AudioFileClip(str(p)) for p in audios]
    full_audio = concatenate_audioclips(aud_clips)
    video_clips = [ImageClip(str(img)).set_duration(ac.duration)
                   for img, ac in zip(images, aud_clips)]
    concatenate_videoclips(video_clips, method="compose").set_audio(full_audio)\
        .write_videofile(str(outfile), fps=30, codec="libx264", audio_codec="aac", logger=None)

# ----- main -----------------------------------------------------------------

def main():
    load_dotenv()

    ap = argparse.ArgumentParser(description="Generate manga‑style stills (+optional video)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("-t", "--text", help="raw prompt")
    g.add_argument("-a", "--audio", help="WAV for Whisper STT")
    ap.add_argument("--scenes", type=int, help="force scene count")
    ap.add_argument("--video", action="store_true", help="also build MP4 video")
    ap.add_argument("--stills", action="store_true", help="only export PNG + MP3 (default)")
    ap.add_argument("-o", "--out", default="out", help="output directory")
    args = ap.parse_args()
    if not args.video: args.stills = True

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    # 1. prompt (or Whisper)
    if args.audio:
        with open(args.audio, "rb") as f:
            prompt_txt = openai.audio.transcriptions.create(model="whisper-1", file=f, response_format="text").text
    else:
        prompt_txt = args.text
    print("Prompt:", prompt_txt)

    # 2. GPT scene → tags/subtitle
    scenes = story_to_scenes(prompt_txt, args.scenes)
    scenes = scenes["scenes"]
    print(f"GPT returned {len(scenes)} scenes")

    # 3. Generate images + per‑scene TTS
    tmp_aud = tempfile.TemporaryDirectory()
    img_paths, aud_paths = [], []
    for i, sc in enumerate(tqdm(scenes, desc="Scenes")):
        img_bin = gen_image_tags(sc["tags"])
        img_path = out_dir / f"scene-{i}.png"
        img_path.write_bytes(img_bin)
        img_paths.append(img_path)

        aud_paths.append(tts_clip(sc["subtitle"], Path(tmp_aud.name)))

    # 4. Optional video
    if args.video:
        mp4 = out_dir / "story.mp4"
        build_video(img_paths, aud_paths, mp4)
        print("Saved video →", mp4)
    else:
        print("Done. PNG + MP3 clips saved in", out_dir)


if __name__ == "__main__":
    main()
