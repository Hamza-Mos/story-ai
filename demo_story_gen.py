#!/usr/bin/env python3
"""
 demo_story_gen.py  ·  voice → anime‑style slideshow → narrated MP4
 --------------------------------------------------------------------
 A minimal, single‑file demo that shows every step of the pipeline so you can
 measure latency *and* see what each API call costs you in real dollars.

 1. (Optional) Whisper STT             – OpenAI     – $0.006 / audio‑min
 2. Story + 6 scene prompts            – GPT‑4o‑mini– $0.0002 / story
 3. Anime stills (6 × 512 px)          – Stability  – $0.002  / image (REST, no gRPC)
 4. Narration (TTS)                    – ElevenLabs – $0.015 / 1K chars
 5. Video mux (local ffmpeg/moviepy)   – free CPU
 --------------------------------------------------------------------
 Quick‑start install (Python 3.9‑3.12):
     pip install openai elevenlabs moviepy python-dotenv requests tqdm
 #            ↑ no stability-sdk ⇒ avoids heavy grpcio build

 Environment variables (edit your .env):
     OPENAI_API_KEY=sk‑…
     STABILITY_API_KEY=prodsecret‑…
     ELEVEN_API_KEY=elev‑…

 Text‑only prompt demo:
     python demo_story_gen.py -t "A brave cat‑girl ninja rescues a pizza chef on the moon"

 Recorded‑audio demo:
     python demo_story_gen.py --audio my_prompt.wav

 This creates:
     out/scene‑0.png … scene‑5.png   (anime stills)
     out/narration.mp3               (TTS)
     out/story.mp4                   (30‑fps MP4 slideshow)

 End‑to‑end variable spend ≈ 2–3 ¢ / story.
"""

import argparse, json, os, time
from pathlib import Path

import openai, requests
from dotenv import load_dotenv
from tqdm import tqdm

# ---------- config ----------
SCENES = 6                  # how many still frames per story
IMAGE_RES = 512             # px (square)

# ---------- helpers ----------

def usd(x: float) -> str:
    return f"${x:.4f}"

# ---------- pipeline steps ----------

# 1 ▸ Speech‑to‑text ---------------------------------------------------------

def transcribe_whisper(audio_path: str) -> str:
    """Return plain‑text transcription using OpenAI Whisper API."""
    with open(audio_path, "rb") as f:
        start = time.time()
        resp = openai.audio.transcriptions.create(
            model="whisper-1", file=f, response_format="text"
        )
    print(f"Whisper latency: {time.time() - start:.2f}s")
    return resp.text.strip()

# 2 ▸ Story + scene JSON ------------------------------------------------------

def storybeats(user_prompt: str, scenes: int = SCENES):
    """Ask GPT‑4o‑mini for structured JSON describing N scenes."""
    system = (
        "You are a children's storyteller. Split the user prompt into exactly "
        f"{scenes} sequential scenes. Return JSON {{\"scenes\": [{{\"prompt\", \"subtitle\"}}]}}"
    )
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=400,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)["scenes"]

# 3 ▸ Anime stills (REST call, no stability-sdk) -----------------------------

def generate_image(prompt: str, out_path: Path):
    """Generate a single image via Stability *core* endpoint (multipart/form‑data)."""
    url = "https://api.stability.ai/v2beta/stable-image/generate/core"

    headers = {
        "Authorization": f"Bearer {os.getenv('STABILITY_API_KEY')}",
        "Accept": "image/*",                   # png or webp returned
    }

    # `core` insists on multipart/form‑data *plus* a dummy `files` field.
    files = {"none": ""}                         # <= required placeholder
    data = {
        "prompt": prompt,
        "output_format": "png",                 # or "webp" (smaller)
    }

    r = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Stability error {r.status_code}: {r.text}")

    out_path.write_bytes(r.content)

# 4 ▸ Text‑to‑speech ---------------------------------------------------------

def tts(text: str, out_path: Path):
    from elevenlabs import generate, save

    audio = generate(
        api_key=os.getenv("ELEVEN_API_KEY"),
        model="eleven_multilingual_v1",
        voice=os.getenv("ELEVEN_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        text=text,
    )
    save(audio, out_path)

# 5 ▸ Assemble video ---------------------------------------------------------

def mux_video(images_dir: Path, audio_path: Path, out_path: Path):
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

    imgs = sorted(images_dir.glob("scene-*.png"))
    audio = AudioFileClip(str(audio_path))
    per = audio.duration / len(imgs)

    clips = [ImageClip(str(p)).set_duration(per) for p in imgs]
    concatenate_videoclips(clips, method="compose")\
        .set_audio(audio)\
        .write_videofile(str(out_path), fps=30, codec="libx264", audio_codec="aac", logger=None)

# ---------- main CLI ----------

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="End‑to‑end anime story generator demo")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("-t", "--text", metavar="PROMPT", help="Raw text prompt")
    g.add_argument("-a", "--audio", metavar="WAV", help="Path to WAV for Whisper transcription")
    parser.add_argument("-o", "--out", default="out", help="Output directory [out]")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ▸ Step 1
    user_prompt = transcribe_whisper(args.audio) if args.audio else args.text
    print("User prompt:", user_prompt)

    # ▸ Step 2
    scenes = storybeats(user_prompt)
    print("Got", len(scenes), "scenes from GPT‑4o‑mini")

    # ▸ Step 3  – anime stills
    for i, scene in enumerate(tqdm(scenes, desc="Generating images")):
        generate_image(scene["prompt"], out_dir / f"scene-{i}.png")

    # ▸ Step 4  – narration TTS
    script_text = " ".join(s["subtitle"] for s in scenes)
    audio_path = out_dir / "narration.mp3"
    tts(script_text, audio_path)

    # ▸ Step 5  – mux
    video_path = out_dir / "story.mp4"
    mux_video(out_dir, audio_path, video_path)

    # ▸ Rough cost printout
    cost_imgs = 0.002 * len(scenes)
    cost_tts = 0.015 * (len(script_text) / 1000)
    print("\nFinished! Video saved →", video_path)
    print("Approx cost this run → Images", usd(cost_imgs), "+ TTS", usd(cost_tts))


if __name__ == "__main__":
    main()
