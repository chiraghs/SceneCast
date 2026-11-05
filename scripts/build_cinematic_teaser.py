#!/usr/bin/env python3
"""
AdVision — Standalone 20-Second Cinematic Teaser Generator
- Purely demonstrates the viewer's movie-watching experience on TV
- Contrasts annoying unskippable pop-up ads against seamless in-scene Meta AI placement
- Studio Neural Voiceover (en-US-AndrewMultilingualNeural)
- Full 1080p broadcast video
"""

import os
import sys
import time
import asyncio
import subprocess
import shutil

BASE_DIR = "/Volumes/DiskD/Nanda/Advision - Scenecast"
TEMP_DIR = os.path.join(BASE_DIR, "scripts", "video_temp")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "advision-cinematic-teaser-20s.mp4")
ARTIFACT_VIDEO = "/Users/chiraghs/.gemini/antigravity/brain/794021fa-b530-4b49-9ed7-5589a1ec4ece/advision-cinematic-teaser-20s.mp4"

os.makedirs(TEMP_DIR, exist_ok=True)

NARRATION_TEXT = (
    "Tired of loud, unskippable commercials interrupting the climax of your favorite movie? "
    "AdVision transforms streaming entertainment by embedding personalized brand campaigns naturally inside the scene. "
    "Zero interruptions. Pure cinematic immersion. Powered by Meta AI."
)


async def generate_speech():
    import edge_tts
    print("\n--- [Phase 1/3] Generating Teaser Neural Narration via Edge TTS ---")
    audio_file = os.path.join(TEMP_DIR, "teaser_narration.mp3")
    comm = edge_tts.Communicate(NARRATION_TEXT, "en-US-AndrewMultilingualNeural", rate="+6%")
    await comm.save(audio_file)
    
    res = subprocess.run([
        "/opt/homebrew/bin/ffprobe", "-i", audio_file,
        "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"
    ], capture_output=True, text=True)
    dur = float(res.stdout.strip())
    print(f"Teaser narration duration: {dur:.2f}s")
    return audio_file, dur


def record_teaser_video(target_duration):
    from playwright.sync_api import sync_playwright
    print(f"\n--- [Phase 2/3] Recording 1080p TV Experience ({target_duration:.2f}s) via Playwright ---")
    
    record_dir = os.path.join(TEMP_DIR, "teaser_raw_capture")
    if os.path.exists(record_dir):
        shutil.rmtree(record_dir)
    os.makedirs(record_dir, exist_ok=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-web-security",
                "--autoplay-policy=no-user-gesture-required",
                "--window-size=1920,1080"
            ]
        )
        
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=record_dir,
            record_video_size={"width": 1920, "height": 1080}
        )
        page = context.new_page()
        
        page.goto("http://localhost:3001/teaser.html")
        page.wait_for_load_state("networkidle")
        print("Teaser player running...")
        
        # Record full duration
        time.sleep(target_duration + 1.0)
        
        print("Finalizing teaser capture...")
        page.close()
        context.close()
        browser.close()

    recorded_files = [
        os.path.join(record_dir, f)
        for f in os.listdir(record_dir)
        if f.endswith(".webm") or f.endswith(".mp4")
    ]
    if not recorded_files:
        raise RuntimeError("No raw recording file found for teaser.")
    
    raw_video = recorded_files[0]
    print(f"Raw teaser recording saved: {raw_video}")
    return raw_video


def compose_teaser(raw_video, narration_audio, narration_duration):
    print("\n--- [Phase 3/3] Muxing Teaser Master Video ---")
    
    # Generate warm, gentle 432 Hz harmonic ambient pad
    ambient_audio = os.path.join(TEMP_DIR, "teaser_ambient.mp3")
    fade_out_start = max(0, narration_duration - 3.5)
    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-y", "-f", "lavfi",
        "-i", f"sine=frequency=432:sample_rate=48000:duration={narration_duration + 2}",
        "-af", f"lowpass=f=450,volume=0.030,afade=t=in:ss=0:d=2,afade=t=out:st={fade_out_start:.2f}:d=3",
        ambient_audio
    ], check=True)

    # Mix narration with ambient track
    mixed_audio = os.path.join(TEMP_DIR, "teaser_mixed.mp3")
    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-y",
        "-i", narration_audio,
        "-i", ambient_audio,
        "-filter_complex", "[0:a]volume=1.0[v];[1:a]volume=0.6[a];[v][a]amix=inputs=2:duration=first:dropout_transition=2[out]",
        "-map", "[out]", "-c:a", "libmp3lame", "-b:a", "192k",
        mixed_audio
    ], check=True)

    # Probe raw video
    res_v = subprocess.run([
        "/opt/homebrew/bin/ffprobe", "-i", raw_video,
        "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"
    ], capture_output=True, text=True)
    video_duration = float(res_v.stdout.strip())
    
    target_duration = max(19.5, narration_duration)
    speed_factor = video_duration / target_duration
    print(f"Synchronizing speed factor: {speed_factor:.4f} for target: {target_duration:.2f}s")

    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-y",
        "-i", raw_video,
        "-i", mixed_audio,
        "-filter_complex", f"[0:v]setpts={1.0/speed_factor:.5f}*PTS,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,format=yuv420p[v]",
        "-map", "[v]",
        "-map", "1:a",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-t", f"{target_duration:.2f}",
        OUTPUT_VIDEO
    ], check=True)

    print(f"\n Master Teaser Video Created: {OUTPUT_VIDEO}")
    try:
        shutil.copy2(OUTPUT_VIDEO, ARTIFACT_VIDEO)
        print(f" Artifact copy saved: {ARTIFACT_VIDEO}")
    except Exception as e:
        print("Note on artifact copy:", e)

    # Final probe
    res_final = subprocess.run([
        "/opt/homebrew/bin/ffprobe", "-i", OUTPUT_VIDEO,
        "-show_entries", "format=duration,size", "-v", "quiet", "-of", "json"
    ], capture_output=True, text=True)
    print(f"Teaser video details:\n{res_final.stdout}")


def main():
    narration_audio, dur = asyncio.run(generate_speech())
    raw_video = record_teaser_video(dur)
    compose_teaser(raw_video, narration_audio, dur)
    print("\n Standalone 20-Second Cinematic Teaser complete!")


if __name__ == "__main__":
    main()
