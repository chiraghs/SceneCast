#!/usr/bin/env python3
"""
AdVision 2.5-Minute Meta Hackathon Master Video Generator
- Professional voiceover via Microsoft Edge Neural TTS (en-US-AndrewMultilingualNeural)
- Target duration: ~150 seconds (~2.5 minutes)
- Full 1080p automated recording via Playwright
- Structure:
  1. Introduction & Dashboard Overview (http://localhost:3001)
  2. Spatial Studio & Computer Vision Homography (http://localhost:3001/studio)
  3. Meta AI Intent Synthesizer & Graph API Generator (http://localhost:3001/targeting)
  4. Generative Creative Ad Studio (http://localhost:3001/ad-studio)
  5. 4-Way Synchronized Persona Stream Compare (http://localhost:3001/compare)
  6. In-Scene Analytics & Brand Recall (http://localhost:3001/analytics)
"""

import os
import sys
import time
import asyncio
import subprocess
import shutil

BASE_DIR = "/Volumes/DiskD/Nanda/Advision - Scenecast"
TEMP_DIR = os.path.join(BASE_DIR, "scripts", "video_temp")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "advision-meta-demo-2.5min.mp4")
ARTIFACT_VIDEO = "/Users/chiraghs/.gemini/antigravity/brain/794021fa-b530-4b49-9ed7-5589a1ec4ece/advision-meta-demo-2.5min.mp4"

os.makedirs(TEMP_DIR, exist_ok=True)

NARRATION_SECTIONS = [
    {
        "id": "part1_teaser_tv_hook",
        "title": "The TV Movie Experience & In-Scene Placement",
        "text": (
            "Tired of loud, unskippable commercials interrupting the climax of your favorite movie? "
            "AdVision solves this by embedding personalized brand campaigns naturally inside the scene itself—like this RepBull Energy placement. "
            "Zero interruptions. Pure cinematic immersion. Powered by Meta AI."
        ),
    },
    {
        "id": "part2_spatial_studio",
        "title": "Spatial Studio Video Upload & Auto Surface Mapping",
        "text": (
            "Here in the Spatial Studio, we demonstrate live video ingestion. "
            "Watch as we upload raw footage of Night Highway. "
            "AdVision immediately loads the video locally and initializes Meta SAM 3.1. "
            "Within seconds, the computer vision engine scans the footage, detects high-contrast planar signage, "
            "and automatically locks four-corner homography pins onto the roadside display with ninety-nine percent certainty. "
            "It computes the sub-pixel 3D perspective matrix, calibrates to ambient lux levels, and configures the editorial prominence cap, "
            "completely automating what used to require weeks of manual VFX rotoscoping."
        ),
    },
    {
        "id": "part3_intent_targeting",
        "title": "Meta AI Intent Synthesizer & Graph API Generator",
        "text": (
            "Under Intent Targeting, we introduce the Meta AI Intent Synthesizer, powered by Meta Llama 3.3. "
            "Instead of relying on tracking cookies, advertisers input a natural language intent prompt, such as "
            "College students cramming for finals at 2 AM craving high-caffeine energy drinks. "
            "Meta AI extracts psychological triggers, scores in-scene receptivity at ninety-five percent, "
            "and compiles an exact Meta Ads Manager Graph API version twenty-one payload with official Meta Audience Network interest tags ready for deployment."
        ),
    },
    {
        "id": "part4_ad_studio",
        "title": "Generative Creative Ad Studio",
        "text": (
            "Our built-in Ad Studio takes audience intent and automatically synthesizes production-grade creatives using Meta Muse generative vision models. "
            "It outputs one-to-one square feed units, four-to-five portrait posts, and nine-to-sixteen vertical formats. "
            "It dynamically localizes headlines and cultural copy across English, Japanese, Telugu, and French, "
            "while running automated brand safety and contrast verification before dispatching to the ad delivery server."
        ),
    },
    {
        "id": "part5_compare_sync",
        "title": "4-Way Synchronized Persona Stream Compare",
        "text": (
            "In our 4-Way Persona Sync view, all viewports stream in locked frame synchronization. "
            "One scene, four viewers, and zero storyline interruption. "
            "Alex in the United States sees RepBull Energy. "
            "Ren in Tokyo sees a Japanese Sony Alpha camera. "
            "Aarav in India sees MRF performance tires, while Leo and Mia on a kids profile receive a public-service civic announcement. "
            "All overlays render client-side at sixty frames per second using hardware-accelerated CSS 3D transforms."
        ),
    },
    {
        "id": "part6_analytics",
        "title": "Ad Scene Campaign Analytics & Conclusion",
        "text": (
            "Finally, under Ad Scene Campaign Analytics, the telemetry dashboard tracks viewability, "
            "engagement rates, brand recall lift, and effective CPMs across global markets. "
            "By uniting Meta's open-weight AI models with the reach of Meta Audience Network, "
            "AdVision unlocks billions of dollars in non-intrusive in-stream advertising for streaming platforms worldwide. "
            "Thank you."
        ),
    },
]


async def generate_speech_tracks():
    import edge_tts
    print("\n--- [Phase 1/4] Generating Studio-Quality Neural Voiceover via Edge TTS ---")
    durations = []
    
    for idx, sec in enumerate(NARRATION_SECTIONS):
        audio_file = os.path.join(TEMP_DIR, f"meta_narration_{idx:02d}_{sec['id']}.mp3")
        print(f"Generating audio {idx+1}/{len(NARRATION_SECTIONS)}: {sec['title']}...")
        communicate = edge_tts.Communicate(sec["text"], "en-US-AndrewMultilingualNeural", rate="+6%")
        await communicate.save(audio_file)
        
        # Probe exact duration
        res = subprocess.run([
            "/opt/homebrew/bin/ffprobe", "-i", audio_file,
            "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"
        ], capture_output=True, text=True)
        dur = float(res.stdout.strip())
        durations.append(dur)
        sec["audio_file"] = audio_file
        sec["duration"] = dur
        print(f"  -> Duration: {dur:.2f}s")
        
    total_dur = sum(durations)
    print(f"\nTotal Narration Duration: {total_dur:.2f} seconds ({total_dur/60:.2f} minutes)")
    return durations


def record_playwright_video(durations):
    from playwright.sync_api import sync_playwright
    print("\n--- [Phase 2/4] Recording 1080p Interactive UI Flow via Playwright ---")
    
    record_dir = os.path.join(TEMP_DIR, "meta_raw_capture")
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

        # =====================================================================
        # PART 1: The TV Movie Experience & In-Scene Red Bull Placement (/teaser.html)
        # =====================================================================
        t0 = time.time()
        target_dur1 = durations[0]
        print(f"[Part 1] TV Movie Experience with In-Scene Red Bull Placement (Target: {target_dur1:.2f}s)...")
        page.goto("http://localhost:3001/teaser.html")
        page.wait_for_load_state("networkidle")
        print("Teaser TV player running (movie -> pop-up ad -> dissolves to in-scene Red Bull billboard)...")

        # Let the teaser sequence execute (20s)
        time.sleep(target_dur1)

        elapsed = time.time() - t0
        rem = max(0.5, target_dur1 - elapsed)
        time.sleep(rem)

        # =====================================================================
        # PART 2: Spatial Studio Video Upload & Auto Surface Mapping (/studio)
        # =====================================================================
        t0 = time.time()
        target_dur2 = durations[1]
        print(f"[Part 2] Spatial Studio Video Upload & Auto Surface Mapping (Target: {target_dur2:.2f}s)...")
        page.goto("http://localhost:3001/studio")
        page.wait_for_load_state("networkidle")
        time.sleep(2.0)

        # 1. Upload custom video: demo-videos/Night Highway.mp4
        night_highway_path = "/Volumes/DiskD/Nanda/Advision - Scenecast/demo-videos/Night Highway.mp4"
        try:
            input_file = page.locator("input[type='file']").first
            input_file.set_input_files(night_highway_path)
            print(">>> UPLOADED CUSTOM VIDEO: Night Highway.mp4 <<<")
            time.sleep(1.5)
        except Exception as e:
            print("Video upload note:", e)

        # 2. Wait for auto-scouting status to complete
        time.sleep(3.5)

        # 3. Hover over the video player with newly mapped pins
        try:
            page.locator("video").first.hover()
            time.sleep(3.0)
        except Exception as e:
            print("Video hover note:", e)

        # 4. Hover over the AI Surface Scorecard and Prominence Cap
        try:
            scorecard = page.locator("text=Scorecard, text=SAM 3.1, text=Certainty").first
            if scorecard.is_visible():
                scorecard.hover()
                print("Hovered over AI Surface Scorecard")
                time.sleep(2.5)
        except Exception as e:
            print("Scorecard hover note:", e)

        # 5. Click "Save Rules & Zones"
        try:
            save_btn = page.locator("button:has-text('Save Rules & Zones')").first
            if save_btn.is_visible():
                save_btn.click()
                print("Saved mapped zones and rules")
                time.sleep(1.5)
        except Exception as e:
            print("Save btn note:", e)

        page.evaluate("window.scrollBy({ top: 220, behavior: 'smooth' });")
        time.sleep(2.0)
        page.evaluate("window.scrollBy({ top: -220, behavior: 'smooth' });")
        time.sleep(1.5)

        elapsed = time.time() - t0
        rem = max(0.5, target_dur2 - elapsed)
        time.sleep(rem)

        # =====================================================================
        # PART 3: Meta AI Intent Synthesizer (/targeting)
        # =====================================================================
        t0 = time.time()
        target_dur3 = durations[2]
        print(f"[Part 3] Meta AI Intent Synthesizer (Target: {target_dur3:.2f}s)...")
        page.goto("http://localhost:3001/targeting")
        page.wait_for_load_state("networkidle")
        time.sleep(2.0)

        # Scroll smoothly to the Meta AI Intent Synthesizer section
        page.evaluate("window.scrollBy({ top: 320, behavior: 'smooth' });")
        time.sleep(1.5)

        # Click "Late-Night Energy" preset
        try:
            energy_preset = page.locator("button:has-text('Late-Night Energy')").first
            if energy_preset.is_visible():
                energy_preset.click()
                print("Clicked preset: Late-Night Energy")
                time.sleep(1.0)
        except Exception as e:
            print("Preset click note:", e)

        # Click "Synthesize Intent & Target" button
        try:
            synth_btn = page.locator("button:has-text('Synthesize Intent & Target')").first
            if synth_btn.is_visible():
                synth_btn.click()
                print("Triggered Meta AI Intent Synthesis...")
                time.sleep(3.0)
        except Exception as e:
            print("Synth click note:", e)

        # Scroll slightly to show Graph API payload
        page.evaluate("window.scrollBy({ top: 180, behavior: 'smooth' });")
        time.sleep(2.0)

        # Click to expand Graph API payload
        try:
            graph_toggle = page.locator("button:has-text('Meta Ads Manager Graph API Payload')").first
            if graph_toggle.is_visible():
                graph_toggle.click()
                print("Expanded Graph API JSON")
                time.sleep(1.5)
                # Click copy button
                copy_btn = page.locator("button:has-text('Copy Graph API JSON')").first
                if copy_btn.is_visible():
                    copy_btn.click()
                    print("Clicked Copy Graph API JSON")
                    time.sleep(1.5)
        except Exception as e:
            print("Graph toggle note:", e)

        elapsed = time.time() - t0
        rem = max(0.5, target_dur3 - elapsed)
        time.sleep(rem)

        # =====================================================================
        # PART 4: Generative Creative Ad Studio (/ad-studio)
        # =====================================================================
        t0 = time.time()
        target_dur4 = durations[3]
        print(f"[Part 4] Generative Ad Studio (Target: {target_dur4:.2f}s)...")
        page.goto("http://localhost:3001/ad-studio")
        page.wait_for_load_state("networkidle")
        time.sleep(2.0)

        # Pick preset: Sony Alpha Optics or RepBull
        try:
            preset_btn = page.locator("button:has-text('Sony Alpha Optics'), button:has-text('RepBull Energy')").first
            if preset_btn.is_visible():
                preset_btn.click()
                print("Selected Ad Studio Brand Preset")
                time.sleep(1.5)
        except Exception as e:
            print("Ad studio preset note:", e)

        # Click Generate Creative Variants
        try:
            gen_btn = page.locator("button:has-text('Generate Creative Variants'), button:has-text('Generate Creatives')").first
            if gen_btn.is_visible():
                gen_btn.click()
                print("Generating multi-aspect creative variants...")
                time.sleep(3.5)
        except Exception as e:
            print("Gen btn note:", e)

        # Scroll to show 1:1, 4:5, 9:16 aspect previews
        page.evaluate("window.scrollBy({ top: 350, behavior: 'smooth' });")
        time.sleep(3.0)

        # Click approve
        try:
            appr_btn = page.locator("button:has-text('Approve & Send'), button:has-text('Approve')").first
            if appr_btn.is_visible():
                appr_btn.click()
                print("Approved creative")
                time.sleep(1.5)
        except Exception as e:
            print("Approve note:", e)

        elapsed = time.time() - t0
        rem = max(0.5, target_dur4 - elapsed)
        time.sleep(rem)

        # =====================================================================
        # PART 5: 4-Way Synchronized Persona Stream Compare (/compare)
        # =====================================================================
        t0 = time.time()
        target_dur5 = durations[4]
        print(f"[Part 5] 4-Way Synchronized Stream Compare (Target: {target_dur5:.2f}s)...")
        page.goto("http://localhost:3001/compare")
        page.wait_for_load_state("networkidle")
        time.sleep(2.0)

        # Ensure video is playing
        try:
            play_btn = page.locator("button:has-text('Play')").first
            if play_btn.is_visible():
                play_btn.click()
                print("Started synchronized 4-way stream playback")
                time.sleep(1.0)
        except Exception as e:
            print("Play btn note:", e)

        # Hover over top-left viewport (Alex / RepBull)
        try:
            viewports = page.locator(".aspect-video")
            if viewports.count() >= 2:
                viewports.nth(0).hover()
                time.sleep(3.0)
                viewports.nth(1).hover()
                time.sleep(3.0)
        except Exception as e:
            print("Viewport hover note:", e)

        # Smooth scroll down to show bottom viewports (Aarav & Kids PSA)
        page.evaluate("window.scrollBy({ top: 280, behavior: 'smooth' });")
        time.sleep(3.5)

        try:
            viewports = page.locator(".aspect-video")
            if viewports.count() >= 4:
                viewports.nth(2).hover()
                time.sleep(2.5)
                viewports.nth(3).hover()
                time.sleep(2.5)
        except Exception as e:
            print("Bottom viewports note:", e)

        page.evaluate("window.scrollBy({ top: -280, behavior: 'smooth' });")
        time.sleep(2.0)

        elapsed = time.time() - t0
        rem = max(0.5, target_dur5 - elapsed)
        time.sleep(rem)

        # =====================================================================
        # PART 6: Ad Scene Campaign Analytics & Conclusion (/analytics)
        # =====================================================================
        t0 = time.time()
        target_dur6 = durations[5]
        print(f"[Part 6] Ad Scene Campaign Analytics & Conclusion (Target: {target_dur6:.2f}s)...")
        page.goto("http://localhost:3001/analytics")
        page.wait_for_load_state("networkidle")
        time.sleep(2.0)

        # Highlight the new "Ad Scene Campaign Analytics" heading
        try:
            heading = page.locator("h1").first
            if heading.is_visible():
                heading.hover()
                print("Hovered over Ad Scene Campaign Analytics heading")
                time.sleep(2.0)
        except Exception as e:
            print("Heading hover note:", e)

        # Hover over metrics
        try:
            page.locator(".grid").first.hover()
            time.sleep(2.5)
        except Exception as e:
            print("Analytics hover note:", e)

        page.evaluate("window.scrollBy({ top: 320, behavior: 'smooth' });")
        time.sleep(3.0)
        page.evaluate("window.scrollBy({ top: -320, behavior: 'smooth' });")
        time.sleep(1.5)

        elapsed = time.time() - t0
        rem = max(0.5, target_dur6 - elapsed)
        time.sleep(rem)

        # Finish recording
        print("Closing browser context and finalizing capture...")
        page.close()
        context.close()
        browser.close()

    recorded_files = [
        os.path.join(record_dir, f)
        for f in os.listdir(record_dir)
        if f.endswith(".webm") or f.endswith(".mp4")
    ]
    if not recorded_files:
        raise RuntimeError("No raw recording file found from Playwright.")
    
    raw_video = recorded_files[0]
    print(f"Raw recording saved: {raw_video}")
    return raw_video


def compose_master_video(raw_video):
    print("\n--- [Phase 3/4] Composing Master Narration & Cinematic Audio ---")
    concat_list_path = os.path.join(TEMP_DIR, "meta_audio_concat.txt")
    with open(concat_list_path, "w") as f:
        for sec in NARRATION_SECTIONS:
            f.write(f"file '{sec['audio_file']}'\n")

    master_narration = os.path.join(TEMP_DIR, "meta_master_narration.mp3")
    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list_path, "-c", "copy", master_narration
    ], check=True)

    # Get master narration length
    res = subprocess.run([
        "/opt/homebrew/bin/ffprobe", "-i", master_narration,
        "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"
    ], capture_output=True, text=True)
    narration_duration = float(res.stdout.strip())
    print(f"Master narration duration: {narration_duration:.2f}s ({narration_duration/60:.2f} min)")

    # Generate warm, gentle 432 Hz harmonic ambient pad
    ambient_audio = os.path.join(TEMP_DIR, "meta_ambient_pad.mp3")
    fade_out_start = max(0, narration_duration - 4)
    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-y", "-f", "lavfi",
        "-i", f"sine=frequency=432:sample_rate=48000:duration={narration_duration + 3}",
        "-af", f"lowpass=f=450,volume=0.028,afade=t=in:ss=0:d=3,afade=t=out:st={fade_out_start:.2f}:d=4",
        ambient_audio
    ], check=True)

    # Mix narration with ambient track
    mixed_audio = os.path.join(TEMP_DIR, "meta_mixed_soundtrack.mp3")
    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-y",
        "-i", master_narration,
        "-i", ambient_audio,
        "-filter_complex", "[0:a]volume=1.0[v];[1:a]volume=0.6[a];[v][a]amix=inputs=2:duration=first:dropout_transition=2[out]",
        "-map", "[out]", "-c:a", "libmp3lame", "-b:a", "192k",
        mixed_audio
    ], check=True)

    print("\n--- [Phase 4/4] Muxing 1080p Video and Master Soundtrack ---")
    
    res_v = subprocess.run([
        "/opt/homebrew/bin/ffprobe", "-i", raw_video,
        "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"
    ], capture_output=True, text=True)
    video_duration = float(res_v.stdout.strip())
    print(f"Raw video duration: {video_duration:.2f}s")

    target_duration = 180.0
    speed_factor = video_duration / target_duration
    print(f"Master Target Duration: exactly {target_duration:.2f}s (3.00 minutes)")
    print(f"Synchronizing video speed factor: {speed_factor:.4f}")

    # Build broadcast-grade H.264 1080p MP4
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

    print(f"\n Master broadcast video generated: {OUTPUT_VIDEO}")
    
    # Also save to conversation artifacts directory
    try:
        shutil.copy2(OUTPUT_VIDEO, ARTIFACT_VIDEO)
        print(f" Artifact copy saved: {ARTIFACT_VIDEO}")
    except Exception as e:
        print("Note: Could not copy to artifact directory:", e)

    # Print final video info
    res_final = subprocess.run([
        "/opt/homebrew/bin/ffprobe", "-i", OUTPUT_VIDEO,
        "-show_entries", "format=duration,size,bit_rate", "-v", "quiet", "-of", "json"
    ], capture_output=True, text=True)
    print(f"Final video details:\n{res_final.stdout}")


def main():
    durations = asyncio.run(generate_speech_tracks())
    raw_video = record_playwright_video(durations)
    compose_master_video(raw_video)
    print("\n AdVision 2.5-Minute Master Demo Video successfully created!")


if __name__ == "__main__":
    main()
