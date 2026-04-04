# -*- coding: utf-8 -*-
"""
Analyze the existing Script_5_0164.wav to understand the bleed pattern.
Compare its duration/content with what fresh Whisper runs produce.
"""
import sys, os
import numpy as np
import soundfile as sf

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BLEEDING_WAV = os.path.join(BASE, "datasets", "wavs", "Script_5_0164.wav")
RAW_AUDIO = os.path.join(BASE, "rawdata", "audio", "Script_5_44-200.wav")

# Pipeline parameters
PREATTACK_SILENCE_MS = 400
TAIL_SILENCE_MS = 730
RMS_WINDOW_MS = 10
SILENCE_THRESHOLD_DB = -65


def compute_rms_windowed(samples, sr, window_ms=10):
    window_size = int(sr * window_ms / 1000)
    n_windows = len(samples) // window_size
    if n_windows == 0:
        return np.array([])
    trimmed = samples[:n_windows * window_size].reshape(n_windows, window_size)
    rms_lin = np.sqrt(np.mean(trimmed ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms_lin, 1e-10))


def main():
    print("=" * 70)
    print("ANALYSIS OF EXISTING BLEEDING FILE: Script_5_0164.wav")
    print("=" * 70)

    # Load the bleeding file
    samples, sr = sf.read(BLEEDING_WAV, dtype='float64')
    total_ms = len(samples) / sr * 1000
    print(f"\nFile: {BLEEDING_WAV}")
    print(f"Duration: {total_ms:.0f}ms, SR: {sr}")

    # Determine R6 envelope boundaries
    pre_samples = int(sr * PREATTACK_SILENCE_MS / 1000)
    tail_samples = int(sr * TAIL_SILENCE_MS / 1000)
    voiced = samples[pre_samples:-tail_samples] if tail_samples > 0 else samples[pre_samples:]
    voiced_ms = len(voiced) / sr * 1000
    print(f"Pre-silence: {PREATTACK_SILENCE_MS}ms, Tail-silence: {TAIL_SILENCE_MS}ms")
    print(f"Voiced region: {voiced_ms:.0f}ms ({pre_samples} to {len(samples) - tail_samples} samples)")

    # Compute RMS energy envelope of voiced region
    rms_db = compute_rms_windowed(voiced, sr, RMS_WINDOW_MS)
    window_size = int(sr * RMS_WINDOW_MS / 1000)

    # Find energy valleys (significant drops) in the second half
    print(f"\n--- Energy Envelope Analysis (voiced region, {RMS_WINDOW_MS}ms windows) ---")
    print(f"Total windows: {len(rms_db)}")

    # Find voice onset and offset within the voiced region
    above_thresh = np.where(rms_db >= SILENCE_THRESHOLD_DB)[0]
    if len(above_thresh) > 0:
        voice_start_win = above_thresh[0]
        voice_end_win = above_thresh[-1]
        voice_start_ms = voice_start_win * RMS_WINDOW_MS
        voice_end_ms = (voice_end_win + 1) * RMS_WINDOW_MS
        print(f"Voice detected: {voice_start_ms:.0f}ms to {voice_end_ms:.0f}ms within voiced region")
        print(f"Voice duration: {voice_end_ms - voice_start_ms:.0f}ms")
    else:
        print("No voice above threshold!")
        return

    # Look for energy valleys in the last 40% of the voiced content
    search_start = int(len(rms_db) * 0.6)
    print(f"\n--- Searching for energy valleys in last 40% ({search_start * RMS_WINDOW_MS}ms onwards) ---")

    # Find valleys: windows where energy drops significantly from surrounding
    VALLEY_THRESHOLD = 15  # dB drop from local max
    MIN_VALLEY_DURATION = 3  # windows (30ms)

    in_valley = False
    valley_start = -1
    valleys = []
    local_max = rms_db[search_start] if search_start < len(rms_db) else -100

    for i in range(search_start, len(rms_db)):
        local_max = max(local_max, rms_db[i])
        if rms_db[i] < local_max - VALLEY_THRESHOLD:
            if not in_valley:
                valley_start = i
                in_valley = True
        else:
            if in_valley:
                valley_duration = i - valley_start
                if valley_duration >= MIN_VALLEY_DURATION:
                    valleys.append((valley_start, i, valley_duration))
                in_valley = False
                local_max = rms_db[i]

    if in_valley:
        valley_duration = len(rms_db) - valley_start
        if valley_duration >= MIN_VALLEY_DURATION:
            valleys.append((valley_start, len(rms_db), valley_duration))

    if valleys:
        print(f"Found {len(valleys)} energy valley(s):")
        for vs, ve, vd in valleys:
            ms_start = vs * RMS_WINDOW_MS
            ms_end = ve * RMS_WINDOW_MS
            min_db = np.min(rms_db[vs:ve])
            print(f"  Valley at {ms_start}-{ms_end}ms ({vd * RMS_WINDOW_MS}ms), min energy: {min_db:.1f}dB")
    else:
        print("No significant energy valleys found in last 40%!")

    # Show energy profile of the last 2 seconds in detail
    print(f"\n--- Detailed energy profile: last 2000ms of voiced region ---")
    last_2s_start = max(0, len(rms_db) - 200)  # 200 windows = 2000ms
    for i in range(last_2s_start, len(rms_db), 5):  # every 50ms
        ms = i * RMS_WINDOW_MS
        chunk = rms_db[i:min(i+5, len(rms_db))]
        avg_db = np.mean(chunk)
        bar = "#" * max(0, int((avg_db + 100) / 2))  # visual bar
        marker = " <<< BELOW THRESH" if avg_db < SILENCE_THRESHOLD_DB else ""
        print(f"  {ms:5.0f}ms: {avg_db:6.1f}dB {bar}{marker}")

    # Now extract from raw audio using the fresh Whisper timestamps from diagnostic
    # Pipeline fresh: 975.41-981.11s, Experiment fresh: 974.98-981.68s
    print(f"\n--- Back-calculating original pipeline extraction ---")
    # From the file: voiced_ms tells us how long the Stage 1 extraction was
    # Stage 2 onset/offset detection + safety margins determined voiced region
    # So the raw Stage 1 extraction was AT LEAST voiced_ms long
    print(f"Voiced content in file: {voiced_ms:.0f}ms")

    # If fresh pipeline gives 5700ms extraction and fresh experiment gives 6700ms,
    # but the file has {voiced_ms}ms of voiced content...
    # The original extraction must have included more audio.

    # Let's estimate: the raw extraction had to be long enough that after
    # onset/offset detection + OFFSET_SAFETY_MS=80, we get voiced_ms
    # This means the raw extraction was at least voiced_ms + some trimming

    print(f"\nFresh pipeline extraction would be ~5800ms (5700ms + 50ms pad each side)")
    print(f"Fresh experiment extraction would be ~6800ms (6700ms + 50ms pad each side)")
    print(f"Existing file voiced duration: {voiced_ms:.0f}ms")

    if voiced_ms > 5800 + 200:
        print(f"\n>>> EXISTING FILE IS LONGER than fresh pipeline extraction!")
        print(f">>> Difference: {voiced_ms - 5800:.0f}ms extra")
        print(f">>> This confirms the original pipeline run got DIFFERENT timestamps")
        print(f">>> that extended ~{voiced_ms - 5800:.0f}ms further into the next sentence.")
    elif voiced_ms < 5800 - 200:
        print(f"\n>>> Existing file is shorter than expected. Stage 2 trimmed effectively.")
    else:
        print(f"\n>>> Existing file matches fresh pipeline extraction length.")


if __name__ == "__main__":
    main()
