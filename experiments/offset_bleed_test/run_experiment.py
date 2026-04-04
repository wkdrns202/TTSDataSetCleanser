# -*- coding: utf-8 -*-
"""
Tiny experiment: Re-extract Script_5 line 164 from raw audio
with different OFFSET_SAFETY_MS values to find the right tail trim.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import static_ffmpeg
static_ffmpeg.add_paths()
import whisper

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_AUDIO = os.path.join(BASE, "rawdata", "audio", "Script_5_44-200.wav")
OUT_DIR = os.path.dirname(__file__)

# The ground-truth text for line 164
TARGET_LINE = 164
TARGET_TEXT = "경찰관 한 명이 차에서 내려 나에게 다가온다 괜찮으십니까 무슨 일이십니까 하고 묻는다."

# Pipeline parameters (same as align_and_split.py)
AUDIO_PAD_MS = 50
SILENCE_THRESHOLD_DB = -65
RMS_WINDOW_MS = 10
ONSET_SAFETY_MS = 30
FADE_MS = 10
PEAK_NORMALIZE_DB = -1.0
PREATTACK_SILENCE_MS = 400
TAIL_SILENCE_MS = 730

# Values to test
OFFSET_SAFETY_VALUES = [0, 20, 40, 60, 80]


def compute_rms_windowed(samples, sr=48000, window_ms=10):
    window_size = int(sr * window_ms / 1000)
    n_windows = len(samples) // window_size
    if n_windows == 0:
        rms_lin = np.sqrt(np.mean(samples**2))
        return np.array([20 * np.log10(max(rms_lin, 1e-10))])
    trimmed = samples[:n_windows * window_size].reshape(n_windows, window_size)
    rms_lin = np.sqrt(np.mean(trimmed ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms_lin, 1e-10))


def find_voice_onset_offset(samples, sr=48000):
    rms_db = compute_rms_windowed(samples, sr, RMS_WINDOW_MS)
    window_size = int(sr * RMS_WINDOW_MS / 1000)
    voiced = np.where(rms_db >= SILENCE_THRESHOLD_DB)[0]
    if len(voiced) == 0:
        return 0, len(samples)
    onset_sample = voiced[0] * window_size
    offset_sample = min((voiced[-1] + 1) * window_size, len(samples))
    return onset_sample, offset_sample


def make_raised_cosine_fade(length):
    if length <= 0:
        return np.array([], dtype=np.float64)
    return 0.5 * (1 - np.cos(np.pi * np.arange(length) / length))


def process_segment(raw_data, sr, seg_start_ms, seg_end_ms, next_seg_start_ms, offset_safety_ms):
    """Extract and post-process one segment, mimicking align_and_split.py pipeline."""
    # Stage 1: Extract with padding
    gap_after = next_seg_start_ms - seg_end_ms if next_seg_start_ms else 9999
    pad_after = AUDIO_PAD_MS if gap_after >= 30 else 0  # MIN_GAP_FOR_PAD_MS=30

    start_sample = max(0, int(sr * (seg_start_ms - AUDIO_PAD_MS) / 1000))
    end_sample = min(len(raw_data), int(sr * (seg_end_ms + pad_after) / 1000))
    samples = raw_data[start_sample:end_sample].copy().astype(np.float64)

    # Stage 2: Voice onset/offset detection
    onset, offset = find_voice_onset_offset(samples, sr)
    onset_safety = int(sr * ONSET_SAFETY_MS / 1000)
    offset_safety = int(sr * offset_safety_ms / 1000)
    onset = max(0, onset - onset_safety)
    offset = min(len(samples), offset + offset_safety)
    voiced = samples[onset:offset]

    # Fade
    fade_samples = int(sr * FADE_MS / 1000)
    fade_in = min(fade_samples, len(voiced) // 4)
    fade_out = min(fade_samples, len(voiced) // 4)
    if fade_in > 0:
        voiced[:fade_in] *= make_raised_cosine_fade(fade_in)
    if fade_out > 0:
        voiced[-fade_out:] *= make_raised_cosine_fade(fade_out)[::-1]

    # Normalize
    peak = np.max(np.abs(voiced))
    if peak > 0:
        voiced = voiced * (10 ** (PEAK_NORMALIZE_DB / 20) / peak)

    # R6 envelope
    pre_silence = np.zeros(int(sr * PREATTACK_SILENCE_MS / 1000), dtype=np.float64)
    tail_silence = np.zeros(int(sr * TAIL_SILENCE_MS / 1000), dtype=np.float64)
    final = np.concatenate([pre_silence, voiced, tail_silence])

    return final, onset, offset, len(samples)


def main():
    print(f"Loading raw audio: {RAW_AUDIO}")
    raw_data, sr = sf.read(RAW_AUDIO)
    raw_data = raw_data.astype(np.float64)
    print(f"  Duration: {len(raw_data)/sr:.1f}s, SR: {sr}")

    print(f"\nRunning Whisper medium transcription...")
    model = whisper.load_model("medium", device="cuda")
    result = model.transcribe(RAW_AUDIO, language="ko", condition_on_previous_text=False)
    segments = result["segments"]
    print(f"  Got {len(segments)} segments")

    # Find the segment matching line 164
    # Line 164 is relative to script start (line 1).
    # In this raw file (lines 44-200), line 164 is the 121st line (164-44+1=121)
    # But we match by text similarity
    import difflib
    import re

    def normalize(text):
        return re.sub(r'[^가-힣a-zA-Z0-9]', '', text)

    target_norm = normalize(TARGET_TEXT)
    best_seg = None
    best_sim = 0
    best_idx = -1

    for i, seg in enumerate(segments):
        seg_norm = normalize(seg["text"])
        sim = difflib.SequenceMatcher(None, target_norm, seg_norm).ratio()
        if sim > best_sim:
            best_sim = sim
            best_seg = seg
            best_idx = i

    print(f"\nBest match for line 164:")
    print(f"  Segment #{best_idx}: sim={best_sim:.3f}")
    print(f"  Whisper: {best_seg['text']}")
    print(f"  GT:      {TARGET_TEXT}")
    print(f"  Start:   {best_seg['start']:.3f}s")
    print(f"  End:     {best_seg['end']:.3f}s")

    seg_start_ms = best_seg["start"] * 1000
    seg_end_ms = best_seg["end"] * 1000

    # Find next segment start (for gap calculation)
    next_start_ms = None
    if best_idx + 1 < len(segments):
        next_start_ms = segments[best_idx + 1]["start"] * 1000
        gap = next_start_ms - seg_end_ms
        print(f"  Next seg starts: {next_start_ms:.0f}ms (gap: {gap:.0f}ms)")

    # Extract with different OFFSET_SAFETY_MS values
    print(f"\n{'='*60}")
    print(f"Extracting with different OFFSET_SAFETY_MS values:")
    print(f"{'='*60}")

    for offset_val in OFFSET_SAFETY_VALUES:
        final, onset, offset, raw_len = process_segment(
            raw_data, sr, seg_start_ms, seg_end_ms, next_start_ms, offset_val
        )
        out_path = os.path.join(OUT_DIR, f"line164_offset_{offset_val}ms.wav")
        sf.write(out_path, final, sr, subtype='PCM_24')

        voiced_dur = (offset - onset) / sr * 1000
        total_dur = len(final) / sr * 1000
        print(f"  OFFSET_SAFETY={offset_val:3d}ms -> voiced={voiced_dur:.0f}ms, "
              f"total={total_dur:.0f}ms, onset={onset/sr*1000:.0f}ms, offset={offset/sr*1000:.0f}ms")

    # Also save the raw extraction (no Stage 2 processing) for reference
    pad_after = AUDIO_PAD_MS if (next_start_ms and next_start_ms - seg_end_ms >= 30) else 0
    start_sample = max(0, int(sr * (seg_start_ms - AUDIO_PAD_MS) / 1000))
    end_sample = min(len(raw_data), int(sr * (seg_end_ms + pad_after) / 1000))
    raw_extract = raw_data[start_sample:end_sample]
    raw_path = os.path.join(OUT_DIR, "line164_raw_extract.wav")
    sf.write(raw_path, raw_extract, sr, subtype='PCM_24')
    print(f"\n  Raw extraction (no Stage 2): {raw_path}")
    print(f"  Duration: {len(raw_extract)/sr*1000:.0f}ms")

    print(f"\nDone! Listen to files in: {OUT_DIR}")


if __name__ == "__main__":
    main()
