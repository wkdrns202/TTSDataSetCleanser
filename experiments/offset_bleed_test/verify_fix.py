# -*- coding: utf-8 -*-
"""
Verify the adaptive tail extension fix for line 164.
Simulates what the pipeline would now do vs what it did before.
"""
import sys, os
import numpy as np
import soundfile as sf

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_AUDIO = os.path.join(BASE, "rawdata", "audio", "Script_5_44-200.wav")
OUT_DIR = os.path.dirname(__file__)

# Parameters
AUDIO_PAD_MS = 50
TAIL_EXTEND_MAX_MS = 400
MIN_GAP_FOR_PAD_MS = 30
SILENCE_THRESHOLD_DB = -65
RMS_WINDOW_MS = 10
ONSET_SAFETY_MS = 30
OFFSET_SAFETY_MS = 80
FADE_MS = 10
PEAK_NORMALIZE_DB = -1.0
PREATTACK_SILENCE_MS = 400
TAIL_SILENCE_MS = 730


def compute_rms_windowed(samples, sr, window_ms=10):
    window_size = int(sr * window_ms / 1000)
    n_windows = len(samples) // window_size
    if n_windows == 0:
        return np.array([])
    trimmed = samples[:n_windows * window_size].reshape(n_windows, window_size)
    rms_lin = np.sqrt(np.mean(trimmed ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms_lin, 1e-10))


def find_voice_onset_offset(samples, sr):
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


def process_with_params(raw_data, sr, seg_start_s, seg_end_s, right_pad_ms, label):
    """Extract and process with given right padding."""
    start_sample = max(0, int(sr * (seg_start_s - AUDIO_PAD_MS / 1000)))
    end_sample = min(len(raw_data), int(sr * (seg_end_s + right_pad_ms / 1000)))
    samples = raw_data[start_sample:end_sample].copy().astype(np.float64)

    extraction_ms = len(samples) / sr * 1000
    extraction_end_s = end_sample / sr

    # Check energy at extraction boundary
    rms_db = compute_rms_windowed(samples, sr, RMS_WINDOW_MS)
    if len(rms_db) > 0:
        end_energy = rms_db[-1]
    else:
        end_energy = -100

    # Stage 2: voice onset/offset
    onset, offset = find_voice_onset_offset(samples, sr)
    onset_safety = int(sr * ONSET_SAFETY_MS / 1000)
    offset_safety = int(sr * OFFSET_SAFETY_MS / 1000)
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

    # Check energy at voiced region end (before R6)
    voiced_rms = compute_rms_windowed(voiced, sr, RMS_WINDOW_MS)
    voiced_end_energy = voiced_rms[-1] if len(voiced_rms) > 0 else -100

    # R6 envelope
    pre_silence = np.zeros(int(sr * PREATTACK_SILENCE_MS / 1000), dtype=np.float64)
    tail_silence = np.zeros(int(sr * TAIL_SILENCE_MS / 1000), dtype=np.float64)
    final = np.concatenate([pre_silence, voiced, tail_silence])

    voiced_ms = len(voiced) / sr * 1000

    print(f"\n  [{label}]")
    print(f"    Right padding: {right_pad_ms}ms")
    print(f"    Extraction: {seg_start_s - AUDIO_PAD_MS/1000:.3f}s to {extraction_end_s:.3f}s ({extraction_ms:.0f}ms)")
    print(f"    Energy at extraction boundary: {end_energy:.1f}dB", end="")
    if end_energy > SILENCE_THRESHOLD_DB:
        print(f"  ** STILL SPEECH - Stage 2 cannot trim properly **")
    else:
        print(f"  (silence - Stage 2 can trim naturally)")
    print(f"    Voice onset/offset: {onset/sr*1000:.0f}ms to {offset/sr*1000:.0f}ms")
    print(f"    Voiced duration: {voiced_ms:.0f}ms")
    print(f"    Energy at voiced end: {voiced_end_energy:.1f}dB")
    print(f"    Total output: {len(final)/sr*1000:.0f}ms")

    return final


def main():
    print("=" * 70)
    print("VERIFY FIX: Adaptive tail extension for line 164")
    print("=" * 70)

    raw_data, sr = sf.read(RAW_AUDIO, dtype='float64')

    # Use fresh pipeline timestamps (from diagnostic)
    seg_start_s = 975.410
    seg_end_s = 981.110
    next_seg_start_s = 983.210
    gap_ms = (next_seg_start_s - seg_end_s) * 1000

    print(f"\nWhisper segment: {seg_start_s:.3f}s to {seg_end_s:.3f}s")
    print(f"Gap to next segment: {gap_ms:.0f}ms")

    # OLD behavior: fixed 50ms right pad
    old_right_pad = AUDIO_PAD_MS  # 50ms
    old_final = process_with_params(raw_data, sr, seg_start_s, seg_end_s, old_right_pad,
                                     "OLD: Fixed 50ms pad")
    old_path = os.path.join(OUT_DIR, "verify_OLD_50ms_pad.wav")
    sf.write(old_path, old_final, sr, subtype='PCM_24')

    # NEW behavior: adaptive extension
    right_gap = gap_ms
    if right_gap < MIN_GAP_FOR_PAD_MS:
        new_right_pad = 0
    else:
        max_extend = min(TAIL_EXTEND_MAX_MS, right_gap - 20)
        new_right_pad = max(AUDIO_PAD_MS, max_extend)

    new_final = process_with_params(raw_data, sr, seg_start_s, seg_end_s, new_right_pad,
                                     f"NEW: Adaptive {new_right_pad}ms pad")
    new_path = os.path.join(OUT_DIR, "verify_NEW_adaptive_pad.wav")
    sf.write(new_path, new_final, sr, subtype='PCM_24')

    # Also test with a moderate extension
    mod_right_pad = 200
    mod_final = process_with_params(raw_data, sr, seg_start_s, seg_end_s, mod_right_pad,
                                     f"MODERATE: {mod_right_pad}ms pad")
    mod_path = os.path.join(OUT_DIR, "verify_MODERATE_200ms_pad.wav")
    sf.write(mod_path, mod_final, sr, subtype='PCM_24')

    print(f"\n{'=' * 70}")
    print("FILES SAVED:")
    print(f"  {old_path}")
    print(f"  {new_path}")
    print(f"  {mod_path}")
    print(f"\nListen and compare. The OLD file should have the same abrupt ending")
    print(f"as the current pipeline output. The NEW file should trail off naturally.")


if __name__ == "__main__":
    main()
