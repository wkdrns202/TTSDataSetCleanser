# -*- coding: utf-8 -*-
"""
Cross-correlate the pipeline output (Script_5_0164.wav) with raw audio
to find exactly where in the raw timeline it starts and ends.
This definitively shows whether the pipeline output contains next-sentence audio.
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

# Also compare with experiment files
EXP_DIR = os.path.dirname(__file__)

PREATTACK_SILENCE_MS = 400
TAIL_SILENCE_MS = 730


def find_in_raw(raw_data, snippet, sr, search_center_s, search_radius_s=5):
    """Find snippet in raw_data using cross-correlation. Returns offset in samples."""
    # Use first 1 second of snippet for matching (faster)
    match_len = min(int(sr * 1.0), len(snippet))
    template = snippet[:match_len]

    search_start = max(0, int((search_center_s - search_radius_s) * sr))
    search_end = min(len(raw_data), int((search_center_s + search_radius_s) * sr))
    search_region = raw_data[search_start:search_end]

    if len(search_region) < match_len:
        return -1

    # Normalized cross-correlation
    corr = np.correlate(search_region, template, mode='valid')
    best_offset = np.argmax(corr)
    best_corr = corr[best_offset]

    # Normalize
    template_energy = np.sqrt(np.sum(template ** 2))
    if template_energy > 0:
        # Compute local energy at best offset
        local = search_region[best_offset:best_offset + match_len]
        local_energy = np.sqrt(np.sum(local ** 2))
        if local_energy > 0:
            norm_corr = best_corr / (template_energy * local_energy)
        else:
            norm_corr = 0
    else:
        norm_corr = 0

    absolute_sample = search_start + best_offset
    return absolute_sample, norm_corr


def main():
    print("=" * 70)
    print("CROSS-CORRELATION: Finding exact position of pipeline output in raw audio")
    print("=" * 70)

    raw_data, sr = sf.read(RAW_AUDIO, dtype='float64')
    print(f"Raw audio: {len(raw_data)/sr:.1f}s, SR: {sr}")

    # Load pipeline output, extract voiced region (strip R6 envelope)
    pipe_data, pipe_sr = sf.read(BLEEDING_WAV, dtype='float64')
    pre_samples = int(pipe_sr * PREATTACK_SILENCE_MS / 1000)
    tail_samples = int(pipe_sr * TAIL_SILENCE_MS / 1000)
    pipe_voiced = pipe_data[pre_samples:-tail_samples] if tail_samples > 0 else pipe_data[pre_samples:]
    print(f"\nPipeline output voiced: {len(pipe_voiced)/pipe_sr*1000:.0f}ms")

    # Find start position in raw audio (search around 975s based on fresh Whisper)
    print("\nSearching for pipeline voiced content in raw audio...")
    result = find_in_raw(raw_data, pipe_voiced, sr, search_center_s=976, search_radius_s=10)
    if result[1] > 0.5:
        start_sample, corr = result
        start_s = start_sample / sr
        end_s = start_s + len(pipe_voiced) / sr
        print(f"  FOUND! Correlation: {corr:.4f}")
        print(f"  Pipeline voiced starts at: {start_s:.3f}s in raw audio")
        print(f"  Pipeline voiced ends at:   {end_s:.3f}s in raw audio")
        print(f"  Duration: {(end_s - start_s)*1000:.0f}ms")
    else:
        print(f"  Low correlation: {result[1]:.4f} - match uncertain")
        start_s = result[0] / sr
        end_s = start_s + len(pipe_voiced) / sr
        print(f"  Best guess: {start_s:.3f}s to {end_s:.3f}s")

    # Also find experiment offset_80ms file position
    exp_80_path = os.path.join(EXP_DIR, "line164_offset_80ms.wav")
    if os.path.exists(exp_80_path):
        exp_data, exp_sr = sf.read(exp_80_path, dtype='float64')
        exp_voiced = exp_data[pre_samples:-tail_samples] if tail_samples > 0 else exp_data[pre_samples:]
        print(f"\nExperiment offset_80ms voiced: {len(exp_voiced)/exp_sr*1000:.0f}ms")

        result_exp = find_in_raw(raw_data, exp_voiced, sr, search_center_s=976, search_radius_s=10)
        if result_exp[1] > 0.5:
            exp_start = result_exp[0] / sr
            exp_end = exp_start + len(exp_voiced) / sr
            print(f"  FOUND! Correlation: {result_exp[1]:.4f}")
            print(f"  Experiment voiced starts at: {exp_start:.3f}s")
            print(f"  Experiment voiced ends at:   {exp_end:.3f}s")
        else:
            print(f"  Low correlation: {result_exp[1]:.4f}")

    # Show what's in the raw audio around the end of the pipeline output
    print(f"\n--- Raw audio energy around pipeline end ({end_s:.3f}s) ---")
    RMS_WINDOW_MS = 10
    window_size = int(sr * RMS_WINDOW_MS / 1000)

    # Show 500ms before and 500ms after the end
    analysis_start = max(0, int((end_s - 0.5) * sr))
    analysis_end = min(len(raw_data), int((end_s + 0.5) * sr))
    analysis = raw_data[analysis_start:analysis_end]

    n_windows = len(analysis) // window_size
    trimmed = analysis[:n_windows * window_size].reshape(n_windows, window_size)
    rms_lin = np.sqrt(np.mean(trimmed ** 2, axis=1))
    rms_db = 20 * np.log10(np.maximum(rms_lin, 1e-10))

    for i in range(0, len(rms_db), 5):
        ms = (analysis_start / sr + i * RMS_WINDOW_MS / 1000) * 1000
        abs_s = analysis_start / sr + i * RMS_WINDOW_MS / 1000
        chunk = rms_db[i:min(i + 5, len(rms_db))]
        avg_db = np.mean(chunk)
        bar = "#" * max(0, int((avg_db + 100) / 2))
        marker = ""
        if abs(abs_s - end_s) < 0.03:
            marker = " <<< PIPELINE END"
        elif abs(abs_s - start_s) < 0.03:
            marker = " <<< PIPELINE START"
        print(f"  {abs_s:.3f}s ({ms:.0f}ms): {avg_db:6.1f}dB {bar}{marker}")

    # Final comparison
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  Pipeline voiced region in raw: {start_s:.3f}s to {end_s:.3f}s")
    print(f"  Fresh Whisper (pipeline params): 975.410s to 981.110s")
    print(f"  Fresh Whisper (experiment params): 974.980s to 981.680s")

    # What's the next sentence? From fresh Whisper: starts at 983.21s
    # Check if the pipeline output extends past the target sentence end
    # The target sentence should end around 981s based on both Whisper runs
    if end_s > 981.5:
        print(f"\n  >>> Pipeline output extends to {end_s:.3f}s")
        print(f"  >>> This is {(end_s - 981.1)*1000:.0f}ms past the fresh Whisper segment end (981.11s)")
        print(f"  >>> The extra content may be the next sentence's speech")
    elif end_s < 980.5:
        print(f"\n  >>> Pipeline output ends early at {end_s:.3f}s (possible truncation)")
    else:
        print(f"\n  >>> Pipeline output ends at expected position")


if __name__ == "__main__":
    main()
