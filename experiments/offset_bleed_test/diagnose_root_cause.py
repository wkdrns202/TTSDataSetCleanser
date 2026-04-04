# -*- coding: utf-8 -*-
"""
Diagnostic: Compare Whisper timestamps for line 164 between
pipeline parameters and experiment parameters to find ROOT CAUSE of bleeding.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import soundfile as sf
import difflib
import re
import warnings
warnings.filterwarnings("ignore")

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import static_ffmpeg
static_ffmpeg.add_paths()
import whisper

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_AUDIO = os.path.join(BASE, "rawdata", "audio", "Script_5_44-200.wav")
TARGET_TEXT = "경찰관 한 명이 차에서 내려 나에게 다가온다 괜찮으십니까 무슨 일이십니까 하고 묻는다."

def normalize_text(text):
    return re.sub(r'[^가-힣a-zA-Z0-9]', '', text)


def refine_boundaries_with_words(merged_segs, gt_text, normalize_fn):
    """Exact copy from align_and_split.py"""
    all_words = []
    for s in merged_segs:
        words = s.get('words', [])
        if words:
            all_words.extend(words)

    if len(all_words) < 2:
        return None

    norm_gt = normalize_fn(gt_text)
    if len(norm_gt) < 3:
        return None

    best_start_idx = 0
    best_start_sim = 0.0

    for start_idx in range(min(len(all_words), 8)):
        suffix_text = ''.join(w.get('word', '') for w in all_words[start_idx:])
        norm_suffix = normalize_fn(suffix_text)
        if len(norm_suffix) < len(norm_gt) * 0.4:
            break
        sim = difflib.SequenceMatcher(None, norm_suffix, norm_gt).ratio()
        if sim > best_start_sim:
            best_start_sim = sim
            best_start_idx = start_idx

    best_end_idx = len(all_words) - 1
    best_end_sim = 0.0

    for end_idx in range(len(all_words) - 1, max(best_start_idx, len(all_words) - 8) - 1, -1):
        prefix_text = ''.join(w.get('word', '') for w in all_words[best_start_idx:end_idx + 1])
        norm_prefix = normalize_fn(prefix_text)
        if len(norm_prefix) < len(norm_gt) * 0.4:
            break
        sim = difflib.SequenceMatcher(None, norm_prefix, norm_gt).ratio()
        if sim > best_end_sim:
            best_end_sim = sim
            best_end_idx = end_idx

    full_text = ''.join(w.get('word', '') for w in all_words)
    norm_full = normalize_fn(full_text)
    full_sim = difflib.SequenceMatcher(None, norm_full, norm_gt).ratio()

    if best_start_sim > full_sim + 0.02 or best_end_sim > full_sim + 0.02:
        refined_start = all_words[best_start_idx].get('start', all_words[0].get('start'))
        refined_end = all_words[best_end_idx].get('end', all_words[-1].get('end'))
        return refined_start, refined_end, all_words, best_start_idx, best_end_idx

    return None


def find_best_match(segments, target_text, max_merge=5):
    """Find best matching segment(s) for target text."""
    norm_target = normalize_text(target_text)
    best_sim = 0
    best_idx = -1
    best_merge = 1
    best_text = ""

    for i in range(len(segments)):
        for merge_count in range(1, min(max_merge + 1, len(segments) - i + 1)):
            merged = segments[i:i + merge_count]
            merged_text = " ".join(s['text'].strip() for s in merged)
            norm_merged = normalize_text(merged_text)
            sim = difflib.SequenceMatcher(None, norm_target, norm_merged).ratio()
            if sim > best_sim:
                best_sim = sim
                best_idx = i
                best_merge = merge_count
                best_text = merged_text

    return best_idx, best_merge, best_sim, best_text


def main():
    print("=" * 70)
    print("ROOT CAUSE DIAGNOSTIC: Pipeline vs Experiment Whisper Parameters")
    print("=" * 70)

    model = whisper.load_model("medium", device="cuda")

    # ====== RUN A: Pipeline parameters (word_timestamps=True, condition_on_previous_text=True) ======
    print("\n[A] PIPELINE PARAMETERS: word_timestamps=True, condition_on_previous_text=True (default)")
    result_a = model.transcribe(RAW_AUDIO, language="ko", verbose=False,
                                 fp16=True, word_timestamps=True)
    segs_a = result_a['segments']
    print(f"    Total segments: {len(segs_a)}")

    idx_a, merge_a, sim_a, text_a = find_best_match(segs_a, TARGET_TEXT)
    merged_a = segs_a[idx_a:idx_a + merge_a]
    start_a = merged_a[0]['start']
    end_a = merged_a[-1]['end']

    print(f"    Best match: seg[{idx_a}:{idx_a + merge_a}], sim={sim_a:.3f}")
    print(f"    Whisper text: {text_a}")
    print(f"    Segment timestamps: {start_a:.3f}s - {end_a:.3f}s ({(end_a - start_a)*1000:.0f}ms)")

    # Check word-level refinement (what pipeline does)
    refined = refine_boundaries_with_words(merged_a, TARGET_TEXT, normalize_text)
    if refined:
        ref_start, ref_end, words, si, ei = refined
        print(f"    WORD REFINEMENT ACTIVE: {ref_start:.3f}s - {ref_end:.3f}s ({(ref_end - ref_start)*1000:.0f}ms)")
        print(f"      Start word [{si}]: '{words[si].get('word', '')}' @ {words[si].get('start', '?')}s")
        print(f"      End word   [{ei}]: '{words[ei].get('word', '')}' @ {words[ei].get('end', '?')}s")
        pipeline_start = ref_start
        pipeline_end = ref_end
    else:
        print(f"    No word refinement (using raw segment timestamps)")
        pipeline_start = start_a
        pipeline_end = end_a

    # Gap to next segment
    next_idx_a = idx_a + merge_a
    if next_idx_a < len(segs_a):
        next_start_a = segs_a[next_idx_a]['start']
        gap_a = next_start_a - end_a
        print(f"    Next seg starts: {next_start_a:.3f}s (gap: {gap_a*1000:.0f}ms)")
        print(f"    Next seg text: {segs_a[next_idx_a]['text'][:80]}")
    else:
        gap_a = 9999
        print(f"    No next segment")

    # Show word timestamps for last segment
    last_merged = merged_a[-1]
    print(f"\n    --- Words in last merged segment (seg[{idx_a + merge_a - 1}]) ---")
    for w in last_merged.get('words', []):
        print(f"      [{w.get('start', '?'):.3f}-{w.get('end', '?'):.3f}] '{w.get('word', '')}'")

    # ====== RUN B: Experiment parameters (no word_timestamps, condition_on_previous_text=False) ======
    print("\n" + "=" * 70)
    print("[B] EXPERIMENT PARAMETERS: word_timestamps=False, condition_on_previous_text=False")
    result_b = model.transcribe(RAW_AUDIO, language="ko", verbose=False,
                                 condition_on_previous_text=False)
    segs_b = result_b['segments']
    print(f"    Total segments: {len(segs_b)}")

    idx_b, merge_b, sim_b, text_b = find_best_match(segs_b, TARGET_TEXT)
    merged_b = segs_b[idx_b:idx_b + merge_b]
    start_b = merged_b[0]['start']
    end_b = merged_b[-1]['end']

    print(f"    Best match: seg[{idx_b}:{idx_b + merge_b}], sim={sim_b:.3f}")
    print(f"    Whisper text: {text_b}")
    print(f"    Segment timestamps: {start_b:.3f}s - {end_b:.3f}s ({(end_b - start_b)*1000:.0f}ms)")

    # Gap to next segment
    next_idx_b = idx_b + merge_b
    if next_idx_b < len(segs_b):
        next_start_b = segs_b[next_idx_b]['start']
        gap_b = next_start_b - end_b
        print(f"    Next seg starts: {next_start_b:.3f}s (gap: {gap_b*1000:.0f}ms)")
        print(f"    Next seg text: {segs_b[next_idx_b]['text'][:80]}")
    else:
        gap_b = 9999
        print(f"    No next segment")

    # ====== COMPARISON ======
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(f"\n  Pipeline end:    {pipeline_end:.3f}s")
    print(f"  Experiment end:  {end_b:.3f}s")
    diff_ms = (pipeline_end - end_b) * 1000
    print(f"  DIFFERENCE:      {diff_ms:+.0f}ms")

    if diff_ms > 50:
        print(f"\n  >>> PIPELINE EXTENDS {diff_ms:.0f}ms FURTHER <<<")
        print(f"  >>> This is the ROOT CAUSE of the bleeding! <<<")
        print(f"  >>> The extra {diff_ms:.0f}ms includes speech from the next sentence. <<<")
    elif diff_ms < -50:
        print(f"\n  >>> Experiment extends {-diff_ms:.0f}ms further (unexpected)")
    else:
        print(f"\n  >>> Timestamps are similar. Check segment text differences.")

    # Also show the actual audio extraction ranges
    AUDIO_PAD_MS = 50
    MIN_GAP_FOR_PAD_MS = 30

    right_gap_a = gap_a * 1000
    right_pad_a = AUDIO_PAD_MS if right_gap_a >= MIN_GAP_FOR_PAD_MS else 0
    extract_end_a = pipeline_end * 1000 + right_pad_a

    right_gap_b = gap_b * 1000
    right_pad_b = AUDIO_PAD_MS if right_gap_b >= MIN_GAP_FOR_PAD_MS else 0
    extract_end_b = end_b * 1000 + right_pad_b

    print(f"\n  Pipeline extraction end (with padding): {extract_end_a:.0f}ms (pad={right_pad_a}ms)")
    print(f"  Experiment extraction end (with padding): {extract_end_b:.0f}ms (pad={right_pad_b}ms)")
    print(f"  Extraction difference: {extract_end_a - extract_end_b:+.0f}ms")

    # Save extractions for listening comparison
    raw_data, sr = sf.read(RAW_AUDIO, dtype='float64')
    OUT_DIR = os.path.dirname(__file__)

    # Pipeline extraction (what Stage 1 would produce)
    s1 = max(0, int(sr * (pipeline_start * 1000 - AUDIO_PAD_MS) / 1000))
    e1 = min(len(raw_data), int(sr * extract_end_a / 1000))
    pipe_extract = raw_data[s1:e1]
    pipe_path = os.path.join(OUT_DIR, "diag_pipeline_extract.wav")
    sf.write(pipe_path, pipe_extract, sr, subtype='PCM_24')
    print(f"\n  Saved pipeline-style extraction: {pipe_path}")
    print(f"    Duration: {len(pipe_extract)/sr*1000:.0f}ms")

    # Experiment extraction
    s2 = max(0, int(sr * (start_b * 1000 - AUDIO_PAD_MS) / 1000))
    e2 = min(len(raw_data), int(sr * extract_end_b / 1000))
    exp_extract = raw_data[s2:e2]
    exp_path = os.path.join(OUT_DIR, "diag_experiment_extract.wav")
    sf.write(exp_path, exp_extract, sr, subtype='PCM_24')
    print(f"  Saved experiment-style extraction: {exp_path}")
    print(f"    Duration: {len(exp_extract)/sr*1000:.0f}ms")


if __name__ == "__main__":
    main()
