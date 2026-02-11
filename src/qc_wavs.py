"""
WAV Quality Check Pipeline for TTS Dataset
============================================
Transcribes each WAV with Whisper medium, compares against expected script text,
and classifies each file as GOOD or PROBLEMATIC with specific issue tags.

Issue types detected:
  - START_BLEED:  Audio begins with words from the previous script line's tail
  - END_BLEED:    Audio ends with words from the next script line's beginning
  - TRUNCATED_END: Expected ending words are missing from the transcription
  - TRUNCATED_START: Expected beginning words are missing from the transcription
  - LOW_SIMILARITY: Overall similarity between transcription and expected text is low
  - EMPTY_TRANSCRIPTION: Whisper returned nothing for this audio

Output: CSV report at reports/qc_report.csv
"""

import os
import sys
import csv
import json
import time
import difflib
import re
import numpy as np
import soundfile as sf
import librosa
import whisper
import torch
import warnings

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = r"G:\Projects\AI_Research\TTSDataSetCleanser"
SCRIPT_FILE = os.path.join(BASE_DIR, "datasets", "PreviousVersion", "script.txt")
WAVS_DIR = os.path.join(BASE_DIR, "datasets", "PreviousVersion", "wavs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
REPORT_CSV = os.path.join(REPORT_DIR, "qc_report.csv")
CHECKPOINT_FILE = os.path.join(REPORT_DIR, "qc_checkpoint.json")

# ── Thresholds ─────────────────────────────────────────────────────────────────
SIMILARITY_GOOD_THRESHOLD = 0.70       # SequenceMatcher ratio >= this → no LOW_SIMILARITY flag
BLEED_CHAR_WINDOW = 8                   # Check last/first N chars for bleed detection
BLEED_MATCH_THRESHOLD = 0.75            # Similarity threshold for bleed segments
TRUNCATION_WORD_THRESHOLD = 0.20        # If >20% of expected words missing → truncation flag
START_TRUNC_WORD_CHECK = 3              # Check first N words for start truncation
END_TRUNC_WORD_CHECK = 3                # Check last N words for end truncation

# ── Whisper config ─────────────────────────────────────────────────────────────
WHISPER_MODEL = "medium"
WHISPER_LANGUAGE = "ko"


def normalize_korean(text: str) -> str:
    """Normalize text for comparison: strip punctuation, lowercase, collapse spaces."""
    text = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def word_list(text: str) -> list:
    """Split normalized text into word list."""
    return normalize_korean(text).split()


def similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio between two normalized strings."""
    na = normalize_korean(a)
    nb = normalize_korean(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def check_start_bleed(transcription: str, prev_expected: str) -> bool:
    """Check if the transcription starts with the tail of the previous line."""
    if not prev_expected:
        return False
    trans_norm = normalize_korean(transcription)
    prev_norm = normalize_korean(prev_expected)
    if len(trans_norm) < BLEED_CHAR_WINDOW or len(prev_norm) < BLEED_CHAR_WINDOW:
        return False
    trans_start = trans_norm[:BLEED_CHAR_WINDOW]
    prev_tail = prev_norm[-BLEED_CHAR_WINDOW:]
    return difflib.SequenceMatcher(None, trans_start, prev_tail).ratio() >= BLEED_MATCH_THRESHOLD


def check_end_bleed(transcription: str, next_expected: str) -> bool:
    """Check if the transcription ends with the head of the next line."""
    if not next_expected:
        return False
    trans_norm = normalize_korean(transcription)
    next_norm = normalize_korean(next_expected)
    if len(trans_norm) < BLEED_CHAR_WINDOW or len(next_norm) < BLEED_CHAR_WINDOW:
        return False
    trans_tail = trans_norm[-BLEED_CHAR_WINDOW:]
    next_head = next_norm[:BLEED_CHAR_WINDOW]
    return difflib.SequenceMatcher(None, trans_tail, next_head).ratio() >= BLEED_MATCH_THRESHOLD


def check_truncated_start(transcription: str, expected: str) -> bool:
    """Check if the beginning of the expected text is missing from the transcription."""
    exp_words = word_list(expected)
    trans_words = word_list(transcription)
    if len(exp_words) < START_TRUNC_WORD_CHECK or not trans_words:
        return False
    first_expected = exp_words[:START_TRUNC_WORD_CHECK]
    # Check if first expected words appear near the start of transcription
    trans_joined = ' '.join(trans_words[:START_TRUNC_WORD_CHECK + 2])
    expected_joined = ' '.join(first_expected)
    sim = difflib.SequenceMatcher(None, expected_joined, trans_joined).ratio()
    return sim < 0.5


def check_truncated_end(transcription: str, expected: str) -> bool:
    """Check if the ending of the expected text is missing from the transcription."""
    exp_words = word_list(expected)
    trans_words = word_list(transcription)
    if len(exp_words) < END_TRUNC_WORD_CHECK or not trans_words:
        return False
    last_expected = exp_words[-END_TRUNC_WORD_CHECK:]
    trans_tail = ' '.join(trans_words[-END_TRUNC_WORD_CHECK - 2:]) if len(trans_words) >= END_TRUNC_WORD_CHECK else ' '.join(trans_words)
    expected_joined = ' '.join(last_expected)
    sim = difflib.SequenceMatcher(None, expected_joined, trans_tail).ratio()
    return sim < 0.5


def load_script(path: str) -> list:
    """Load script.txt → list of (filename, expected_text) tuples, in order."""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|', 1)
            if len(parts) == 2:
                entries.append((parts[0].strip(), parts[1].strip()))
    return entries


def load_checkpoint(path: str) -> dict:
    """Load checkpoint of already-processed files."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_checkpoint(path: str, data: dict):
    """Save checkpoint."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 70)
    print("  TTS Dataset WAV Quality Check Pipeline")
    print("=" * 70)

    # Load script
    print(f"\n[1/4] Loading script from {SCRIPT_FILE} ...")
    entries = load_script(SCRIPT_FILE)
    print(f"       Loaded {len(entries)} entries.")

    # Build lookup: filename → (index, expected_text)
    # Also keep ordered list for prev/next bleed checking
    fname_to_idx = {}
    for i, (fname, text) in enumerate(entries):
        fname_to_idx[fname] = i

    # Load Whisper
    print(f"\n[2/4] Loading Whisper '{WHISPER_MODEL}' model on GPU ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(WHISPER_MODEL, device=device)
    print(f"       Model loaded on {device}.")

    # Load checkpoint
    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    already_done = set(checkpoint.keys())
    print(f"\n[3/4] Checkpoint: {len(already_done)} files already processed.")

    # Process
    print(f"\n[4/4] Processing {len(entries)} WAV files ...\n")
    total = len(entries)
    results = dict(checkpoint)  # start from checkpoint
    start_time = time.time()
    processed_this_run = 0

    for i, (fname, expected_text) in enumerate(entries):
        if fname in already_done:
            continue

        wav_path = os.path.join(WAVS_DIR, fname)
        if not os.path.exists(wav_path):
            results[fname] = {
                "transcription": "",
                "similarity": 0.0,
                "issues": ["FILE_MISSING"],
                "verdict": "PROBLEMATIC"
            }
            continue

        # Load audio with soundfile (no ffmpeg needed) and resample to 16kHz
        try:
            audio_data, sr = sf.read(wav_path, dtype='float32')
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)  # stereo → mono
            if sr != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            result = model.transcribe(audio_data, language=WHISPER_LANGUAGE)
            transcription = result["text"].strip()
        except Exception as e:
            results[fname] = {
                "transcription": f"ERROR: {e}",
                "similarity": 0.0,
                "issues": ["TRANSCRIPTION_ERROR"],
                "verdict": "PROBLEMATIC"
            }
            processed_this_run += 1
            continue

        # Analyze
        issues = []

        # Empty check
        if not transcription:
            issues.append("EMPTY_TRANSCRIPTION")
        else:
            # Overall similarity
            sim = similarity(transcription, expected_text)

            # Start bleed (from previous line)
            prev_text = entries[i - 1][1] if i > 0 else ""
            if check_start_bleed(transcription, prev_text):
                issues.append("START_BLEED")

            # End bleed (from next line)
            next_text = entries[i + 1][1] if i < total - 1 else ""
            if check_end_bleed(transcription, next_text):
                issues.append("END_BLEED")

            # Truncated start
            if check_truncated_start(transcription, expected_text):
                issues.append("TRUNCATED_START")

            # Truncated end
            if check_truncated_end(transcription, expected_text):
                issues.append("TRUNCATED_END")

            # Low overall similarity
            if sim < SIMILARITY_GOOD_THRESHOLD:
                issues.append("LOW_SIMILARITY")

        verdict = "GOOD" if len(issues) == 0 else "PROBLEMATIC"
        sim_val = similarity(transcription, expected_text) if transcription else 0.0

        results[fname] = {
            "transcription": transcription,
            "similarity": round(sim_val, 4),
            "issues": issues,
            "verdict": verdict
        }

        processed_this_run += 1
        done_total = len(already_done) + processed_this_run

        # Progress
        if processed_this_run % 25 == 0 or processed_this_run <= 5:
            elapsed = time.time() - start_time
            rate = processed_this_run / elapsed if elapsed > 0 else 0
            remaining = (total - done_total) / rate if rate > 0 else 0
            pct = done_total / total * 100
            print(f"  [{done_total:>5}/{total}] {pct:5.1f}% | {rate:.1f} files/s | "
                  f"ETA: {remaining/60:.1f}m | Last: {fname} → {verdict}")

        # Checkpoint every 100 files
        if processed_this_run % 100 == 0:
            save_checkpoint(CHECKPOINT_FILE, results)

    # Final checkpoint
    save_checkpoint(CHECKPOINT_FILE, results)

    elapsed_total = time.time() - start_time
    print(f"\n  Processing complete: {processed_this_run} new files in {elapsed_total/60:.1f} minutes.")

    # ── Write CSV report ───────────────────────────────────────────────────────
    print(f"\nWriting report to {REPORT_CSV} ...")

    good_count = 0
    problem_count = 0

    with open(REPORT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "script_id", "line_number",
            "expected_text", "transcription",
            "similarity", "issues", "verdict"
        ])

        for fname, expected_text in entries:
            r = results.get(fname, {})
            # Parse script_id and line_number from filename
            parts = fname.replace('.wav', '').split('_')
            script_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else ""
            line_num = parts[2] if len(parts) >= 3 else ""

            verdict = r.get("verdict", "UNKNOWN")
            if verdict == "GOOD":
                good_count += 1
            else:
                problem_count += 1

            writer.writerow([
                fname,
                script_id,
                line_num,
                expected_text,
                r.get("transcription", ""),
                r.get("similarity", 0.0),
                "|".join(r.get("issues", [])),
                verdict
            ])

    # ── Summary ────────────────────────────────────────────────────────────────
    # Count issue types
    issue_counts = {}
    for fname, r in results.items():
        for iss in r.get("issues", []):
            issue_counts[iss] = issue_counts.get(iss, 0) + 1

    print("\n" + "=" * 70)
    print("  QUALITY CHECK SUMMARY")
    print("=" * 70)
    print(f"  Total files checked:   {len(entries)}")
    print(f"  GOOD:                  {good_count} ({good_count/len(entries)*100:.1f}%)")
    print(f"  PROBLEMATIC:           {problem_count} ({problem_count/len(entries)*100:.1f}%)")
    print(f"\n  Issue breakdown:")
    for iss, cnt in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"    {iss:<25s} {cnt:>5d}")
    print(f"\n  Report saved to: {REPORT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
