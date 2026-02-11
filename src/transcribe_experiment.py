"""
Experiment: Pure Transcription of Pre-Segmented WAV Files
==========================================================
Transcribes 260 WAV files from rawdata/Experiment/wavs/ using Whisper medium.
Evaluates against answer key in rawdata/Experiment/Script_5_A0.txt.
Score = per-line (matched words / answer words), target >= 95%.
"""

import os
import sys
import re
import time
import difflib
import numpy as np
import soundfile as sf
import librosa
import whisper
import torch
import warnings

warnings.filterwarnings("ignore")

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = r"G:\Projects\AI_Research\TTSDataSetCleanser"
WAVS_DIR = os.path.join(BASE_DIR, "rawdata", "Experiment", "wavs")
ANSWER_KEY = os.path.join(BASE_DIR, "rawdata", "Experiment", "Script_5_A0.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "rawdata", "Experiment", "transcribed_script.txt")
REPORT_FILE = os.path.join(BASE_DIR, "reports", "transcription_experiment_report.txt")

WHISPER_MODEL = "medium"
WHISPER_LANGUAGE = "ko"


def normalize(text):
    text = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def word_similarity(transcribed, answer):
    """Score = matched words / answer words (word-level similarity)."""
    t_words = normalize(transcribed).split()
    a_words = normalize(answer).split()
    if not a_words:
        return 1.0 if not t_words else 0.0
    if not t_words:
        return 0.0
    matcher = difflib.SequenceMatcher(None, t_words, a_words)
    matching = sum(block.size for block in matcher.get_matching_blocks())
    return matching / len(a_words)


def main():
    print("=" * 70)
    print("  Experiment: Pure Transcription of Pre-Segmented WAVs")
    print("=" * 70)

    # Load answer key (1-indexed lines)
    with open(ANSWER_KEY, 'r', encoding='utf-8') as f:
        answer_lines = [line.strip() for line in f.readlines()]
    print(f"\n  Answer key loaded: {len(answer_lines)} lines")

    # List WAV files
    wav_files = sorted([f for f in os.listdir(WAVS_DIR) if f.endswith('.wav')])
    print(f"  WAV files to transcribe: {len(wav_files)}")

    # Load Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Loading Whisper '{WHISPER_MODEL}' on {device} ...")
    model = whisper.load_model(WHISPER_MODEL, device=device)
    print(f"  Model loaded.\n")

    # Transcribe
    results = []
    start_time = time.time()

    for i, fname in enumerate(wav_files):
        wav_path = os.path.join(WAVS_DIR, fname)

        # Load audio (no ffmpeg)
        audio, sr = sf.read(wav_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Transcribe
        result = model.transcribe(audio, language=WHISPER_LANGUAGE)
        transcription = result["text"].strip()

        # Map filename to answer key line
        # Script_5_542.wav → line index 541 (0-based)
        line_num = int(re.search(r'(\d+)\.wav$', fname).group(1))
        answer_text = answer_lines[line_num - 1] if line_num <= len(answer_lines) else ""

        score = word_similarity(transcription, answer_text)

        results.append({
            "filename": fname,
            "line_num": line_num,
            "transcription": transcription,
            "answer": answer_text,
            "score": score
        })

        if (i + 1) % 25 == 0 or (i + 1) <= 3:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(wav_files) - i - 1) / rate
            print(f"  [{i+1:>4}/{len(wav_files)}] {rate:.1f} files/s | "
                  f"ETA: {eta/60:.1f}m | {fname} → {score:.3f}")

    elapsed_total = time.time() - start_time

    # Write output (script.txt format)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(f"{r['filename']}|{r['transcription']}\n")
    print(f"\n  Transcription saved to: {OUTPUT_FILE}")

    # Evaluate
    scores = [r['score'] for r in results]
    total_score = sum(scores)
    avg_score = total_score / len(scores) * 100

    perfect = sum(1 for s in scores if s >= 1.0)
    high = sum(1 for s in scores if 0.95 <= s < 1.0)
    medium = sum(1 for s in scores if 0.80 <= s < 0.95)
    low = sum(1 for s in scores if s < 0.80)

    passed = avg_score >= 95.0

    # Worst performers
    worst = sorted(results, key=lambda r: r['score'])[:15]

    # Per-range breakdown
    ranges = {}
    for r in results:
        bucket = (r['line_num'] // 50) * 50
        key = f"{bucket}-{bucket+49}"
        if key not in ranges:
            ranges[key] = []
        ranges[key].append(r['score'])

    # Build report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  TRANSCRIPTION EXPERIMENT REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"  Date:              2026-02-08")
    report_lines.append(f"  Whisper model:     {WHISPER_MODEL}")
    report_lines.append(f"  Device:            {device}")
    report_lines.append(f"  Processing time:   {elapsed_total/60:.1f} minutes ({elapsed_total:.0f}s)")
    report_lines.append(f"  Files transcribed: {len(results)}")
    report_lines.append(f"")
    report_lines.append(f"  OVERALL SCORE:     {avg_score:.2f}%  {'PASS' if passed else 'FAIL'} (target: >= 95%)")
    report_lines.append(f"")
    report_lines.append(f"  Score distribution:")
    report_lines.append(f"    100% (perfect):  {perfect:>4d} ({perfect/len(results)*100:.1f}%)")
    report_lines.append(f"    95-99%:          {high:>4d} ({high/len(results)*100:.1f}%)")
    report_lines.append(f"    80-94%:          {medium:>4d} ({medium/len(results)*100:.1f}%)")
    report_lines.append(f"    < 80%:           {low:>4d} ({low/len(results)*100:.1f}%)")
    report_lines.append(f"")
    report_lines.append(f"  Statistics:")
    report_lines.append(f"    Mean:   {np.mean(scores)*100:.2f}%")
    report_lines.append(f"    Median: {np.median(scores)*100:.2f}%")
    report_lines.append(f"    Min:    {np.min(scores)*100:.2f}%")
    report_lines.append(f"    Max:    {np.max(scores)*100:.2f}%")
    report_lines.append(f"    Stdev:  {np.std(scores)*100:.2f}%")
    report_lines.append(f"")

    report_lines.append(f"  --- Line Range Breakdown ---")
    for key in sorted(ranges.keys(), key=lambda x: int(x.split('-')[0])):
        r_scores = ranges[key]
        avg = np.mean(r_scores) * 100
        report_lines.append(f"    Lines {key:>8s}: {len(r_scores):>3d} files, avg {avg:.1f}%")
    report_lines.append(f"")

    report_lines.append(f"  --- 15 Worst Scoring Lines ---")
    report_lines.append(f"  {'File':<22s} {'Score':>7s}  Answer vs Transcription")
    report_lines.append(f"  {'-'*22} {'-'*7}  {'-'*50}")
    for r in worst:
        report_lines.append(f"  {r['filename']:<22s} {r['score']*100:>6.1f}%")
        report_lines.append(f"    ANS: {r['answer'][:80]}")
        report_lines.append(f"    GOT: {r['transcription'][:80]}")
    report_lines.append(f"")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)

    # Print and save
    print(report_text)

    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n  Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
