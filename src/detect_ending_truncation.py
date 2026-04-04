"""
Detect Korean formal ending truncation in TTS dataset.

Phase 1: Text-based scan — identify segments with formal endings (fast, no GPU)
Phase 2: Audio verification — re-transcribe WITHOUT GT-prompting, compare endings
         (uses only the tail portion of audio for speed)

Key principle: Do NOT use GT-prompting for verification.
GT-prompting can cause Whisper to hallucinate missing text, masking truncation.
"""

import os
import re
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE / "datasets"
SCRIPT_FILE = DATASET_DIR / "script.txt"
WAV_DIR = DATASET_DIR / "wavs"
LOG_DIR = BASE / "logs"
REPORT_FILE = LOG_DIR / "ending_truncation_report.json"

# ── Korean formal endings to check ────────────────────────────────────
# These endings have natural micro-pauses that Whisper may misinterpret
# as segment boundaries. Ordered by risk (longer = more likely to have pause).
FORMAL_ENDINGS = [
    # -ㅂ니다 계열 (합쇼체)
    "습니다", "습니까", "십시오", "십시요",
    # -것입니다 계열
    "것입니다", "것이었습니다",
    # -였/었 + 습니다
    "었습니다", "였습니다", "겠습니다",
    "었습니까", "였습니까", "겠습니까",
    # -하십시오 계열
    "하십시오", "마십시오",
    # -ㅂ시다
    "읍시다", "합시다",
    # -세요 계열 (micro-pause less likely but still possible)
    "으세요", "하세요", "으셨어요",
]

# Minimum number of ending characters to compare
ENDING_COMPARE_LEN = 4


def normalize_text(text):
    """Strip punctuation, keep only Hangul + alphanumeric."""
    import unicodedata
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[^가-힣a-zA-Z0-9]', '', text)
    return text


def load_script():
    """Load datasets/script.txt → dict of {filename: text}."""
    entries = {}
    with open(SCRIPT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                fname, text = line.split('|', 1)
                entries[fname.strip()] = text.strip()
    return entries


def phase1_text_scan(entries):
    """Identify segments whose GT text ends with formal endings."""
    candidates = []
    for fname, gt_text in entries.items():
        norm_gt = normalize_text(gt_text)
        if len(norm_gt) < 6:
            continue

        for ending in FORMAL_ENDINGS:
            norm_ending = normalize_text(ending)
            if norm_gt.endswith(norm_ending):
                candidates.append({
                    'filename': fname,
                    'ground_truth': gt_text,
                    'ending_pattern': ending,
                    'gt_normalized_tail': norm_gt[-12:],
                })
                break  # first match is enough

    return candidates


def phase2_audio_verify(candidates, model_size="medium", max_items=None):
    """Re-transcribe candidates WITHOUT GT-prompting, check ending match.

    Only transcribes the audio — no GT prompt, no initial_prompt.
    Compares the last N characters of transcription vs ground truth.
    """
    import torch
    import whisper
    import soundfile as sf
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = (device == "cuda")

    print(f"\nLoading Whisper {model_size} on {device}...")
    model = whisper.load_model(model_size, device=device)
    print("Model loaded.\n")

    results = []
    items = candidates[:max_items] if max_items else candidates
    total = len(items)

    for i, cand in enumerate(items):
        fname = cand['filename']
        wav_path = WAV_DIR / fname

        if not wav_path.exists():
            cand['status'] = 'FILE_NOT_FOUND'
            results.append(cand)
            continue

        # Read audio
        samples, sr = sf.read(str(wav_path), dtype='float32')
        if sr != 16000:
            # Resample to 16kHz for Whisper
            duration_sec = len(samples) / sr
            target_len = int(duration_sec * 16000)
            indices = np.linspace(0, len(samples) - 1, target_len).astype(int)
            audio_16k = samples[indices].astype(np.float32)
        else:
            audio_16k = samples

        # Strip R6 envelope before transcription (same as evaluate_dataset.py)
        strip_lead = int(16000 * 350 / 1000)  # 350ms
        strip_tail = int(16000 * 700 / 1000)  # 700ms
        if len(audio_16k) > strip_lead + strip_tail + 1600:  # at least 100ms of speech
            audio_16k = audio_16k[strip_lead:-strip_tail] if strip_tail > 0 else audio_16k[strip_lead:]

        # Transcribe WITHOUT GT-prompting — critical for honest verification
        try:
            result = whisper.transcribe(
                model, audio_16k,
                language="ko",
                verbose=False,
                fp16=use_fp16,
                condition_on_previous_text=False,
                # NO initial_prompt — this is the whole point
            )
            whisper_text = result.get('text', '').strip()
        except Exception as e:
            cand['status'] = f'TRANSCRIBE_ERROR: {e}'
            results.append(cand)
            continue

        norm_whisper = normalize_text(whisper_text)
        norm_gt = normalize_text(cand['ground_truth'])

        # Compare endings
        gt_tail = norm_gt[-ENDING_COMPARE_LEN:]
        ending_pattern = normalize_text(cand['ending_pattern'])

        # Check if whisper transcription ends with the same formal ending
        whisper_has_ending = norm_whisper.endswith(ending_pattern)

        # Also check a looser match: last N chars
        whisper_tail = norm_whisper[-ENDING_COMPARE_LEN:] if len(norm_whisper) >= ENDING_COMPARE_LEN else norm_whisper
        tail_match = (whisper_tail == gt_tail)

        # Determine truncation
        if whisper_has_ending:
            truncated = False
            status = "OK"
        elif tail_match:
            truncated = False
            status = "OK_TAIL_MATCH"
        else:
            truncated = True
            # How much is missing?
            # Find the longest suffix of GT that matches the end of whisper
            missing_chars = 0
            for cut in range(1, min(len(ending_pattern) + 5, len(norm_gt))):
                gt_prefix = norm_gt[:-cut]
                if norm_whisper.endswith(gt_prefix[-6:]) if len(gt_prefix) >= 6 else False:
                    missing_chars = cut
                    break
            status = f"TRUNCATED (missing ~{missing_chars} chars)"

        cand['whisper_text'] = whisper_text
        cand['whisper_normalized_tail'] = norm_whisper[-12:] if norm_whisper else ""
        cand['whisper_has_formal_ending'] = whisper_has_ending
        cand['tail_match'] = tail_match
        cand['truncated'] = truncated
        cand['status'] = status

        results.append(cand)

        # Progress
        marker = "TRUNCATED" if truncated else "OK"
        print(f"  [{i+1}/{total}] {fname}: {marker}")
        if truncated:
            print(f"           GT ending: ...{norm_gt[-15:]}")
            print(f"           Whisper:   ...{norm_whisper[-15:]}")

    return results


def generate_report(candidates, verified_results, output_path):
    """Generate JSON report."""
    truncated = [r for r in verified_results if r.get('truncated', False)]

    report = {
        "timestamp": datetime.now().isoformat(),
        "phase1_candidates": len(candidates),
        "phase2_verified": len(verified_results),
        "truncation_detected": len(truncated),
        "truncation_rate": f"{len(truncated)/len(verified_results)*100:.1f}%" if verified_results else "N/A",
        "truncated_files": [
            {
                "filename": r['filename'],
                "ground_truth": r['ground_truth'],
                "whisper_text": r.get('whisper_text', ''),
                "ending_pattern": r['ending_pattern'],
                "gt_tail": r.get('gt_normalized_tail', ''),
                "whisper_tail": r.get('whisper_normalized_tail', ''),
            }
            for r in truncated
        ],
        "all_results": [
            {
                "filename": r['filename'],
                "ending_pattern": r['ending_pattern'],
                "status": r.get('status', 'PENDING'),
                "truncated": r.get('truncated', None),
            }
            for r in verified_results
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(description="Detect Korean ending truncation")
    parser.add_argument('--phase1-only', action='store_true',
                        help="Only run text-based scan (no GPU needed)")
    parser.add_argument('--max-items', type=int, default=None,
                        help="Limit Phase 2 verification to N items (for testing)")
    parser.add_argument('--model', default='medium',
                        help="Whisper model size for Phase 2 (default: medium)")
    args = parser.parse_args()

    print("=" * 60)
    print("Korean Formal Ending Truncation Detector")
    print("=" * 60)

    # Phase 1: Text scan
    print("\n--- Phase 1: Text-based scan ---")
    entries = load_script()
    print(f"Total entries in script.txt: {len(entries)}")

    candidates = phase1_text_scan(entries)
    print(f"Candidates with formal endings: {len(candidates)}")

    # Breakdown by ending pattern
    pattern_counts = {}
    for c in candidates:
        p = c['ending_pattern']
        pattern_counts[p] = pattern_counts.get(p, 0) + 1

    print("\nEnding pattern distribution:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {pattern:20s}  {count:>4d} segments")

    if args.phase1_only:
        print(f"\n Phase 1 complete. {len(candidates)} candidates identified.")
        print("Run without --phase1-only to verify with Whisper.")
        # Save partial report
        report = generate_report(candidates, [], REPORT_FILE)
        print(f"Report saved: {REPORT_FILE}")
        return

    # Phase 2: Audio verification
    print(f"\n--- Phase 2: Audio verification (Whisper {args.model}, NO GT-prompting) ---")
    if args.max_items:
        print(f"Limited to first {args.max_items} items.")

    verified = phase2_audio_verify(candidates, model_size=args.model, max_items=args.max_items)

    # Summary
    truncated = [r for r in verified if r.get('truncated', False)]
    ok = [r for r in verified if not r.get('truncated', False)]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Verified:   {len(verified)}")
    print(f"OK:         {len(ok)}")
    print(f"TRUNCATED:  {len(truncated)}")
    if verified:
        print(f"Truncation rate: {len(truncated)/len(verified)*100:.1f}%")

    if truncated:
        print(f"\n--- Truncated files ---")
        for r in truncated:
            print(f"  {r['filename']}: {r['ending_pattern']}")
            print(f"    GT:      ...{r.get('gt_normalized_tail', '')}")
            print(f"    Whisper: ...{r.get('whisper_normalized_tail', '')}")

    # Save report
    report = generate_report(candidates, verified, REPORT_FILE)
    print(f"\nFull report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
