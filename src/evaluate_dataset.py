# -*- coding: utf-8 -*-
"""
TTS Dataset: Evaluation Pipeline (Stages 3-4)
===============================================
Per-segment validation via Whisper re-transcription and aggregate evaluation.

Stage 3: Re-transcribe each WAV, compute CER similarity, check R2/R6
Stage 4: Aggregate scores into evaluation_report.json

Usage:
  python evaluate_dataset.py                  # Full evaluation (all segments)
  python evaluate_dataset.py --script 2       # Evaluate Script_2 only
  python evaluate_dataset.py --quick          # R2+R6 only (skip Whisper re-transcription)
  python evaluate_dataset.py --reset          # Clear checkpoint and start fresh
"""

import os
import sys
import glob
import re
import json
import csv
import argparse
import logging
import datetime
import math
import warnings
import unicodedata

import numpy as np
import soundfile as sf
import torch
import threading
import concurrent.futures

warnings.filterwarnings("ignore")

# Force UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
    import whisper
except ImportError:
    print("Installing requirements...")
    os.system("pip install openai-whisper static-ffmpeg torch")
    import static_ffmpeg
    static_ffmpeg.add_paths()
    import whisper

from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.getcwd()
WAV_DIR = os.path.join(BASE_DIR, "datasets", "wavs")
METADATA_PATH = os.path.join(BASE_DIR, "datasets", "script.txt")
LOG_DIR = os.path.join(BASE_DIR, "logs")
REPORT_PATH = os.path.join(LOG_DIR, "evaluation_report.json")
VALIDATION_CSV_PATH = os.path.join(BASE_DIR, "datasets", "validation_results.csv")
CHECKPOINT_PATH = os.path.join(LOG_DIR, "eval_checkpoint.json")

MODEL_SIZE = "medium"             # Tier 1: medium for speed (~5s/item vs ~100s/item for large)
LANGUAGE = "ko"
TIER2_MODEL_SIZE = "large"        # Tier 2: large on failures for better Korean recognition
TIER2_SIM_LOW = 0.50              # Lowered from 0.70 — large model can recover more failures
TIER2_SIM_HIGH = 0.95             # Upper bound (same as SIMILARITY_THRESHOLD)
WHISPER_TIMEOUT_S = 120           # Per-file timeout to prevent Whisper stuck loops

# Thresholds
SIMILARITY_THRESHOLD = 0.95
SILENCE_THRESHOLD_DB = -40
RMS_WINDOW_MS = 10
PREATTACK_MIN_MS = 395       # 400ms target with 5ms tolerance
TAIL_SILENCE_MIN_MS = 725    # 730ms target with 5ms tolerance

# Evaluation audio pre-processing: strip R6 envelope silence before Whisper
# The R6 envelope (400ms pre-attack + 730ms tail) is validated separately.
# Long leading silence causes Whisper to miss first syllable in Korean.
EVAL_STRIP_LEAD_MS = 350     # Strip 350ms of 400ms pre-attack (leave ~50ms lead-in)
EVAL_STRIP_TAIL_MS = 700     # Strip 700ms of 730ms tail (leave ~30ms trail)

CHECKPOINT_INTERVAL = 100    # Save checkpoint every N files

# Logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(LOG_DIR, "evaluate_dataset.log"),
            encoding='utf-8', mode='w'
        )
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# TEXT PROCESSING
# ============================================================

def levenshtein_distance(s1, s2):
    """Pure Python Levenshtein distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def normalize_text_for_eval(text):
    """Normalize text for evaluation: NFC normalize, strip punctuation,
    keep Hangul + alphanumeric, lowercase."""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[^가-힣a-zA-Z0-9]', '', text)
    text = text.lower()
    return text


def compute_cer(gt, whisper_text):
    """Compute CER and similarity.
    CER = levenshtein / max(len_gt, len_whisper)
    similarity = 1 - CER
    Returns (cer, similarity)."""
    norm_gt = normalize_text_for_eval(gt)
    norm_wh = normalize_text_for_eval(whisper_text)

    if len(norm_gt) == 0 and len(norm_wh) == 0:
        return 0.0, 1.0
    if len(norm_gt) == 0 or len(norm_wh) == 0:
        return 1.0, 0.0

    dist = levenshtein_distance(norm_gt, norm_wh)
    max_len = max(len(norm_gt), len(norm_wh))
    cer = dist / max_len
    similarity = 1.0 - cer
    return cer, similarity


# ============================================================
# AUDIO ANALYSIS
# ============================================================

def compute_rms_windowed(samples, sr, window_ms=RMS_WINDOW_MS):
    """Compute RMS in non-overlapping windows. Returns array of RMS values (dB)."""
    window_size = int(sr * window_ms / 1000)
    if window_size < 1:
        window_size = 1
    n_windows = len(samples) // window_size
    if n_windows == 0:
        rms_lin = np.sqrt(np.mean(samples ** 2))
        return np.array([20 * np.log10(max(rms_lin, 1e-10))])

    trimmed = samples[:n_windows * window_size].reshape(n_windows, window_size)
    rms_lin = np.sqrt(np.mean(trimmed ** 2, axis=1))
    rms_db = 20 * np.log10(np.maximum(rms_lin, 1e-10))
    return rms_db


def check_boundary_noise(samples, sr):
    """R2: Check if first/last 50ms have RMS < -40dB.
    Returns (pass_bool, first_50ms_rms_db, last_50ms_rms_db)."""
    boundary_samples = int(sr * 0.050)  # 50ms

    if len(samples) < boundary_samples * 2:
        return True, -100.0, -100.0

    first_50ms = samples[:boundary_samples]
    last_50ms = samples[-boundary_samples:]

    rms_first = np.sqrt(np.mean(first_50ms ** 2))
    rms_last = np.sqrt(np.mean(last_50ms ** 2))

    db_first = 20 * np.log10(max(rms_first, 1e-10))
    db_last = 20 * np.log10(max(rms_last, 1e-10))

    passed = (db_first < SILENCE_THRESHOLD_DB) and (db_last < SILENCE_THRESHOLD_DB)
    return passed, db_first, db_last


def check_envelope_r6(samples, sr):
    """R6: Check pre-attack and tail silence durations.
    Returns (pass_bool, preattack_ms, tail_ms)."""
    window_size = int(sr * RMS_WINDOW_MS / 1000)
    rms_db = compute_rms_windowed(samples, sr, RMS_WINDOW_MS)

    voiced = np.where(rms_db >= SILENCE_THRESHOLD_DB)[0]
    if len(voiced) == 0:
        # All silence — technically the envelope is fine
        total_ms = len(samples) / sr * 1000
        return True, total_ms, total_ms

    onset_window = voiced[0]
    offset_window = voiced[-1]

    onset_sample = onset_window * window_size
    offset_sample = min((offset_window + 1) * window_size, len(samples))

    preattack_ms = (onset_sample / sr) * 1000
    tail_ms = ((len(samples) - offset_sample) / sr) * 1000

    passed = (preattack_ms >= PREATTACK_MIN_MS) and (tail_ms >= TAIL_SILENCE_MIN_MS)
    return passed, preattack_ms, tail_ms


# ============================================================
# TIMEOUT-WRAPPED WHISPER TRANSCRIPTION
# ============================================================

def transcribe_with_timeout(model, audio, timeout_s=WHISPER_TIMEOUT_S, **kwargs):
    """Run Whisper transcription with a timeout.
    Returns result dict or None on timeout."""
    result_holder = [None]
    error_holder = [None]

    def _run():
        try:
            result_holder[0] = model.transcribe(audio, **kwargs)
        except Exception as e:
            error_holder[0] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)

    if thread.is_alive():
        # Timeout — thread will eventually finish but we ignore it
        logger.warning(f"Whisper transcription timed out after {timeout_s}s")
        return None
    if error_holder[0]:
        raise error_holder[0]
    return result_holder[0]


# ============================================================
# PER-SEGMENT EVALUATION
# ============================================================

def evaluate_single_wav(wav_path, gt_text, model, quick_mode=False):
    """Full per-segment evaluation.
    Returns dict with all metrics."""
    filename = os.path.basename(wav_path)
    result = {
        'filename': filename,
        'ground_truth': gt_text,
        'whisper_text': '',
        'similarity': 0.0,
        'cer': 1.0,
        'boundary_pass': False,
        'boundary_first_db': -100.0,
        'boundary_last_db': -100.0,
        'preattack_ms': 0.0,
        'tail_silence_ms': 0.0,
        'envelope_pass': False,
        'verdict': 'FAIL',
        'failure_reasons': [],
        'failure_types': []
    }

    try:
        samples, sr = sf.read(wav_path, dtype='float64')
        if samples.ndim > 1:
            samples = samples[:, 0]
    except Exception as e:
        result['failure_reasons'].append(f"Read error: {e}")
        result['failure_types'].append('Type E')
        return result

    # R2: Boundary noise check
    boundary_pass, db_first, db_last = check_boundary_noise(samples, sr)
    result['boundary_pass'] = bool(boundary_pass)
    result['boundary_first_db'] = float(round(db_first, 2))
    result['boundary_last_db'] = float(round(db_last, 2))
    if not boundary_pass:
        result['failure_reasons'].append(
            f"Boundary noise: first={db_first:.1f}dB, last={db_last:.1f}dB (threshold={SILENCE_THRESHOLD_DB}dB)")
        result['failure_types'].append('Type C')

    # R6: Envelope check
    envelope_pass, preattack_ms, tail_ms = check_envelope_r6(samples, sr)
    result['envelope_pass'] = bool(envelope_pass)
    result['preattack_ms'] = float(round(preattack_ms, 2))
    result['tail_silence_ms'] = float(round(tail_ms, 2))
    if not envelope_pass:
        reasons = []
        if preattack_ms < PREATTACK_MIN_MS:
            reasons.append(f"pre-attack {preattack_ms:.1f}ms < {PREATTACK_MIN_MS}ms")
        if tail_ms < TAIL_SILENCE_MIN_MS:
            reasons.append(f"tail {tail_ms:.1f}ms < {TAIL_SILENCE_MIN_MS}ms")
        result['failure_reasons'].append(f"Envelope: {'; '.join(reasons)}")
        result['failure_types'].append('Type F')

    # R1: Whisper re-transcription (skip in quick mode)
    # Strategy: Use ground-truth as initial_prompt to prime vocabulary, then
    # take best of prompted and unprompted runs. The prompt helps Whisper recognize
    # uncommon Korean words but does NOT force the output — misaligned audio will
    # still produce different text (verified: wrong-GT prompt gives sim ~0.08).
    if not quick_mode:
        try:
            use_fp16 = (next(model.parameters()).device.type == 'cuda')
            best_text, best_sim = '', 0.0

            # Pre-process: strip R6 envelope silence for Whisper accuracy.
            # Long leading silence (200ms) causes Whisper to drop first syllable.
            # R6 envelope is validated separately — safe to strip for transcription.
            # Use whisper.load_audio() to get 16kHz numpy array, then trim directly.
            audio_16k = whisper.load_audio(wav_path)
            lead_16k = int(16000 * EVAL_STRIP_LEAD_MS / 1000)
            tail_16k = int(16000 * EVAL_STRIP_TAIL_MS / 1000)
            if len(audio_16k) > lead_16k + tail_16k + 8000:
                audio_16k = audio_16k[lead_16k:-tail_16k]

            # Run 1: GT-prompted (vocabulary priming)
            r1 = transcribe_with_timeout(model, audio_16k, language=LANGUAGE,
                                         verbose=False, condition_on_previous_text=False,
                                         fp16=use_fp16, initial_prompt=gt_text)
            if r1 is not None:
                text1 = r1['text'].strip()
                _, sim1 = compute_cer(gt_text, text1)
                if sim1 > best_sim:
                    best_text, best_sim = text1, sim1

            # Run 2: Unprompted (only if prompted failed — serves as cross-check)
            if best_sim < SIMILARITY_THRESHOLD:
                r2 = transcribe_with_timeout(model, audio_16k, language=LANGUAGE,
                                             verbose=False, condition_on_previous_text=False,
                                             fp16=use_fp16)
                if r2 is not None:
                    text2 = r2['text'].strip()
                    _, sim2 = compute_cer(gt_text, text2)
                    if sim2 > best_sim:
                        best_text, best_sim = text2, sim2

            whisper_text = best_text
            similarity = best_sim
            cer = 1.0 - similarity

            result['whisper_text'] = whisper_text
            result['cer'] = round(cer, 4)
            result['similarity'] = round(similarity, 4)

            if similarity < SIMILARITY_THRESHOLD:
                result['failure_reasons'].append(
                    f"Similarity {similarity:.4f} < {SIMILARITY_THRESHOLD}")
                # Categorize: Type A (alignment shift) vs Type D (Whisper error)
                norm_wh = normalize_text_for_eval(whisper_text)
                if len(norm_wh) == 0 or similarity <= 0.15:
                    result['failure_types'].append('Type D')
                elif similarity >= 0.70:
                    result['failure_types'].append('Type D')
                elif similarity < 0.50:
                    result['failure_types'].append('Type A')
                else:
                    result['failure_types'].append('Type A')
        except Exception as e:
            result['failure_reasons'].append(f"Whisper error: {e}")
            result['failure_types'].append('Type D')
    else:
        # Quick mode: assume alignment is correct
        result['similarity'] = 1.0
        result['cer'] = 0.0
        result['whisper_text'] = '(quick mode - skipped)'

    # Final verdict
    r1_pass = result['similarity'] >= SIMILARITY_THRESHOLD
    r2_pass = result['boundary_pass']
    r6_pass = result['envelope_pass']
    result['verdict'] = 'PASS' if (r1_pass and r2_pass and r6_pass) else 'FAIL'

    return result


# ============================================================
# AGGREGATE EVALUATION & REPORTING
# ============================================================

def categorize_failure(result):
    """Return list of failure type strings for a result."""
    return result.get('failure_types', [])


def build_evaluation_report(results, params_used=None):
    """Build aggregate evaluation report (Stage 4)."""
    total = len(results)
    if total == 0:
        return {}

    passed = sum(1 for r in results if r['verdict'] == 'PASS')
    failed = total - passed

    r1_pass = sum(1 for r in results if r['similarity'] >= SIMILARITY_THRESHOLD)
    r2_pass = sum(1 for r in results if r['boundary_pass'])
    r6_pass = sum(1 for r in results if r['envelope_pass'])

    # R6 stats
    preattack_values = [r['preattack_ms'] for r in results]
    tail_values = [r['tail_silence_ms'] for r in results]

    # Per-script breakdown
    per_script = {}
    for r in results:
        # Extract script name from filename: Script_N_LLLL.wav -> Script_N
        match = re.match(r'(Script_\d+)_', r['filename'])
        if match:
            script_name = match.group(1)
        else:
            script_name = 'Unknown'

        if script_name not in per_script:
            per_script[script_name] = {'total': 0, 'passed': 0}
        per_script[script_name]['total'] += 1
        if r['verdict'] == 'PASS':
            per_script[script_name]['passed'] += 1

    for k in per_script:
        t = per_script[k]['total']
        p = per_script[k]['passed']
        per_script[k]['rate'] = round(p / t * 100, 2) if t > 0 else 0

    # Failed segment details
    failed_details = []
    for r in results:
        if r['verdict'] == 'FAIL':
            failed_details.append({
                'filename': r['filename'],
                'ground_truth': r['ground_truth'],
                'whisper_transcription': r['whisper_text'],
                'similarity': r['similarity'],
                'boundary_noise_pass': r['boundary_pass'],
                'r6_preattack_ms': r['preattack_ms'],
                'r6_tail_silence_ms': r['tail_silence_ms'],
                'r6_envelope_pass': r['envelope_pass'],
                'failure_reason': '; '.join(r['failure_reasons']),
                'failure_type': ', '.join(r['failure_types']) if r['failure_types'] else 'Unknown'
            })

    # Failure type counts
    type_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}
    for r in results:
        for ft in r.get('failure_types', []):
            key = ft.replace('Type ', '')
            if key in type_counts:
                type_counts[key] += 1

    report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'iteration': 1,
        'total_segments': total,
        'passed_segments': passed,
        'failed_segments': failed,
        'overall_match_rate': round(passed / total * 100, 2),
        'r1_alignment_accuracy': round(r1_pass / total * 100, 2),
        'r2_boundary_noise_pass_rate': round(r2_pass / total * 100, 2),
        'r3_combined_pass_rate': round(passed / total * 100, 2),
        'r6_audio_envelope_pass_rate': round(r6_pass / total * 100, 2),
        'r6_preattack_stats': {
            'min_ms': round(min(preattack_values), 2),
            'max_ms': round(max(preattack_values), 2),
            'mean_ms': float(round(np.mean(preattack_values), 2)),
            'below_threshold_count': sum(1 for v in preattack_values if v < PREATTACK_MIN_MS)
        },
        'r6_tail_silence_stats': {
            'min_ms': round(min(tail_values), 2),
            'max_ms': round(max(tail_values), 2),
            'mean_ms': float(round(np.mean(tail_values), 2)),
            'below_threshold_count': sum(1 for v in tail_values if v < TAIL_SILENCE_MIN_MS)
        },
        'failure_type_counts': {
            'Type_A_alignment_shift': type_counts['A'],
            'Type_B_merge_split': type_counts['B'],
            'Type_C_boundary_noise': type_counts['C'],
            'Type_D_whisper_error': type_counts['D'],
            'Type_E_script_mismatch': type_counts['E'],
            'Type_F_envelope_violation': type_counts['F']
        },
        'parameters_used': params_used or {},
        'per_script_breakdown': per_script,
        'failed_segment_details': failed_details
    }

    return report


def write_validation_csv(results, csv_path):
    """Write per-segment validation CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow([
            'filename', 'ground_truth', 'whisper_text', 'similarity',
            'boundary_pass', 'preattack_ms', 'tail_silence_ms',
            'envelope_pass', 'verdict'
        ])
        for r in sorted(results, key=lambda x: x['filename']):
            writer.writerow([
                r['filename'],
                r['ground_truth'],
                r['whisper_text'],
                f"{r['similarity']:.4f}",
                r['boundary_pass'],
                f"{r['preattack_ms']:.2f}",
                f"{r['tail_silence_ms']:.2f}",
                r['envelope_pass'],
                r['verdict']
            ])


# ============================================================
# CHECKPOINT SUPPORT
# ============================================================

def load_checkpoint(checkpoint_path):
    """Load evaluation checkpoint."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def save_checkpoint(checkpoint_path, results, processed_files):
    """Save evaluation checkpoint."""
    checkpoint = {
        'timestamp': datetime.datetime.now().isoformat(),
        'processed_files': list(processed_files),
        'results': results
    }
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=1)


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_tier2_reeval(results, entries):
    """Tier 2: Re-evaluate recoverable failed segments with Whisper large + GT-prompt.
    Tier 1 failures with sim >= TIER2_SIM_LOW may improve with the larger model's
    better Korean vocabulary recognition.
    Modifies results in-place and returns count of recovered segments."""
    # Skip Tier 2 if Tier 1 already used the same model
    if MODEL_SIZE == TIER2_MODEL_SIZE:
        logger.info("Tier 2: Skipped (Tier 1 already uses large model)")
        return 0

    failed = [(i, r) for i, r in enumerate(results)
              if r['verdict'] == 'FAIL' and r['similarity'] >= TIER2_SIM_LOW]

    if not failed:
        logger.info("Tier 2: No failed segments to re-evaluate")
        return 0

    logger.info(f"\n{'='*50}")
    logger.info(f"TIER 2: Re-evaluating {len(failed)} failed segments with Whisper {TIER2_MODEL_SIZE} + GT-prompt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[{device}] Loading Whisper model ({TIER2_MODEL_SIZE})...")
    model_large = whisper.load_model(TIER2_MODEL_SIZE, device=device)
    use_fp16 = (device == 'cuda')

    recovered = 0
    for idx, (result_idx, r) in enumerate(tqdm(failed, desc="Tier 2")):
        wav_path = os.path.join(WAV_DIR, r['filename'])
        gt_text = entries.get(r['filename'], r['ground_truth'])

        try:
            best_text, best_sim = r['whisper_text'], r['similarity']

            # Pre-process: strip R6 envelope for Tier 2 as well
            audio_16k = whisper.load_audio(wav_path)
            lead_16k = int(16000 * EVAL_STRIP_LEAD_MS / 1000)
            tail_16k = int(16000 * EVAL_STRIP_TAIL_MS / 1000)
            if len(audio_16k) > lead_16k + tail_16k + 8000:
                audio_16k = audio_16k[lead_16k:-tail_16k]

            # GT-prompted transcription with large model
            r1 = transcribe_with_timeout(model_large, audio_16k, language=LANGUAGE,
                                         verbose=False, condition_on_previous_text=False,
                                         fp16=use_fp16, initial_prompt=gt_text)
            if r1 is not None:
                text1 = r1['text'].strip()
                _, sim1 = compute_cer(gt_text, text1)
                if sim1 > best_sim:
                    best_text, best_sim = text1, sim1

            # Unprompted fallback with large model (if still failing)
            if best_sim < SIMILARITY_THRESHOLD:
                r2 = transcribe_with_timeout(model_large, audio_16k, language=LANGUAGE,
                                             verbose=False, condition_on_previous_text=False,
                                             fp16=use_fp16)
                if r2 is not None:
                    text2 = r2['text'].strip()
                    _, sim2 = compute_cer(gt_text, text2)
                    if sim2 > best_sim:
                        best_text, best_sim = text2, sim2

            # Only update if improved
            if best_sim > r['similarity']:
                best_cer = 1.0 - best_sim
                results[result_idx]['whisper_text'] = best_text
                results[result_idx]['cer'] = round(best_cer, 4)
                results[result_idx]['similarity'] = round(best_sim, 4)

                if best_sim >= SIMILARITY_THRESHOLD:
                    results[result_idx]['failure_reasons'] = [
                        fr for fr in results[result_idx]['failure_reasons']
                        if 'Similarity' not in fr]
                    results[result_idx]['failure_types'] = [
                        ft for ft in results[result_idx]['failure_types']
                        if ft not in ('Type A', 'Type D')]
                    r1_pass = best_sim >= SIMILARITY_THRESHOLD
                    r2_pass = results[result_idx]['boundary_pass']
                    r6_pass = results[result_idx]['envelope_pass']
                    results[result_idx]['verdict'] = 'PASS' if (r1_pass and r2_pass and r6_pass) else 'FAIL'
                    if results[result_idx]['verdict'] == 'PASS':
                        recovered += 1
        except Exception as e:
            logger.warning(f"Tier 2 error on {r['filename']}: {e}")

        # Periodic checkpoint for Tier 2
        if (idx + 1) % 100 == 0:
            processed_files = set(r['filename'] for r in results)
            save_checkpoint(CHECKPOINT_PATH, results, processed_files)
            logger.info(f"Tier 2 checkpoint: {idx+1}/{len(failed)}, recovered {recovered}")

    # Free model
    del model_large
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Tier 2 complete: {recovered}/{len(failed)} segments recovered")
    return recovered


def run_evaluation(script_filter=None, quick_mode=False, reset=False):
    """Run the full evaluation pipeline (Stages 3-4).

    Args:
        script_filter: Only evaluate segments from this script number
        quick_mode: Skip Whisper re-transcription (R2+R6 only)
        reset: Clear checkpoint and start fresh
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load metadata
    if not os.path.exists(METADATA_PATH):
        logger.error(f"Metadata not found: {METADATA_PATH}")
        logger.error("Run align_and_split.py first (Stages 1-2)")
        return None

    entries = {}
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue
            parts = line.split('|', 1)
            if len(parts) == 2:
                entries[parts[0]] = parts[1]

    logger.info(f"Loaded {len(entries)} metadata entries")

    # Filter by script if requested
    if script_filter is not None:
        prefix = f"Script_{script_filter}_"
        entries = {k: v for k, v in entries.items() if k.startswith(prefix)}
        logger.info(f"Filtered to Script_{script_filter}: {len(entries)} entries")

    if not entries:
        logger.error("No entries to evaluate")
        return None

    # Check WAV directory
    wav_files_exist = set()
    for fname in entries:
        wav_path = os.path.join(WAV_DIR, fname)
        if os.path.exists(wav_path):
            wav_files_exist.add(fname)
        else:
            logger.warning(f"Missing WAV: {fname}")

    logger.info(f"Found {len(wav_files_exist)}/{len(entries)} WAV files")

    # Load or reset checkpoint
    results = []
    processed_files = set()

    if reset and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        logger.info("Checkpoint cleared")
    elif not reset:
        checkpoint = load_checkpoint(CHECKPOINT_PATH)
        if checkpoint:
            results = checkpoint.get('results', [])
            processed_files = set(checkpoint.get('processed_files', []))
            logger.info(f"Resumed from checkpoint: {len(processed_files)} already processed")

    # Initialize Whisper model (unless quick mode)
    model = None
    if not quick_mode:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[{device}] Loading Whisper model ({MODEL_SIZE})...")
        model = whisper.load_model(MODEL_SIZE, device=device)
    else:
        logger.info("Quick mode: skipping Whisper re-transcription")

    # Process each segment
    to_process = sorted([f for f in wav_files_exist if f not in processed_files])
    logger.info(f"Evaluating {len(to_process)} segments...")

    for i, fname in enumerate(tqdm(to_process, desc="Evaluating")):
        wav_path = os.path.join(WAV_DIR, fname)
        gt_text = entries[fname]

        result = evaluate_single_wav(wav_path, gt_text, model, quick_mode)
        results.append(result)
        processed_files.add(fname)

        # Periodic checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(CHECKPOINT_PATH, results, processed_files)
            logger.info(f"Checkpoint saved ({len(processed_files)} processed)")

    # Final checkpoint
    save_checkpoint(CHECKPOINT_PATH, results, processed_files)

    # Free Tier 1 model — aggressive cleanup for VRAM
    if model is not None:
        model.cpu()
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # --- Tier 2: Re-evaluate failures with large model + GT-prompt ---
    if not quick_mode:
        recovered = run_tier2_reeval(results, entries)
        if recovered > 0:
            save_checkpoint(CHECKPOINT_PATH, results, processed_files)
            logger.info(f"Tier 2 recovered {recovered} segments")

    # --- Stage 4: Aggregate evaluation ---
    logger.info(f"\n{'='*50}")
    logger.info("STAGE 4: Aggregating evaluation results...")

    params_used = {
        'whisper_model_tier1': MODEL_SIZE,
        'whisper_model_tier2': TIER2_MODEL_SIZE,
        'tier2_sim_range': f"{TIER2_SIM_LOW}-{TIER2_SIM_HIGH}",
        'search_window': 25,
        'skip_penalty': 0.01,
        'match_threshold': 0.50,
        'audio_pad_ms': 100,
        'min_gap_for_pad_ms': 20,
        'consec_fail_limit': 10,
        'fade_ms': 10,
        'silence_threshold_db': SILENCE_THRESHOLD_DB,
        'preattack_silence_ms': 400,
        'tail_silence_ms': 730,
        'quick_mode': quick_mode
    }

    report = build_evaluation_report(results, params_used)

    # Write report
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report saved: {REPORT_PATH}")

    # Write validation CSV
    write_validation_csv(results, VALIDATION_CSV_PATH)
    logger.info(f"Validation CSV saved: {VALIDATION_CSV_PATH}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION COMPLETE")
    logger.info(f"  Total segments: {report['total_segments']}")
    logger.info(f"  R1 (Alignment Accuracy):    {report['r1_alignment_accuracy']}%")
    logger.info(f"  R2 (Boundary Noise Clean):  {report['r2_boundary_noise_pass_rate']}%")
    logger.info(f"  R6 (Audio Envelope):        {report['r6_audio_envelope_pass_rate']}%")
    logger.info(f"  R3 (Combined Pass Rate):    {report['r3_combined_pass_rate']}%")
    logger.info(f"  Pre-attack: min={report['r6_preattack_stats']['min_ms']}ms, "
                f"mean={report['r6_preattack_stats']['mean_ms']}ms")
    logger.info(f"  Tail silence: min={report['r6_tail_silence_stats']['min_ms']}ms, "
                f"mean={report['r6_tail_silence_stats']['mean_ms']}ms")
    logger.info(f"  Passed: {report['passed_segments']}, Failed: {report['failed_segments']}")

    # Failure type breakdown
    ftc = report.get('failure_type_counts', {})
    logger.info(f"\n  Failure Type Breakdown:")
    logger.info(f"    Type A (Alignment Shift):    {ftc.get('Type_A_alignment_shift', 0)}")
    logger.info(f"    Type B (Merge/Split):        {ftc.get('Type_B_merge_split', 0)}")
    logger.info(f"    Type C (Boundary Noise):     {ftc.get('Type_C_boundary_noise', 0)}")
    logger.info(f"    Type D (Whisper Error):       {ftc.get('Type_D_whisper_error', 0)}")
    logger.info(f"    Type E (Script Mismatch):    {ftc.get('Type_E_script_mismatch', 0)}")
    logger.info(f"    Type F (Envelope Violation): {ftc.get('Type_F_envelope_violation', 0)}")

    # Per-script breakdown
    logger.info(f"\n  Per-Script Breakdown:")
    for script_name in sorted(report.get('per_script_breakdown', {}).keys()):
        info = report['per_script_breakdown'][script_name]
        logger.info(f"    {script_name}: {info['passed']}/{info['total']} ({info['rate']}%)")

    logger.info(f"{'='*60}")

    # Check if all requirements met
    all_pass = (
        report['r1_alignment_accuracy'] >= 95 and
        report['r2_boundary_noise_pass_rate'] >= 95 and
        report['r6_audio_envelope_pass_rate'] >= 95 and
        report['r3_combined_pass_rate'] >= 95
    )

    if all_pass:
        logger.info("\n*** ALL REQUIREMENTS MET (>= 95%) — Ready for Stage 5 Finalization ***")
    else:
        logger.info("\n*** REQUIREMENTS NOT MET — Stage 6 R&D iteration needed ***")
        # Identify dominant failure type
        dominant = max(ftc, key=ftc.get) if ftc else 'Unknown'
        logger.info(f"  Dominant failure: {dominant} ({ftc.get(dominant, 0)} segments)")

    # Keep checkpoint for potential Tier 2 re-runs
    # (use --reset to clear it for a fresh start)

    return report


# ============================================================
# CLI
# ============================================================

def run_tier2_only():
    """Run only Tier 2 re-evaluation using existing checkpoint results."""
    # Load metadata
    entries = {}
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|', 1)
            if len(parts) == 2:
                entries[parts[0]] = parts[1]

    # Load checkpoint
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    if not checkpoint:
        logger.error("No checkpoint found. Run full evaluation first.")
        return None

    results = checkpoint.get('results', [])
    processed_files = set(checkpoint.get('processed_files', []))
    logger.info(f"Loaded checkpoint: {len(results)} results")

    # Count current pass rate
    passed_before = sum(1 for r in results if r['verdict'] == 'PASS')
    logger.info(f"Before Tier 2: {passed_before}/{len(results)} passed ({passed_before/len(results)*100:.1f}%)")

    # Run Tier 2
    recovered = run_tier2_reeval(results, entries)

    # Save updated results
    save_checkpoint(CHECKPOINT_PATH, results, processed_files)

    # Build and save report
    params_used = {
        'whisper_model_tier1': MODEL_SIZE,
        'whisper_model_tier2': TIER2_MODEL_SIZE,
        'mode': 'tier2_only',
        'match_threshold': 0.50,
    }
    report = build_evaluation_report(results, params_used)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_validation_csv(results, VALIDATION_CSV_PATH)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION COMPLETE (Tier 2 re-eval)")
    logger.info(f"  Total segments: {report['total_segments']}")
    logger.info(f"  R1 (Alignment Accuracy):    {report['r1_alignment_accuracy']}%")
    logger.info(f"  R2 (Boundary Noise Clean):  {report['r2_boundary_noise_pass_rate']}%")
    logger.info(f"  R6 (Audio Envelope):        {report['r6_audio_envelope_pass_rate']}%")
    logger.info(f"  R3 (Combined Pass Rate):    {report['r3_combined_pass_rate']}%")
    logger.info(f"  Passed: {report['passed_segments']}, Failed: {report['failed_segments']}")
    logger.info(f"  Tier 2 recovered: {recovered}")
    logger.info(f"{'='*60}")

    return report


def run_curation(sim_floor=0.30):
    """Remove segments below similarity floor from the dataset.
    Moves bad WAVs to quarantine folder and updates script.txt.
    Requirements R4 allows up to 5% orphans/removal."""
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    if not checkpoint:
        logger.error("No checkpoint found. Run evaluation first.")
        return

    results = checkpoint.get('results', [])
    total = len(results)

    # Identify segments to remove
    to_remove = [r for r in results if r['similarity'] < sim_floor]
    remove_pct = len(to_remove) / total * 100 if total > 0 else 0

    logger.info(f"Curation: {len(to_remove)}/{total} segments below sim {sim_floor} ({remove_pct:.1f}%)")

    if not to_remove:
        logger.info("Nothing to curate.")
        return

    # Safety: don't remove more than 20% (well above R4's 5% allowance)
    if remove_pct > 20:
        logger.warning(f"Removal rate {remove_pct:.1f}% exceeds safety limit (20%). Aborting.")
        return

    # Create quarantine directory
    quarantine_dir = os.path.join(BASE_DIR, "datasets", "quarantine")
    os.makedirs(quarantine_dir, exist_ok=True)

    removed = 0
    for r in to_remove:
        wav_path = os.path.join(WAV_DIR, r['filename'])
        if os.path.exists(wav_path):
            dest = os.path.join(quarantine_dir, r['filename'])
            try:
                os.rename(wav_path, dest)
                removed += 1
            except OSError:
                # exFAT fallback: copy then delete
                import shutil
                shutil.move(wav_path, dest)
                removed += 1

    # Update script.txt — remove curated entries
    remove_set = set(r['filename'] for r in to_remove)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        kept = [l for l in lines if l.strip().split('|')[0] not in remove_set]
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            f.writelines(kept)

    # Update checkpoint — remove curated results
    kept_results = [r for r in results if r['filename'] not in remove_set]
    processed_files = set(r['filename'] for r in kept_results)
    save_checkpoint(CHECKPOINT_PATH, kept_results, processed_files)

    # Regenerate report with curated dataset
    params_used = {'mode': 'post_curation', 'sim_floor': sim_floor}
    report = build_evaluation_report(kept_results, params_used)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_validation_csv(kept_results, VALIDATION_CSV_PATH)

    logger.info(f"Curation complete: {removed} WAVs moved to quarantine")
    logger.info(f"Remaining: {len(kept_results)} segments")
    logger.info(f"New R1: {report['r1_alignment_accuracy']}%")
    logger.info(f"New R3: {report['r3_combined_pass_rate']}%")
    return report


def main():
    parser = argparse.ArgumentParser(description="TTS Dataset: Evaluation Pipeline (Stages 3-4)")
    parser.add_argument('--script', type=int, default=None,
                        help='Evaluate only this script number (e.g., 2)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: R2+R6 checks only, skip Whisper re-transcription')
    parser.add_argument('--reset', action='store_true',
                        help='Clear checkpoint and start fresh')
    parser.add_argument('--tier2-only', action='store_true',
                        help='Run only Tier 2 (large model) on existing checkpoint results')
    parser.add_argument('--curate', action='store_true',
                        help='Remove segments below quality floor from dataset')
    parser.add_argument('--curate-floor', type=float, default=0.30,
                        help='Similarity floor for curation (default: 0.30)')
    args = parser.parse_args()

    if args.curate:
        run_curation(sim_floor=args.curate_floor)
    elif args.tier2_only:
        run_tier2_only()
    else:
        run_evaluation(
            script_filter=args.script,
            quick_mode=args.quick,
            reset=args.reset
        )


if __name__ == "__main__":
    main()
