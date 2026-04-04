# -*- coding: utf-8 -*-
"""
TTS Dataset: Align & Split Pipeline
=====================================
Processes raw audio files against ground-truth scripts using Whisper ASR.
Produces individually sliced WAV segments with metadata.

Stages 1-2 of the TTS Dataset Pipeline (see OrderSheets/TTS_DATASET_PIPELINE_REQUIREMENTS.md)

Usage:
  python align_and_split.py                     # Process all scripts
  python align_and_split.py --script 2          # Process Script_2 only
  python align_and_split.py --model large       # Use larger Whisper model
  python align_and_split.py --dry-run           # Show what would be processed without running
"""

import os
import sys
import glob
import re
import argparse
import json
import logging
import datetime
import difflib
import struct
import math
import time
import warnings
import unicodedata

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore")

# Force UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
    import whisper
    from pydub import AudioSegment
except ImportError:
    print("Installing requirements...")
    os.system("pip install openai-whisper pydub static-ffmpeg torch")
    import static_ffmpeg
    static_ffmpeg.add_paths()
    import whisper
    from pydub import AudioSegment

from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.getcwd()
RAW_AUDIO_DIR = os.path.join(BASE_DIR, "rawdata", "audio")
SCRIPT_DIR = os.path.join(BASE_DIR, "rawdata", "Scripts")
# Canonical output (read-only reference for evaluation/training)
CANONICAL_WAV_DIR = os.path.join(BASE_DIR, "datasets", "wavs")
CANONICAL_METADATA = os.path.join(BASE_DIR, "datasets", "script.txt")
# Versioned output — set at runtime via _make_versioned_output_dir()
OUTPUT_WAV_DIR = None  # populated in align_and_split()
METADATA_PATH = None   # populated in align_and_split()
LOG_DIR = os.path.join(BASE_DIR, "logs")


def _make_versioned_output_dir(base_dir, tag=None):
    """Create a versioned output directory: datasets/output_YYYYMMDD_vN/wavs.

    Never overwrites existing runs. Bumps version number automatically.
    Returns (wav_dir, metadata_path, run_dir).
    """
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    prefix = f"output_{date_str}"
    if tag:
        prefix = f"output_{tag}_{date_str}"

    output_root = os.path.join(base_dir, "datasets")
    version = 1
    while True:
        run_name = f"{prefix}_v{version}"
        run_dir = os.path.join(output_root, run_name)
        if not os.path.exists(run_dir):
            break
        version += 1

    wav_dir = os.path.join(run_dir, "wavs")
    metadata_path = os.path.join(run_dir, "script.txt")
    os.makedirs(wav_dir, exist_ok=True)
    return wav_dir, metadata_path, run_dir

MODEL_SIZE = "medium"
LANGUAGE = "ko"
CHECKPOINT_PATH = os.path.join(LOG_DIR, "pipeline_checkpoint.json")

# Alignment tuning (proven optimal for Korean TTS)
SEG_SEARCH_WINDOW = 25       # Forward-only: search up to N lines ahead
SKIP_PENALTY = 0.01          # Similarity penalty per skipped script line
MATCH_THRESHOLD = 0.50       # Minimum adjusted similarity to accept a match
CONSEC_FAIL_LIMIT = 10       # After N consecutive failures, try re-sync
MAX_MERGE = 5                # Maximum consecutive segments to merge
AUDIO_PAD_MS = 50            # Base padding in ms (50ms proven optimal; 100ms caused bleed in Iter 4a)
MIN_GAP_FOR_PAD_MS = 30      # If gap to neighbor < this, zero-pad on that edge (30ms for dense Korean)
TAIL_EXTEND_MAX_MS = 400     # Max extra right padding to reach silence (prevents mid-speech cutoff)
FADE_MS = 10                 # Fade-in duration in ms
FADE_OUT_MS = 5              # Fade-out duration in ms (shorter to preserve speech tail)
# Stage 2: Post-processing / R6 Audio Envelope
PREATTACK_SILENCE_MS = 400    # Pre-attack silence (ms) — generous for TTS training
TAIL_SILENCE_MS = 730        # Tail silence (ms) — generous for TTS training
SILENCE_THRESHOLD_DB = -65   # RMS threshold for silence detection (dB, matches recording noise floor)
RMS_WINDOW_MS = 10           # RMS sliding window size (ms)
PEAK_NORMALIZE_DB = -1.0     # Peak normalization target (dB)
ONSET_SAFETY_MS = 30         # Pull onset back by this much to preserve consonant attacks
OFFSET_SAFETY_MS = 120       # Extend offset to preserve natural speech decay (120ms for Korean endings)

# Set up logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(LOG_DIR, "align_and_split.log"),
            encoding='utf-8', mode='w'
        )
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_text(text):
    """Remove punctuation, keep Korean/English/numbers for comparison."""
    return re.sub(r'[^가-힣a-zA-Z0-9]', '', text)


def load_script(script_path, start_line=1):
    """Load script file with proper Korean encoding.
    Scripts are plain text, one sentence per line (no index prefix).
    Returns (dict {line_num: text}, encoding_used)."""
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']
    for enc in encodings:
        try:
            with open(script_path, 'r', encoding=enc) as f:
                lines = f.readlines()
            if lines:
                sentences = {}
                line_num = 0
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    line_num += 1
                    if line_num >= start_line:
                        sentences[line_num] = line
                return sentences, enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return {}, None


def parse_audio_filename(filename):
    """Parse script number and line range from audio filename.
    e.g. Script_1_1-122.wav -> (1, 1, 122)"""
    match = re.search(r"Script_(\d+)_(\d+)-(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None


def apply_fade(audio_segment, fade_in_ms=FADE_MS, fade_out_ms=FADE_MS):
    """Apply fade-in and fade-out to avoid clicks at boundaries."""
    if len(audio_segment) < fade_in_ms + fade_out_ms + 10:
        return audio_segment
    return audio_segment.fade_in(fade_in_ms).fade_out(fade_out_ms)


def compute_rms_db(audio_segment):
    """Compute RMS level in dB for an audio segment."""
    if len(audio_segment) == 0:
        return -100.0
    rms = audio_segment.rms
    if rms == 0:
        return -100.0
    return 20 * math.log10(rms / 32768.0)



# Korean formal endings that have natural micro-pauses which Whisper may
# misinterpret as segment boundaries, causing word-final truncation.
# Ordered longest-first so the most specific pattern matches first.
KOREAN_FORMAL_ENDINGS = [
    "것이었습니다", "것입니다",
    "었습니다", "였습니다", "겠습니다",
    "었습니까", "였습니까", "겠습니까",
    "하십시오", "마십시오",
    "습니다", "습니까",
    "십시오", "십시요",
    "읍시다", "합시다",
    "으셨어요",
    "으세요", "하세요",
]


def _gt_ends_with_formal(norm_gt):
    """Check if normalized GT text ends with a Korean formal ending.
    Returns the matched ending pattern or None."""
    for ending in KOREAN_FORMAL_ENDINGS:
        if norm_gt.endswith(ending):
            return ending
    return None


def refine_boundaries_with_words(merged_segs, gt_text, normalize_fn):
    """Use word-level timestamps to find precise GT text boundaries.

    When a Whisper segment contains extra text (bleed from adjacent segments),
    this function finds where the GT text actually starts/ends within the
    segment's word list and returns tighter timestamps.

    Includes protection against Korean formal ending truncation:
    if the refined end boundary would cut off a formal ending (-습니다, etc.),
    the boundary is extended to include the full ending.

    Returns (refined_start_sec, refined_end_sec) or None if no word data.
    """
    # Collect all words from merged segments
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

    # Build cumulative text from the start, find where similarity peaks
    # This detects prefix bleed: words before GT text starts
    best_start_idx = 0
    best_start_sim = 0.0

    for start_idx in range(min(len(all_words), 8)):
        # Text from this word onwards
        suffix_text = ''.join(w.get('word', '') for w in all_words[start_idx:])
        norm_suffix = normalize_fn(suffix_text)
        if len(norm_suffix) < len(norm_gt) * 0.4:
            break
        sim = difflib.SequenceMatcher(None, norm_suffix, norm_gt).ratio()
        if sim > best_start_sim:
            best_start_sim = sim
            best_start_idx = start_idx

    # Find where GT text ends: try trimming words from the end
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

    # Only use refined boundaries if they improve the match
    # (i.e., trimming actually helped)
    full_text = ''.join(w.get('word', '') for w in all_words)
    norm_full = normalize_fn(full_text)
    full_sim = difflib.SequenceMatcher(None, norm_full, norm_gt).ratio()

    if best_start_sim > full_sim + 0.02 or best_end_sim > full_sim + 0.02:
        refined_start = all_words[best_start_idx].get('start', all_words[0].get('start'))
        refined_end = all_words[best_end_idx].get('end', all_words[-1].get('end'))

        # --- Korean formal ending protection ---
        # Check if the refined text is missing a formal ending that GT has.
        # If so, extend end boundary to include more words.
        formal_ending = _gt_ends_with_formal(norm_gt)
        if formal_ending is not None:
            refined_text = ''.join(
                normalize_fn(w.get('word', ''))
                for w in all_words[best_start_idx:best_end_idx + 1]
            )
            if not refined_text.endswith(formal_ending):
                # Ending was truncated — extend to include remaining words
                # up to the full segment boundary
                extended_end_idx = len(all_words) - 1
                refined_end = all_words[extended_end_idx].get(
                    'end', all_words[-1].get('end'))
                logger.debug(
                    f"    Formal ending protection: extended end from word "
                    f"{best_end_idx} to {extended_end_idx} "
                    f"(preserving '{formal_ending}')")

        return refined_start, refined_end

    return None


# ============================================================
# STAGE 2: DSP UTILITY FUNCTIONS
# ============================================================

def find_nearest_zero_crossing(samples, center_idx, sr=48000, search_ms=10):
    """Find nearest zero-crossing within ±search_ms of center_idx.
    Returns the index of the closest zero-crossing, or center_idx if none found."""
    search_radius = int(sr * search_ms / 1000)  # ±10ms = ±480 samples at 48kHz
    start = max(0, center_idx - search_radius)
    end = min(len(samples) - 1, center_idx + search_radius)

    if end - start < 2:
        return center_idx

    # Find sign changes
    signs = np.sign(samples[start:end])
    # Handle exact zeros
    signs[signs == 0] = 1
    crossings = np.where(np.diff(signs))[0] + start

    if len(crossings) == 0:
        return center_idx

    # Return closest crossing to center
    distances = np.abs(crossings - center_idx)
    return int(crossings[np.argmin(distances)])


def compute_rms_windowed(samples, sr=48000, window_ms=RMS_WINDOW_MS):
    """Compute RMS in non-overlapping windows. Returns array of RMS values (dB)."""
    window_size = int(sr * window_ms / 1000)
    if window_size < 1:
        window_size = 1
    n_windows = len(samples) // window_size
    if n_windows == 0:
        rms_lin = np.sqrt(np.mean(samples ** 2))
        return np.array([20 * np.log10(max(rms_lin, 1e-10))])

    # Reshape into windows and compute RMS
    trimmed = samples[:n_windows * window_size].reshape(n_windows, window_size)
    rms_lin = np.sqrt(np.mean(trimmed ** 2, axis=1))
    rms_db = 20 * np.log10(np.maximum(rms_lin, 1e-10))
    return rms_db


def find_voice_onset_offset(samples, sr=48000, threshold_db=SILENCE_THRESHOLD_DB,
                            window_ms=RMS_WINDOW_MS,
                            sustained_silence_ms=1000):
    """Find voice onset and offset sample indices using RMS sliding window.

    Offset detection requires sustained silence: after the last voiced window,
    at least sustained_silence_ms of consecutive silence (avg < threshold_db)
    must follow to confirm speech has truly ended. If speech resumes within
    that window, the offset extends past it and checks again.

    Returns (onset_sample, offset_sample) where onset is the START of the first
    voiced window and offset is the END of the last voiced window."""
    window_size = int(sr * window_ms / 1000)
    rms_db = compute_rms_windowed(samples, sr, window_ms)
    n_windows = len(rms_db)

    voiced = np.where(rms_db >= threshold_db)[0]
    if len(voiced) == 0:
        # All silence — return full range
        return 0, len(samples)

    onset_window = voiced[0]

    # --- Sustained silence check for offset ---
    # Number of consecutive silent windows needed to confirm end-of-speech
    silence_windows_needed = max(1, int(sustained_silence_ms / window_ms))

    # Start from the last voiced window and scan forward
    candidate_offset = voiced[-1]

    # Walk forward from the candidate: verify silence is sustained
    check_start = candidate_offset + 1
    while check_start < n_windows:
        # Count consecutive silent windows from check_start
        silent_count = 0
        for w in range(check_start, min(check_start + silence_windows_needed, n_windows)):
            if rms_db[w] < threshold_db:
                silent_count += 1
            else:
                # Speech resumed — extend offset to this voiced window
                # and find the next silence gap after it
                candidate_offset = w
                break

        if silent_count >= silence_windows_needed:
            # Confirmed: sustained silence found, offset is valid
            break
        elif silent_count == min(check_start + silence_windows_needed, n_windows) - check_start:
            # Reached end of audio with all silence — offset is valid
            break
        else:
            # Speech resumed at candidate_offset, scan forward from there
            check_start = candidate_offset + 1

    onset_sample = onset_window * window_size
    offset_sample = min((candidate_offset + 1) * window_size, len(samples))

    return onset_sample, offset_sample


def make_raised_cosine_fade(length):
    """Create a raised-cosine (Hann) fade curve of given length in samples."""
    if length <= 0:
        return np.array([], dtype=np.float64)
    return 0.5 * (1 - np.cos(np.pi * np.arange(length) / length))


def safe_write_wav(path, samples, sr, subtype='PCM_24'):
    """Write WAV file with exFAT-safe retry logic."""
    # Remove existing file first (exFAT issue)
    if os.path.exists(path):
        for attempt in range(3):
            try:
                os.remove(path)
                break
            except PermissionError:
                time.sleep(0.1)

    for attempt in range(3):
        try:
            sf.write(path, samples, sr, subtype=subtype)
            return True
        except PermissionError:
            time.sleep(0.1)
    logger.warning(f"Failed to write {path} after 3 attempts")
    return False


def save_checkpoint(data):
    """Save pipeline checkpoint for resume capability."""
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    os.rename(tmp, CHECKPOINT_PATH)
    logger.debug(f"Checkpoint saved: {len(data.get('scripts_done', []))} scripts done")


def load_checkpoint():
    """Load pipeline checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def clear_checkpoint():
    """Remove checkpoint file after successful completion."""
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)


def post_process_wavs(wav_dir, wav_filter=None):
    """Stage 2: Post-process WAVs in-place.

    Args:
        wav_dir: Directory containing output WAVs
        wav_filter: If set, list of WAV basenames to process (others skipped)

    Processing order per requirements Section 9:
    1. Zero-crossing snap at start/end boundaries
    2. Voice onset/offset detection
    3. Trim to voiced region
    4. Fade (raised-cosine on voiced region only)
    5. Peak normalize voiced region to -1dB
    6. R6 Envelope: prepend 50ms silence + append 300ms silence (LAST step)
    7. Export as 48KHz, PCM_24, mono
    """
    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if wav_filter is not None:
        wav_files = [f for f in wav_files if os.path.basename(f) in wav_filter]
    if not wav_files:
        logger.warning("No WAV files to post-process")
        return

    sr_target = 48000
    preattack_samples = int(sr_target * PREATTACK_SILENCE_MS / 1000)   # 2400
    tail_samples = int(sr_target * TAIL_SILENCE_MS / 1000)             # 14400
    fade_in_samples = int(sr_target * FADE_MS / 1000)                   # 480
    fade_out_samples = int(sr_target * FADE_OUT_MS / 1000)             # 240
    
    logger.info(f"\n{'='*50}")
    logger.info(f"STAGE 2: Post-processing {len(wav_files)} WAVs")
    logger.info(f"  R6 envelope: {PREATTACK_SILENCE_MS}ms pre-attack + {TAIL_SILENCE_MS}ms tail")
    logger.info(f"  Fade: {FADE_MS}ms raised-cosine | Normalize: {PEAK_NORMALIZE_DB}dB peak")

    processed = 0
    errors = 0

    for wav_path in tqdm(wav_files, desc="Stage 2 Post-process"):
        try:
            # Load audio as float64 for precision
            samples, sr = sf.read(wav_path, dtype='float64')

            # Ensure mono
            if samples.ndim > 1:
                samples = samples[:, 0]

            # Resample if needed (should already be 48kHz)
            if sr != sr_target:
                logger.warning(f"Unexpected sample rate {sr} in {os.path.basename(wav_path)}")
                # Simple skip — shouldn't happen with our pipeline
                errors += 1
                continue

            # --- Step 1: Zero-crossing snap at boundaries ---
            if len(samples) > 960:  # need enough samples
                new_start = find_nearest_zero_crossing(samples, 0, sr)
                new_end = len(samples) - 1  # Do NOT snap end — preserves speech tail
                if new_end > new_start:
                    samples = samples[new_start:new_end + 1]

            # --- Step 2-3: Voice onset/offset detection + trim ---
            # Apply safety margins to preserve consonant attacks and word-final sounds
            onset, offset = find_voice_onset_offset(samples, sr)
            onset_safety = int(sr * ONSET_SAFETY_MS / 1000)
            offset_safety = int(sr * OFFSET_SAFETY_MS / 1000)
            onset = max(0, onset - onset_safety)
            offset = min(len(samples), offset + offset_safety)
            if onset >= offset:
                # Degenerate — keep as-is but still apply envelope
                voiced = samples
            else:
                voiced = samples[onset:offset]

            if len(voiced) == 0:
                logger.debug(f"Empty voiced region: {os.path.basename(wav_path)}")
                errors += 1
                continue

            # --- Step 4: Fade (on voiced region only) ---
            actual_fade_in = min(fade_in_samples, len(voiced) // 4)
            actual_fade_out = min(fade_out_samples, len(voiced) // 4)

            if actual_fade_in > 0:
                fade_in_curve = make_raised_cosine_fade(actual_fade_in)
                voiced[:actual_fade_in] *= fade_in_curve
            if actual_fade_out > 0:
                fade_out_curve = make_raised_cosine_fade(actual_fade_out)[::-1]
                voiced[-actual_fade_out:] *= fade_out_curve

            # --- Step 5: Peak normalize voiced region to -1dB ---
            peak = np.max(np.abs(voiced))
            if peak > 0:
                target_peak = 10 ** (PEAK_NORMALIZE_DB / 20)  # -1dB ≈ 0.891
                voiced = voiced * (target_peak / peak)

            # --- Step 6: R6 Envelope (LAST step before export) ---
            # Prepend exactly 50ms silence + append exactly 300ms silence
            pre_silence = np.zeros(preattack_samples, dtype=np.float64)
            tail_silence = np.zeros(tail_samples, dtype=np.float64)
            final = np.concatenate([pre_silence, voiced, tail_silence])

            # --- Step 7: Export ---
            # Clip to prevent any overflow
            final = np.clip(final, -1.0, 1.0)
            safe_write_wav(wav_path, final, sr_target, 'PCM_24')
            processed += 1

        except Exception as e:
            logger.error(f"Post-process error on {os.path.basename(wav_path)}: {e}")
            errors += 1

    logger.info(f"Stage 2 complete: {processed} processed, {errors} errors")
    return processed, errors


# ============================================================
# CORE: ALIGN AND SPLIT
# ============================================================

def align_and_split(model_size=MODEL_SIZE, script_filter=None, range_filter=None,
                    resume=True, device_override=None):
    """Main alignment and splitting pipeline.

    Groups audio files by script number and processes each script sequentially,
    maintaining a single current_script_line across all audio files for the same
    script. This is critical because audio filenames don't reliably reflect
    actual content ranges.

    Args:
        model_size: Whisper model size (tiny/base/small/medium/large)
        script_filter: If set, only process this script number (e.g., 2)
        range_filter: If set, tuple (start, end) — only process the audio file
                      whose filename range matches (e.g., (1, 162))
        resume: If True, resume from checkpoint if available
    """
    global OUTPUT_WAV_DIR, METADATA_PATH

    # Create versioned output directory — never overwrite existing data
    tag = f"Script{script_filter}" if script_filter else None
    OUTPUT_WAV_DIR, METADATA_PATH, run_dir = _make_versioned_output_dir(BASE_DIR, tag=tag)
    logger.info(f"Output directory: {run_dir}")

    os.makedirs(OUTPUT_WAV_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Initialize Whisper
    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[{device}] Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size, device=device)

    # Find and group audio files by script number
    audio_files = sorted(glob.glob(os.path.join(RAW_AUDIO_DIR, "*.wav")))
    if not audio_files:
        logger.error("No audio files found in rawdata/audio/")
        return

    # Group by script number: {script_no: [(start, end, path), ...]}
    scripts = {}
    for audio_path in audio_files:
        filename = os.path.basename(audio_path)
        script_no, start_idx, end_idx = parse_audio_filename(filename)
        if script_no is None:
            logger.warning(f"Skipping (filename format mismatch): {filename}")
            continue
        if script_filter is not None and script_no != script_filter:
            continue
        if range_filter is not None:
            if (start_idx, end_idx) != range_filter:
                continue
        if script_no not in scripts:
            scripts[script_no] = []
        scripts[script_no].append((start_idx, end_idx, audio_path))

    # Sort each script's audio files by start line
    for sno in scripts:
        scripts[sno].sort(key=lambda x: x[0])

    logger.info(f"Found {sum(len(v) for v in scripts.values())} audio files across {len(scripts)} scripts")

    # Initialize outputs
    metadata_lines = []
    all_skipped = []
    total_matched = 0
    total_skipped = 0
    total_target_lines = 0
    scripts_done = set()

    # Checkpoint resume
    checkpoint = load_checkpoint() if resume else None
    if checkpoint:
        scripts_done = set(checkpoint.get('scripts_done', []))
        metadata_lines = checkpoint.get('metadata_lines', [])
        total_matched = checkpoint.get('total_matched', 0)
        total_skipped = checkpoint.get('total_skipped', 0)
        total_target_lines = checkpoint.get('total_target_lines', 0)
        logger.info(f"RESUMING from checkpoint: {len(scripts_done)} scripts done "
                    f"({total_matched} matched so far)")

    # Skipped lines log
    skipped_log_path = os.path.join(LOG_DIR, "skipped_lines.log")
    log_mode = 'a' if checkpoint else 'w'
    with open(skipped_log_path, log_mode, encoding='utf-8') as f:
        if not checkpoint:
            f.write("AudioFile|ScriptLine|Reason|ScriptText|WhisperText\n")

    # ---- Pass 1: Transcribe all audio files ----
    # Transcription is done first so we can free the Whisper model before
    # loading large audio files for slicing (avoids MemoryError).
    import gc
    transcription_cache = {}  # key: audio_path -> segments list

    for script_no in sorted(scripts.keys()):
        if script_no in scripts_done:
            continue
        for file_idx, (start_idx, end_idx, audio_path) in enumerate(scripts[script_no]):
            filename = os.path.basename(audio_path)
            logger.info(f"  Transcribing [{file_idx+1}/{len(scripts[script_no])}] {filename}")
            result = model.transcribe(audio_path, language=LANGUAGE, verbose=False,
                                      fp16=(device == "cuda"), word_timestamps=True)
            transcription_cache[audio_path] = result['segments']
            logger.info(f"    {len(result['segments'])} Whisper segments")
            del result

    # Free Whisper model to reclaim memory before audio slicing
    logger.info("Freeing Whisper model...")
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- Pass 2: Alignment and slicing ----
    for script_no in sorted(scripts.keys()):
        # Skip already-completed scripts (checkpoint resume)
        if script_no in scripts_done:
            logger.info(f"\nScript_{script_no}: SKIPPED (already done in checkpoint)")
            continue

        script_filename = f"Script_{script_no}_A0.txt"
        script_path = os.path.join(SCRIPT_DIR, script_filename)

        if not os.path.exists(script_path):
            logger.warning(f"Script file not found: {script_filename}")
            continue

        # Load FULL script (all sentences for this script)
        all_sentences, enc = load_script(script_path)
        if not all_sentences:
            logger.warning(f"Cannot load script: {script_filename}")
            continue

        total_sentences = max(all_sentences.keys())
        # Only count lines covered by available audio files
        audio_file_list = scripts[script_no]
        min_audio_line = min(af[0] for af in audio_file_list)
        max_audio_line = max(af[1] for af in audio_file_list)
        covered_lines = {ln: txt for ln, txt in all_sentences.items()
                         if min_audio_line <= ln <= max_audio_line}
        total_target_lines += len(covered_lines)

        logger.info(f"\n{'='*50}")
        logger.info(f"Script_{script_no}: {len(all_sentences)} total lines (enc: {enc})")
        logger.info(f"  Audio files: {len(audio_file_list)}")
        logger.info(f"  Audio covers lines: {min_audio_line}-{max_audio_line} ({len(covered_lines)} lines)")

        # Maintain a single current_script_line across ALL audio files for this script
        current_script_line = min_audio_line
        script_matched = 0
        script_skipped_entries = []

        for file_idx, (start_idx, end_idx, audio_path) in enumerate(audio_file_list):
            filename = os.path.basename(audio_path)
            segments = transcription_cache.get(audio_path, [])
            logger.info(f"  [{file_idx+1}/{len(audio_file_list)}] {filename}")
            logger.info(f"    Filename range: {start_idx}-{end_idx}, Current line: {current_script_line}")
            logger.info(f"    {len(segments)} Whisper segments (cached)")

            if not segments:
                logger.warning(f"    No segments in {filename}, skipping file")
                continue

            # Load audio for slicing (preserve original quality)
            audio = AudioSegment.from_wav(audio_path)

            # ---- Forward-only sequential matching with segment merging ----
            seg_idx = 0
            used_segments = set()
            consec_fails = 0
            file_matched = 0
            file_skipped_entries = []

            while seg_idx < len(segments) and current_script_line <= total_sentences:
                # Skip already-consumed segments
                if seg_idx in used_segments:
                    seg_idx += 1
                    continue

                seg = segments[seg_idx]
                seg_text = seg['text'].strip()
                norm_seg = normalize_text(seg_text)

                # Skip trivially short segments (noise, breath, etc.)
                if len(norm_seg) < 2:
                    seg_idx += 1
                    continue

                # --- Try 1..MAX_MERGE consecutive segment merges ---
                best_score = 0
                best_line = None
                best_line_text = ""
                best_merge_count = 1
                best_end_time = seg['end']

                for merge_count in range(1, MAX_MERGE + 1):
                    if seg_idx + merge_count > len(segments):
                        break

                    merged_segs = segments[seg_idx:seg_idx + merge_count]
                    merged_text = " ".join(s['text'].strip() for s in merged_segs)
                    norm_merged = normalize_text(merged_text)

                    if len(norm_merged) < 3:
                        continue

                    # Forward-only search window from current position
                    search_end = min(total_sentences + 1,
                                     current_script_line + SEG_SEARCH_WINDOW)

                    for line_num in range(current_script_line, search_end):
                        if line_num not in all_sentences:
                            continue

                        target_text = all_sentences[line_num]
                        norm_target = normalize_text(target_text)

                        if len(norm_target) < 2:
                            continue

                        score = difflib.SequenceMatcher(
                            None, norm_merged, norm_target
                        ).ratio()

                        # Penalize distance from current position
                        skip_count = line_num - current_script_line
                        adjusted_score = score - (skip_count * SKIP_PENALTY)

                        if adjusted_score > best_score:
                            best_score = adjusted_score
                            best_line = line_num
                            best_line_text = target_text
                            best_merge_count = merge_count
                            best_end_time = merged_segs[-1]['end']

                # --- Evaluate best match ---
                if best_score >= MATCH_THRESHOLD and best_line is not None:
                    # --- Match confirmation for borderline matches ---
                    # If score is below 0.80, verify by checking if the next
                    # Whisper segment also roughly matches the next script line.
                    # This prevents false-positive matches from causing pointer
                    # drift and cascading alignment errors.
                    # Raised from 0.70 to 0.80 in Iteration 5 to catch more
                    # false positives, especially with Whisper large alignment.
                    confirmed = True
                    if best_score < 0.80:
                        confirm_seg_idx = seg_idx + best_merge_count
                        confirm_line = best_line + 1
                        if (confirm_seg_idx < len(segments)
                                and confirm_line in all_sentences):
                            confirm_seg_text = normalize_text(
                                segments[confirm_seg_idx]['text'].strip())
                            confirm_line_text = normalize_text(
                                all_sentences[confirm_line])
                            if (len(confirm_seg_text) >= 3
                                    and len(confirm_line_text) >= 2):
                                confirm_score = difflib.SequenceMatcher(
                                    None, confirm_seg_text,
                                    confirm_line_text).ratio()
                                if confirm_score < 0.25:
                                    confirmed = False
                                    logger.debug(
                                        f"    Match rejected (no confirmation): "
                                        f"line {best_line} score={best_score:.3f}, "
                                        f"next confirm={confirm_score:.3f}")

                    if not confirmed:
                        consec_fails += 1
                        seg_idx += 1
                        continue

                    # --- Word-level boundary refinement + gap-aware padding ---
                    raw_start_ms = int(seg['start'] * 1000)
                    raw_end_ms = int(best_end_time * 1000)

                    # Try word-level refinement to eliminate bleed/truncation
                    merged_segs = segments[seg_idx:seg_idx + best_merge_count]
                    refined = refine_boundaries_with_words(
                        merged_segs, best_line_text, normalize_text)
                    used_word_refinement = False
                    if refined is not None:
                        raw_start_ms = int(refined[0] * 1000)
                        raw_end_ms = int(refined[1] * 1000)
                        used_word_refinement = True

                    # Left padding: check gap to previous segment
                    if seg_idx > 0:
                        prev_end_ms = int(segments[seg_idx - 1]['end'] * 1000)
                        left_gap = raw_start_ms - prev_end_ms
                        left_pad = AUDIO_PAD_MS if left_gap >= MIN_GAP_FOR_PAD_MS else 0
                    else:
                        left_pad = AUDIO_PAD_MS

                    # Right padding: adaptive extension to reach silence
                    # Fixed 50ms pad can leave the extraction ending mid-speech
                    # if Whisper's end timestamp falls before the speech naturally
                    # decays. Extend up to TAIL_EXTEND_MAX_MS to reach silence,
                    # capped by the next segment start to prevent actual bleed.
                    #
                    # Korean formal ending protection: when GT ends with a formal
                    # ending (-습니다, -겠습니다, etc.), the ending syllables may
                    # be timestamped as part of the NEXT Whisper segment due to a
                    # micro-pause (e.g., "었" [pause] "습니다"). In this case, the
                    # ending is NOT bleed — it belongs to the current sentence.
                    # We must extend PAST the next segment boundary to capture it.
                    # Stage 2 post-processing will handle any actual bleed via
                    # onset/offset detection and trimming.
                    has_formal_ending = _gt_ends_with_formal(
                        normalize_text(best_line_text)) is not None

                    next_seg_idx = seg_idx + best_merge_count
                    if next_seg_idx < len(segments):
                        next_start_ms = int(segments[next_seg_idx]['start'] * 1000)
                        right_gap = next_start_ms - raw_end_ms

                        if has_formal_ending:
                            # For formal endings: extend aggressively past the
                            # next segment boundary. The ending syllables (습니다,
                            # 겠습니다, etc.) are typically 200-400ms and may be
                            # in the next segment. Allow 650ms extension to
                            # capture the full ending + ~150ms natural decay,
                            # ignoring the next-segment-start cap.
                            right_pad = 650
                        elif right_gap < MIN_GAP_FOR_PAD_MS:
                            right_pad = 0
                        else:
                            safe_limit = max(0, right_gap - 20)
                            right_pad = min(TAIL_EXTEND_MAX_MS, safe_limit)
                    else:
                        right_pad = TAIL_EXTEND_MAX_MS

                    start_ms = max(0, raw_start_ms - left_pad)
                    end_ms = min(len(audio), raw_end_ms + right_pad)

                    chunk = audio[start_ms:end_ms]

                    # Apply fade to avoid clicks
                    chunk = apply_fade(chunk)

                    # Preserve original audio quality (48KHz, 24-bit, mono)
                    # Only convert to mono if needed
                    if chunk.channels > 1:
                        chunk = chunk.set_channels(1)

                    # Save WAV
                    out_filename = f"Script_{script_no}_{best_line:04d}.wav"
                    out_path = os.path.join(OUTPUT_WAV_DIR, out_filename)

                    # Handle exFAT overwrite issue
                    if os.path.exists(out_path):
                        try:
                            os.remove(out_path)
                        except Exception:
                            pass

                    export_ok = False
                    for _attempt in range(3):
                        try:
                            chunk.export(out_path, format="wav")
                            export_ok = True
                            break
                        except PermissionError:
                            time.sleep(0.2)
                            # Try removing the locked file
                            try:
                                os.remove(out_path)
                            except Exception:
                                pass

                    if not export_ok:
                        logger.warning(f"  Cannot write {out_filename} (PermissionError) — skipping line {best_line}")
                        # Still advance past this match to avoid infinite loop
                        current_script_line = best_line + 1
                        for i in range(best_merge_count):
                            used_segments.add(seg_idx + i)
                        seg_idx += best_merge_count
                        continue

                    # --- Accept match ---
                    consec_fails = 0

                    # Log any skipped lines between current position and match
                    for skip_line in range(current_script_line, best_line):
                        if skip_line in all_sentences:
                            entry = (f"{filename}|{skip_line}|SKIPPED"
                                     f"|{all_sentences[skip_line]}|")
                            file_skipped_entries.append(entry)
                            total_skipped += 1

                    metadata_lines.append(f"{out_filename}|{best_line_text}")
                    file_matched += 1

                    # Advance script line past the matched line
                    current_script_line = best_line + 1

                    # Mark all merged segments as consumed
                    for i in range(best_merge_count):
                        used_segments.add(seg_idx + i)
                    seg_idx += best_merge_count

                else:
                    # No match found for this segment
                    consec_fails += 1

                    if consec_fails >= CONSEC_FAIL_LIMIT:
                        # Re-sync: wider search with NEXT segment to find
                        # where we actually are in the script
                        resync_found = False
                        resync_window = SEG_SEARCH_WINDOW * 3  # 75 lines

                        # Try current segment against wider window
                        for resync_merge in range(1, MAX_MERGE + 1):
                            if seg_idx + resync_merge > len(segments):
                                break
                            resync_segs = segments[seg_idx:seg_idx + resync_merge]
                            resync_text = " ".join(
                                s['text'].strip() for s in resync_segs)
                            norm_resync = normalize_text(resync_text)
                            if len(norm_resync) < 3:
                                continue

                            resync_end = min(total_sentences + 1,
                                             current_script_line + resync_window)
                            for rline in range(current_script_line, resync_end):
                                if rline not in all_sentences:
                                    continue
                                rt = normalize_text(all_sentences[rline])
                                if len(rt) < 2:
                                    continue
                                rscore = difflib.SequenceMatcher(
                                    None, norm_resync, rt).ratio()
                                if rscore >= 0.35:  # re-sync threshold
                                    # Log skipped lines
                                    for skip_ln in range(
                                            current_script_line, rline):
                                        if skip_ln in all_sentences:
                                            entry = (
                                                f"{filename}|{skip_ln}|RESYNC_SKIP"
                                                f"|{all_sentences[skip_ln]}|")
                                            file_skipped_entries.append(entry)
                                            total_skipped += 1
                                    current_script_line = rline
                                    resync_found = True
                                    logger.debug(
                                        f"    Re-synced to line {rline}")
                                    break
                            if resync_found:
                                break

                        if not resync_found:
                            # No re-sync possible, advance by 1
                            if current_script_line in all_sentences:
                                entry = (
                                    f"{filename}|{current_script_line}"
                                    f"|NO_MATCH"
                                    f"|{all_sentences[current_script_line]}"
                                    f"|{seg_text}")
                                file_skipped_entries.append(entry)
                                total_skipped += 1
                            current_script_line += 1

                        consec_fails = 0

                    seg_idx += 1

            # Write per-file skipped entries to log
            if file_skipped_entries:
                with open(skipped_log_path, 'a', encoding='utf-8') as f:
                    for entry in file_skipped_entries:
                        f.write(entry + "\n")
                script_skipped_entries.extend(file_skipped_entries)

            script_matched += file_matched
            total_matched += file_matched
            logger.info(f"    Matched: {file_matched}, Script total so far: {script_matched}")

        # Log remaining unmatched lines at end of script (all files processed)
        end_of_script_skipped = []
        for remaining_line in range(current_script_line, max_audio_line + 1):
            if remaining_line in all_sentences:
                entry = (f"Script_{script_no}|{remaining_line}|EOF_UNMATCHED"
                         f"|{all_sentences[remaining_line]}|")
                end_of_script_skipped.append(entry)
                total_skipped += 1

        if end_of_script_skipped:
            with open(skipped_log_path, 'a', encoding='utf-8') as f:
                for entry in end_of_script_skipped:
                    f.write(entry + "\n")
            script_skipped_entries.extend(end_of_script_skipped)

        all_skipped.extend(script_skipped_entries)
        script_rate = (script_matched / len(covered_lines) * 100) if covered_lines else 0
        logger.info(f"  Script_{script_no} DONE: {script_matched}/{len(covered_lines)} ({script_rate:.1f}%)")

        # Save checkpoint after each script
        scripts_done.add(script_no)
        save_checkpoint({
            'scripts_done': list(scripts_done),
            'metadata_lines': metadata_lines,
            'total_matched': total_matched,
            'total_skipped': total_skipped,
            'total_target_lines': total_target_lines,
            'timestamp': datetime.datetime.now().isoformat()
        })

    # ---- Save metadata (versioned output — no merging with canonical) ----
    metadata_lines.sort()

    # Deduplicate
    seen = set()
    unique_metadata = []
    for line in metadata_lines:
        fname = line.split('|')[0]
        if fname not in seen:
            seen.add(fname)
            unique_metadata.append(line)

    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        for line in unique_metadata:
            f.write(line + "\n")

    # ---- Summary ----
    match_rate = (total_matched / total_target_lines * 100) if total_target_lines > 0 else 0

    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE")
    logger.info(f"  Total target lines: {total_target_lines}")
    logger.info(f"  Matched: {total_matched} ({match_rate:.1f}%)")
    logger.info(f"  Skipped: {total_skipped}")
    logger.info(f"  Output WAVs: {OUTPUT_WAV_DIR}")
    logger.info(f"  Metadata: {METADATA_PATH} ({len(unique_metadata)} entries)")
    logger.info(f"  Skipped log: {skipped_log_path}")
    logger.info(f"{'='*60}")

    # Save run parameters for reproducibility
    run_params = {
        "timestamp": datetime.datetime.now().isoformat(),
        "whisper_model": model_size,
        "device": device,
        "search_window": SEG_SEARCH_WINDOW,
        "skip_penalty": SKIP_PENALTY,
        "match_threshold": MATCH_THRESHOLD,
        "consec_fail_limit": CONSEC_FAIL_LIMIT,
        "audio_pad_ms": AUDIO_PAD_MS,
        "min_gap_for_pad_ms": MIN_GAP_FOR_PAD_MS,
        "fade_ms": FADE_MS,
        "total_target_lines": total_target_lines,
        "total_matched": total_matched,
        "total_skipped": total_skipped,
        "match_rate": round(match_rate, 2),
        "script_filter": script_filter,
        "range_filter": list(range_filter) if range_filter else None
    }
    params_path = os.path.join(LOG_DIR, "last_run_params.json")
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(run_params, f, indent=2, ensure_ascii=False)

    # Clear checkpoint on successful completion of Stage 1
    clear_checkpoint()
    logger.info("Stage 1 checkpoint cleared (all scripts done)")

    # ---- Stage 2: Post-process output WAVs ----
    # Only post-process WAVs produced in this run (not the entire directory)
    produced_wavs = set()
    for ml in metadata_lines:
        wav_name = ml.split("|")[0]
        produced_wavs.add(wav_name)
    wav_filter = produced_wavs if (range_filter is not None or script_filter is not None) else None
    logger.info(f"\nStarting Stage 2: Post-processing {len(produced_wavs)} WAVs...")
    post_process_wavs(OUTPUT_WAV_DIR, wav_filter=wav_filter)

    return total_matched, total_skipped, total_target_lines


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="TTS Dataset: Align & Split Pipeline")
    parser.add_argument('--model', default=MODEL_SIZE,
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (default: medium)')
    parser.add_argument('--device', default=None, choices=['cpu', 'cuda'],
                        help='Force device (default: auto-detect)')
    parser.add_argument('--script', type=int, default=None,
                        help='Process only this script number (e.g., 2)')
    parser.add_argument('--range', type=str, default=None, dest='range_str',
                        help='Process only the audio file matching this line range (e.g., 1-162). Requires --script.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be processed without running')
    parser.add_argument('--post-process-only', action='store_true',
                        help='Re-run Stage 2 post-processing without re-running Stage 1')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoint if available (default: True)')
    parser.add_argument('--reset', action='store_true',
                        help='Clear checkpoint and start fresh')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation prompt')
    args = parser.parse_args()

    # Parse --range into a tuple
    range_filter = None
    if args.range_str:
        if args.script is None:
            parser.error("--range requires --script to be set")
        match = re.match(r"(\d+)-(\d+)$", args.range_str)
        if not match:
            parser.error(f"Invalid range format: '{args.range_str}' (expected e.g., 1-162)")
        range_filter = (int(match.group(1)), int(match.group(2)))

    if args.reset:
        clear_checkpoint()
        logger.info("Checkpoint cleared — starting fresh")

    if args.post_process_only:
        logger.info("Running Stage 2 post-processing only...")
        post_process_wavs(CANONICAL_WAV_DIR)
        return

    if args.dry_run:
        audio_files = sorted(glob.glob(os.path.join(RAW_AUDIO_DIR, "*.wav")))
        print(f"\nDry run — {len(audio_files)} audio files found:")
        for af in audio_files:
            fn = os.path.basename(af)
            sno, start, end = parse_audio_filename(fn)
            if sno is not None:
                if args.script is None or sno == args.script:
                    if range_filter is None or (start, end) == range_filter:
                        print(f"  Script_{sno}: lines {start}-{end}  ({fn})")
        return

    # Show target and confirm
    audio_files = sorted(glob.glob(os.path.join(RAW_AUDIO_DIR, "*.wav")))
    targets = []
    for af in audio_files:
        fn = os.path.basename(af)
        sno, start, end = parse_audio_filename(fn)
        if sno is not None:
            if args.script is None or sno == args.script:
                if range_filter is None or (start, end) == range_filter:
                    targets.append((sno, start, end, fn))

    if not targets:
        logger.error("No matching audio files found for the given filters.")
        return

    print(f"\n{'='*50}")
    print(f"Target audio files ({len(targets)}):")
    for sno, start, end, fn in targets:
        print(f"  Script_{sno}: lines {start}-{end}  ({fn})")
    print(f"{'='*50}")
    if not args.yes:
        confirm = input("Proceed? [Y/n] ").strip().lower()
        if confirm not in ('', 'y', 'yes'):
            print("Aborted.")
            return

    align_and_split(model_size=args.model, script_filter=args.script,
                    range_filter=range_filter,
                    resume=args.resume and not args.reset,
                    device_override=args.device)


if __name__ == "__main__":
    main()
