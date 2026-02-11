# -*- coding: utf-8 -*-
"""
Script_2 Audio Alignment and Splitting Tool
- CUDA-only (fails fast if no GPU)
- Resume support via checkpoint file
- Proper Korean encoding handling
- Wide search window (500 lines) anchored by audio file's expected line range
"""

import os
import sys
import re
import json
import difflib
import warnings
import torch
from pathlib import Path

# Force UTF-8 for stdout/stderr on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")

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


def normalize_text(text):
    """Remove punctuation, keep Korean/English/numbers for comparison"""
    text = re.sub(r'[^가-힣a-zA-Z0-9]', '', text)
    return text


def load_script(script_path):
    """Load script file with proper Korean encoding."""
    sentences = {}
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']

    for enc in encodings:
        try:
            with open(script_path, 'r', encoding=enc) as f:
                lines = f.readlines()
            if lines:
                print(f"  [OK] Loaded script with encoding: {enc}")
                break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        print("  [ERROR] Failed to load script with any encoding")
        return {}

    line_num = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line_num += 1
        sentences[line_num] = line

    print(f"  [OK] Loaded {line_num} sentences from script")
    return sentences


def get_audio_files(audio_dir, script_id=2):
    """Get Script_2 audio files sorted by sequence, with line ranges."""
    pattern = re.compile(rf'Script_{script_id}_(\d+)-(\d+)\.wav')
    files = []

    for f in os.listdir(audio_dir):
        match = pattern.match(f)
        if match:
            start_num = int(match.group(1))
            end_num = int(match.group(2))
            files.append((start_num, end_num, os.path.join(audio_dir, f)))

    files.sort(key=lambda x: x[0])
    return files


def load_checkpoint(checkpoint_path):
    """Load checkpoint if exists."""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return None


def save_checkpoint(checkpoint_path, data):
    """Save checkpoint."""
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================
# CONFIGURATION
# ============================================
SEG_SEARCH_WINDOW = 25    # Per-segment: search up to N lines forward from current position
MATCH_THRESHOLD = 0.25    # Minimum similarity score to accept a match
SKIP_PENALTY = 0.01       # Penalty per skipped line (distance from current_script_line)
CONSEC_FAIL_LIMIT = 5     # Consecutive unmatched segments before advancing script line


def process_script2(model_size="medium"):
    """Main processing function for Script_2 with CUDA and resume support."""

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    AUDIO_DIR = BASE_DIR / "rawdata" / "audio"
    SCRIPT_PATH = BASE_DIR / "rawdata" / "Scripts" / "Script_2_A0.txt"
    OUTPUT_DIR = BASE_DIR / "datasets"
    WAVS_DIR = OUTPUT_DIR / "wavs"
    METADATA_PATH = OUTPUT_DIR / "metadata_script2.txt"
    SKIPPED_LOG = BASE_DIR / "skipped_script2.log"
    CHECKPOINT_PATH = BASE_DIR / "checkpoint_script2.json"

    print("=" * 60)
    print("Script_2 Processing Tool (CUDA + Resume)")
    print("=" * 60)

    # ============================================
    # CUDA CHECK - FAIL FAST IF NOT AVAILABLE
    # ============================================
    if not torch.cuda.is_available():
        print("\n[FATAL ERROR] CUDA is NOT available!")
        print("This script requires a CUDA-capable GPU.")
        print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
        sys.exit(1)

    device = "cuda"
    use_fp16 = True  # FP16 for faster inference on GPU

    print(f"\n[GPU INFO]")
    print(f"  Device: {device}")
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  FP16: {use_fp16}")
    print(f"  Model: {model_size}")

    print(f"\n[SEARCH CONFIG]")
    print(f"  Per-segment window: {SEG_SEARCH_WINDOW} lines forward")
    print(f"  Match threshold: {MATCH_THRESHOLD}")
    print(f"  Skip penalty: {SKIP_PENALTY}/line")
    print(f"  Consecutive fail limit: {CONSEC_FAIL_LIMIT}")

    # Create output directories
    WAVS_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================
    # LOAD CHECKPOINT
    # ============================================
    checkpoint = load_checkpoint(CHECKPOINT_PATH)

    if checkpoint:
        resume_file_idx = checkpoint.get('next_file_idx', 0)
        resume_script_line = checkpoint.get('next_script_line', 1)
        total_matched_so_far = checkpoint.get('total_matched', 0)
        print(f"\n[RESUME INFO]")
        print(f"  Checkpoint found: {CHECKPOINT_PATH}")
        print(f"  Resume from file index: {resume_file_idx}")
        print(f"  Resume from script line: {resume_script_line}")
        print(f"  Already matched: {total_matched_so_far}")
    else:
        resume_file_idx = 0
        resume_script_line = 1
        total_matched_so_far = 0
        print(f"\n[FRESH START]")
        print(f"  No checkpoint found, starting from beginning")

    print()

    # ============================================
    # LOAD SCRIPT
    # ============================================
    print("[1/4] Loading script...")
    script_sentences = load_script(str(SCRIPT_PATH))
    if not script_sentences:
        print("ERROR: Could not load script")
        return

    total_sentences = len(script_sentences)
    print(f"  Total sentences to match: {total_sentences}")
    print()

    # ============================================
    # GET AUDIO FILES
    # ============================================
    print("[2/4] Finding audio files...")
    audio_file_info = get_audio_files(str(AUDIO_DIR), script_id=2)
    if not audio_file_info:
        print("ERROR: No Script_2 audio files found")
        return

    print(f"  Found {len(audio_file_info)} audio files:")
    for idx, (start_ln, end_ln, af) in enumerate(audio_file_info):
        marker = ">>>" if idx == resume_file_idx else "   "
        print(f"  {marker} [{idx}] {os.path.basename(af)}  (lines {start_ln}-{end_ln})")
    print()

    # ============================================
    # LOAD WHISPER MODEL (CUDA)
    # ============================================
    print(f"[3/4] Loading Whisper model ({model_size}) on CUDA...")
    model = whisper.load_model(model_size, device=device)
    print("  Model loaded successfully on GPU")
    print()

    # ============================================
    # INITIALIZE STATE
    # ============================================
    current_script_line = resume_script_line
    total_matched = total_matched_so_far
    total_skipped = 0
    consec_fails = 0

    # Initialize/append to logs
    if resume_file_idx == 0:
        # Fresh start - clear logs
        with open(SKIPPED_LOG, 'w', encoding='utf-8') as f:
            f.write("AudioFile|ScriptLine|Reason|ScriptText|WhisperText\n")
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            pass  # Clear metadata

    # ============================================
    # PROCESS AUDIO FILES
    # ============================================
    print("[4/4] Processing audio files...")
    print()

    for audio_idx in range(resume_file_idx, len(audio_file_info)):
        file_start_line, file_end_line, audio_path = audio_file_info[audio_idx]
        audio_filename = os.path.basename(audio_path)

        print(f"  [{audio_idx+1}/{len(audio_file_info)}] Processing: {audio_filename}")
        print(f"      File expected lines: {file_start_line}-{file_end_line}")
        print(f"      Starting from script line: {current_script_line}")

        # Transcribe audio with CUDA
        print(f"      Transcribing with Whisper (CUDA)...")
        result = model.transcribe(
            str(audio_path),
            language="ko",
            verbose=False,
            fp16=use_fp16
        )
        segments = result['segments']
        print(f"      Found {len(segments)} segments")

        if not segments:
            print(f"      WARNING: No segments found, skipping file")
            continue

        # Load audio for slicing
        audio = AudioSegment.from_wav(audio_path)

        # Match segments to script lines
        file_matched = 0
        file_skipped_lines = []

        seg_idx = 0
        used_segments = set()
        consec_fails = 0

        while seg_idx < len(segments) and current_script_line <= total_sentences:
            # Skip already-used segments
            if seg_idx in used_segments:
                seg_idx += 1
                continue

            seg = segments[seg_idx]
            seg_text = seg['text'].strip()
            norm_seg = normalize_text(seg_text)

            # Skip very short segments
            if len(norm_seg) < 2:
                seg_idx += 1
                continue

            # Try different segment merge combinations (1, 2, 3 segments)
            best_score = 0
            best_line = None
            best_line_text = ""
            best_merge_count = 1
            best_end_time = seg['end']

            for merge_count in [1, 2, 3]:
                if seg_idx + merge_count > len(segments):
                    break

                # Build merged segment
                merged_segs = segments[seg_idx:seg_idx + merge_count]
                merged_text = " ".join(s['text'].strip() for s in merged_segs)
                norm_merged = normalize_text(merged_text)

                if len(norm_merged) < 3:
                    continue

                # Search forward in script: up to SEG_SEARCH_WINDOW lines
                search_start = current_script_line
                search_end = min(total_sentences + 1, current_script_line + SEG_SEARCH_WINDOW)

                for line_num in range(search_start, search_end):
                    if line_num not in script_sentences:
                        continue
                    target_text = script_sentences[line_num]
                    norm_target = normalize_text(target_text)

                    if len(norm_target) < 2:
                        continue

                    score = difflib.SequenceMatcher(None, norm_merged, norm_target).ratio()

                    # Penalty for skipping lines (prefer closer matches)
                    skip_count = line_num - current_script_line
                    adjusted_score = score - (skip_count * SKIP_PENALTY)

                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_line = line_num
                        best_line_text = target_text
                        best_merge_count = merge_count
                        best_end_time = merged_segs[-1]['end']

            # Accept match if score is reasonable
            if best_score >= MATCH_THRESHOLD and best_line is not None:
                consec_fails = 0
                # Mark skipped lines
                for skip_line in range(current_script_line, best_line):
                    if skip_line in script_sentences:
                        skip_text = script_sentences[skip_line]
                        file_skipped_lines.append(f"{audio_filename}|{skip_line}|SKIPPED|{skip_text}|")

                # Extract audio with padding
                start_time = seg['start']
                start_ms = max(0, int(start_time * 1000) - 50)
                end_ms = min(len(audio), int(best_end_time * 1000) + 50)
                chunk = audio[start_ms:end_ms]

                # Save audio chunk (delete if exists to avoid permission errors)
                out_filename = f"Script_2_{best_line:04d}.wav"
                out_path = WAVS_DIR / out_filename
                if out_path.exists():
                    try:
                        out_path.unlink()
                    except:
                        pass
                try:
                    chunk.export(str(out_path), format="wav")
                except PermissionError:
                    print(f"      [WARN] Cannot write {out_filename} (locked), skipping export")

                # Append to metadata
                with open(METADATA_PATH, 'a', encoding='utf-8') as f:
                    f.write(f"{out_filename}|{best_line_text}\n")

                file_matched += 1
                total_matched += 1
                current_script_line = best_line + 1

                # Mark consumed segments
                for i in range(best_merge_count):
                    used_segments.add(seg_idx + i)
                seg_idx += best_merge_count
            else:
                # No good match found - track consecutive failures
                consec_fails += 1

                # If stuck for too many segments, advance script line too
                if consec_fails >= CONSEC_FAIL_LIMIT and current_script_line <= total_sentences:
                    if current_script_line in script_sentences:
                        file_skipped_lines.append(f"{audio_filename}|{current_script_line}|NO_MATCH|{script_sentences[current_script_line]}|{seg_text}")
                    current_script_line += 1
                    consec_fails = 0

                seg_idx += 1

        # Log skipped lines
        if file_skipped_lines:
            with open(SKIPPED_LOG, 'a', encoding='utf-8') as f:
                for line in file_skipped_lines:
                    f.write(line + "\n")
        total_skipped += len(file_skipped_lines)

        print(f"      Matched: {file_matched}, Skipped: {len(file_skipped_lines)}")
        print(f"      Next script line: {current_script_line}")
        print(f"      Total matched so far: {total_matched}")

        # ============================================
        # SAVE CHECKPOINT AFTER EACH FILE
        # ============================================
        save_checkpoint(CHECKPOINT_PATH, {
            'next_file_idx': audio_idx + 1,
            'next_script_line': current_script_line,
            'total_matched': total_matched,
            'last_completed_file': audio_filename
        })
        print(f"      [CHECKPOINT SAVED]")
        print()

    # ============================================
    # FINAL SUMMARY
    # ============================================
    print("=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"  Total script lines: {total_sentences}")
    print(f"  Successfully matched: {total_matched}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Match rate: {total_matched / total_sentences * 100:.1f}%")
    print()
    print(f"  Output WAVs: {WAVS_DIR}")
    print(f"  Metadata: {METADATA_PATH}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 60)

    # Clear checkpoint on successful completion
    if current_script_line > total_sentences or audio_idx >= len(audio_file_info) - 1:
        print("\n[All files processed - checkpoint cleared]")
        CHECKPOINT_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process Script_2 audio files (CUDA required)")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint and start fresh")
    args = parser.parse_args()

    if args.reset:
        checkpoint_path = Path(__file__).parent.parent / "checkpoint_script2.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("[Checkpoint cleared]")

    process_script2(model_size=args.model)
