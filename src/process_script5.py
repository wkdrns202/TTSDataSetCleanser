# -*- coding: utf-8 -*-
"""
Script_5 Audio Alignment and Splitting Tool (Lines 201+)
- Processes single file: Script_5_201-541.wav
- Matches against Script_5_A0.txt lines 201-1018
- CUDA-only (fails fast if no GPU)
- Appends results to datasets/script.txt
"""

import os
import sys
import re
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


def load_script(script_path, start_line=201):
    """Load script file with proper Korean encoding, returning lines from start_line onward."""
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
        if line_num >= start_line:
            sentences[line_num] = line

    print(f"  [OK] Loaded {len(sentences)} sentences (lines {start_line}-{line_num})")
    return sentences


# ============================================
# CONFIGURATION
# ============================================
SEG_SEARCH_WINDOW = 25    # Search up to N lines forward AND backward from current position
MATCH_THRESHOLD = 0.25    # Minimum similarity score to accept a match
SKIP_PENALTY = 0.01       # Penalty per skipped line (distance from current_script_line)
CONSEC_FAIL_LIMIT = 5     # Consecutive unmatched segments before advancing script line
START_LINE = 201           # First script line to process


def process_script5(model_size="medium"):
    """Main processing function for Script_5 lines 201+ with CUDA."""

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    AUDIO_PATH = BASE_DIR / "rawdata" / "audio" / "Script_5_201-541.wav"
    SCRIPT_PATH = BASE_DIR / "rawdata" / "Scripts" / "Script_5_A0.txt"
    OUTPUT_DIR = BASE_DIR / "datasets"
    WAVS_DIR = OUTPUT_DIR / "wavs"
    SCRIPT_TXT = OUTPUT_DIR / "script.txt"
    SKIPPED_LOG = BASE_DIR / "skipped_script5.log"

    print("=" * 60)
    print("Script_5 Processing Tool (Lines 201+, CUDA)")
    print("=" * 60)

    # ============================================
    # CUDA CHECK
    # ============================================
    if not torch.cuda.is_available():
        print("\n[FATAL ERROR] CUDA is NOT available!")
        sys.exit(1)

    device = "cuda"
    use_fp16 = True

    print(f"\n[GPU INFO]")
    print(f"  Device: {device}")
    print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"  FP16: {use_fp16}")
    print(f"  Model: {model_size}")

    print(f"\n[SEARCH CONFIG]")
    print(f"  Search window: ±{SEG_SEARCH_WINDOW} lines from current position")
    print(f"  Match threshold: {MATCH_THRESHOLD}")
    print(f"  Skip penalty: {SKIP_PENALTY}/line")
    print(f"  Consecutive fail limit: {CONSEC_FAIL_LIMIT}")
    print(f"  Starting from line: {START_LINE}")

    # Verify input
    if not AUDIO_PATH.exists():
        print(f"\n[ERROR] Audio file not found: {AUDIO_PATH}")
        sys.exit(1)

    # Create output directories
    WAVS_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================
    # LOAD SCRIPT (lines 201+)
    # ============================================
    print("\n[1/4] Loading script...")
    script_sentences = load_script(str(SCRIPT_PATH), start_line=START_LINE)
    if not script_sentences:
        print("ERROR: Could not load script")
        return

    max_line = max(script_sentences.keys())
    print(f"  Lines to process: {START_LINE} to {max_line}")
    print()

    # ============================================
    # LOAD WHISPER MODEL (CUDA)
    # ============================================
    print(f"[2/4] Loading Whisper model ({model_size}) on CUDA...")
    model = whisper.load_model(model_size, device=device)
    print("  Model loaded successfully on GPU")
    print()

    # ============================================
    # TRANSCRIBE AUDIO
    # ============================================
    print(f"[3/4] Transcribing audio: {AUDIO_PATH.name}")
    print(f"  File size: {AUDIO_PATH.stat().st_size / (1024*1024):.0f} MB")
    result = model.transcribe(
        str(AUDIO_PATH),
        language="ko",
        verbose=False,
        fp16=use_fp16
    )
    segments = result['segments']
    print(f"  Found {len(segments)} segments")
    print()

    if not segments:
        print("[ERROR] No segments found in audio")
        return

    # Load audio for slicing
    print("[4/4] Matching segments to script lines...")
    audio = AudioSegment.from_wav(str(AUDIO_PATH))
    print(f"  Audio duration: {len(audio)/1000:.1f}s")

    # ============================================
    # MATCH SEGMENTS TO SCRIPT LINES
    # ============================================
    current_script_line = START_LINE
    total_matched = 0
    skipped_lines = []
    consec_fails = 0
    metadata_entries = []

    seg_idx = 0
    used_segments = set()

    while seg_idx < len(segments) and current_script_line <= max_line:
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

            merged_segs = segments[seg_idx:seg_idx + merge_count]
            merged_text = " ".join(s['text'].strip() for s in merged_segs)
            norm_merged = normalize_text(merged_text)

            if len(norm_merged) < 3:
                continue

            # Search window: ±SEG_SEARCH_WINDOW centered on current position
            search_start = max(START_LINE, current_script_line - SEG_SEARCH_WINDOW)
            search_end = min(max_line + 1, current_script_line + SEG_SEARCH_WINDOW)

            for line_num in range(search_start, search_end):
                if line_num not in script_sentences:
                    continue
                target_text = script_sentences[line_num]
                norm_target = normalize_text(target_text)

                if len(norm_target) < 2:
                    continue

                score = difflib.SequenceMatcher(None, norm_merged, norm_target).ratio()

                # Penalty for distance from current position
                skip_count = abs(line_num - current_script_line)
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

            # Mark skipped lines (only forward skips)
            if best_line > current_script_line:
                for skip_line in range(current_script_line, best_line):
                    if skip_line in script_sentences:
                        skip_text = script_sentences[skip_line]
                        skipped_lines.append(f"Script_5_201-541.wav|{skip_line}|SKIPPED|{skip_text}|")

            # Extract audio with 50ms padding
            start_time = seg['start']
            start_ms = max(0, int(start_time * 1000) - 50)
            end_ms = min(len(audio), int(best_end_time * 1000) + 50)
            chunk = audio[start_ms:end_ms]

            # Save audio chunk
            out_filename = f"Script_5_{best_line:04d}.wav"
            out_path = WAVS_DIR / out_filename
            if out_path.exists():
                try:
                    out_path.unlink()
                except:
                    pass
            try:
                chunk.export(str(out_path), format="wav")
            except PermissionError:
                print(f"  [WARN] Cannot write {out_filename} (locked), skipping export")

            metadata_entries.append(f"{out_filename}|{best_line_text}")

            total_matched += 1
            current_script_line = best_line + 1

            # Progress update every 50 matches
            if total_matched % 50 == 0:
                print(f"  ... matched {total_matched} lines (at script line {current_script_line})")

            # Mark consumed segments
            for i in range(best_merge_count):
                used_segments.add(seg_idx + i)
            seg_idx += best_merge_count
        else:
            consec_fails += 1

            if consec_fails >= CONSEC_FAIL_LIMIT and current_script_line <= max_line:
                if current_script_line in script_sentences:
                    skipped_lines.append(f"Script_5_201-541.wav|{current_script_line}|NO_MATCH|{script_sentences[current_script_line]}|{seg_text}")
                current_script_line += 1
                consec_fails = 0

            seg_idx += 1

    # Mark remaining unmatched script lines as skipped
    while current_script_line <= max_line:
        if current_script_line in script_sentences:
            skipped_lines.append(f"Script_5_201-541.wav|{current_script_line}|REMAINING|{script_sentences[current_script_line]}|")
        current_script_line += 1

    # ============================================
    # WRITE OUTPUTS
    # ============================================

    # Append to datasets/script.txt
    if metadata_entries:
        with open(SCRIPT_TXT, 'a', encoding='utf-8') as f:
            for entry in metadata_entries:
                f.write(entry + "\n")
        print(f"\n  Appended {len(metadata_entries)} entries to {SCRIPT_TXT.name}")

    # Write skipped lines log
    with open(SKIPPED_LOG, 'w', encoding='utf-8') as f:
        f.write("AudioFile|ScriptLine|Reason|ScriptText|WhisperText\n")
        for line in skipped_lines:
            f.write(line + "\n")
    print(f"  Wrote {len(skipped_lines)} skipped lines to {SKIPPED_LOG.name}")

    # ============================================
    # FINAL SUMMARY
    # ============================================
    total_lines = len(script_sentences)
    print()
    print("=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"  Script lines available: {total_lines} (lines {START_LINE}-{max_line})")
    print(f"  Successfully matched: {total_matched}")
    print(f"  Skipped: {len(skipped_lines)}")
    print(f"  Match rate: {total_matched / total_lines * 100:.1f}%")
    print()
    print(f"  Output WAVs: {WAVS_DIR}")
    print(f"  Metadata appended to: {SCRIPT_TXT}")
    print(f"  Skipped log: {SKIPPED_LOG}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process Script_5 audio (lines 201+, CUDA required)")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: medium)")
    args = parser.parse_args()

    process_script5(model_size=args.model)
