# -*- coding: utf-8 -*-
"""
Missed Audio Alignment Tool
- Transcribes individual WAV files and matches to script lines
- WAV filenames encode the expected line number (e.g., Script_1_0124.wav = line 124)
- Uses Whisper medium model on CUDA
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
except ImportError:
    print("Installing requirements...")
    os.system("pip install openai-whisper static-ffmpeg torch")
    import static_ffmpeg
    static_ffmpeg.add_paths()
    import whisper


def normalize_text(text):
    """Remove punctuation, keep Korean/English/numbers for comparison."""
    text = re.sub(r'[^가-힣a-zA-Z0-9]', '', text)
    return text


def load_script(script_path):
    """Load script file with proper Korean encoding."""
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
                    sentences[line_num] = line
                return sentences, enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return {}, None


def process_missed(model_size="medium"):
    """Process missed audio files by transcribing and matching to script lines."""

    BASE_DIR = Path(__file__).parent.parent
    MISSED_DIR = BASE_DIR / "rawdata" / "missed audios and script"
    SCRIPTS_DIR = MISSED_DIR / "TargetScripts"
    OUTPUT_DIR = BASE_DIR / "datasets"
    WAVS_DIR = OUTPUT_DIR / "wavs"
    METADATA_PATH = OUTPUT_DIR / "metadata.txt"

    SEARCH_WINDOW = 10  # Search +/- N lines around expected line number
    MATCH_THRESHOLD = 0.20  # Lower threshold since these are individual utterances

    print("=" * 60)
    print("Missed Audio Alignment Tool (CUDA)")
    print("=" * 60)

    # CUDA check
    if not torch.cuda.is_available():
        print("\n[FATAL ERROR] CUDA is NOT available!")
        sys.exit(1)

    device = "cuda"
    print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[Model] {model_size}")

    # Collect WAV files
    wav_files = sorted(f for f in os.listdir(str(MISSED_DIR)) if f.endswith('.wav'))
    print(f"\n[WAV Files] {len(wav_files)} files to process")

    if not wav_files:
        print("No WAV files found!")
        return

    # Group by script
    script_groups = {}
    pattern = re.compile(r'(Script_\d+)_(\d+)\.wav')
    for f in wav_files:
        match = pattern.match(f)
        if match:
            script_name = match.group(1)
            line_num = int(match.group(2))
            if script_name not in script_groups:
                script_groups[script_name] = []
            script_groups[script_name].append((line_num, f))

    for script_name in sorted(script_groups.keys()):
        print(f"  {script_name}: {len(script_groups[script_name])} files")

    # Load all scripts
    scripts = {}
    for script_file in sorted(os.listdir(str(SCRIPTS_DIR))):
        if not script_file.endswith('.txt'):
            continue
        script_name = script_file.replace('_A0.txt', '')
        script_path = SCRIPTS_DIR / script_file
        sentences, enc = load_script(str(script_path))
        if sentences:
            scripts[script_name] = sentences
            print(f"  [OK] {script_name}: {len(sentences)} lines (encoding: {enc})")

    # Load Whisper model
    print(f"\n[Loading Whisper {model_size} on CUDA...]")
    model = whisper.load_model(model_size, device=device)
    print("  Model loaded\n")

    # Read existing metadata to avoid duplicates
    existing_meta = set()
    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_meta.add(line.split('|')[0])
    print(f"[Existing metadata entries] {len(existing_meta)}")

    # Process each script group
    total_matched = 0
    total_failed = 0
    new_entries = []
    failed_entries = []

    WAVS_DIR.mkdir(parents=True, exist_ok=True)

    for script_name in sorted(script_groups.keys()):
        files = sorted(script_groups[script_name])
        sentences = scripts.get(script_name, {})

        if not sentences:
            print(f"\n[SKIP] {script_name}: No script loaded")
            continue

        print(f"\n{'='*50}")
        print(f"Processing {script_name} ({len(files)} files)")
        print(f"{'='*50}")

        for expected_line, wav_filename in files:
            wav_path = MISSED_DIR / wav_filename
            print(f"  [{wav_filename}] expected line {expected_line}...", end=" ")

            # Transcribe
            try:
                result = model.transcribe(
                    str(wav_path),
                    language="ko",
                    verbose=False,
                    fp16=True
                )
                whisper_text = result['text'].strip()
                norm_whisper = normalize_text(whisper_text)
            except Exception as e:
                print(f"TRANSCRIBE ERROR: {e}")
                failed_entries.append((wav_filename, expected_line, "TRANSCRIBE_ERROR", str(e)))
                total_failed += 1
                continue

            if len(norm_whisper) < 2:
                print(f"TOO SHORT: '{whisper_text}'")
                failed_entries.append((wav_filename, expected_line, "TOO_SHORT", whisper_text))
                total_failed += 1
                continue

            # Search for best match around expected line
            best_score = 0
            best_line = None
            best_text = ""

            search_start = max(1, expected_line - SEARCH_WINDOW)
            search_end = min(max(sentences.keys()), expected_line + SEARCH_WINDOW)

            for line_num in range(search_start, search_end + 1):
                if line_num not in sentences:
                    continue
                target_text = sentences[line_num]
                norm_target = normalize_text(target_text)

                if len(norm_target) < 2:
                    continue

                score = difflib.SequenceMatcher(None, norm_whisper, norm_target).ratio()

                # Slight bonus for exact expected line
                if line_num == expected_line:
                    score += 0.05

                if score > best_score:
                    best_score = score
                    best_line = line_num
                    best_text = target_text

            if best_score >= MATCH_THRESHOLD and best_line is not None:
                # Match found
                if wav_filename in existing_meta:
                    print(f"ALREADY IN META (line {best_line}, score {best_score:.2f})")
                else:
                    new_entries.append((wav_filename, best_text))
                    print(f"MATCH line {best_line} (score {best_score:.2f})")

                    # Copy WAV to datasets/wavs/ if not already there
                    dest_path = WAVS_DIR / wav_filename
                    if not dest_path.exists():
                        import shutil
                        try:
                            shutil.copy2(str(wav_path), str(dest_path))
                        except Exception as e:
                            print(f"    [WARN] Copy failed: {e}")

                total_matched += 1
            else:
                print(f"NO MATCH (best score {best_score:.2f}, whisper: '{whisper_text[:40]}...')")
                failed_entries.append((wav_filename, expected_line, "NO_MATCH", whisper_text))
                total_failed += 1

    # Append new entries to metadata
    if new_entries:
        print(f"\n[Appending {len(new_entries)} new entries to metadata.txt]")
        with open(METADATA_PATH, 'a', encoding='utf-8') as f:
            for wav_filename, text in new_entries:
                f.write(f"{wav_filename}|{text}\n")

    # Summary
    print(f"\n{'='*60}")
    print("COMPLETED")
    print(f"{'='*60}")
    print(f"  Total files processed: {len(wav_files)}")
    print(f"  Matched: {total_matched}")
    print(f"  Failed: {total_failed}")
    print(f"  New metadata entries: {len(new_entries)}")

    if failed_entries:
        print(f"\n  Failed files:")
        for wav_filename, expected_line, reason, detail in failed_entries:
            print(f"    {wav_filename} (line {expected_line}): {reason} - {detail[:60]}")

    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process missed audio files (CUDA required)")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: medium)")
    args = parser.parse_args()
    process_missed(model_size=args.model)
