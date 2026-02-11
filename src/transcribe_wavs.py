# -*- coding: utf-8 -*-
"""
Transcribe all segmented WAV files in datasets/wavs/ using Whisper.
Produces datasets/whisper_transcribed.txt with what Whisper actually hears.

- Does NOT modify any existing files
- Checkpoint/resume: saves progress every 50 files
- Output format: filename.wav|transcribed_text (same as script.txt)

Usage:
  python transcribe_wavs.py                  # Resume or start fresh
  python transcribe_wavs.py --reset          # Clear checkpoint, start over
  python transcribe_wavs.py --model medium   # Choose Whisper model size
"""

import os
import sys
import json
import time
import warnings
import torch
from pathlib import Path

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

CHECKPOINT_INTERVAL = 50  # Save checkpoint every N files


def transcribe_all(model_size="medium", reset=False):
    BASE_DIR = Path(__file__).parent.parent
    WAVS_DIR = BASE_DIR / "datasets" / "wavs"
    OUTPUT_PATH = BASE_DIR / "datasets" / "whisper_transcribed.txt"
    CHECKPOINT_PATH = BASE_DIR / "checkpoint_transcribe.json"

    print("=" * 60)
    print("WAV Transcription Tool (Whisper → whisper_transcribed.txt)")
    print("=" * 60)

    # CUDA check
    if not torch.cuda.is_available():
        print("\n[FATAL] CUDA is NOT available!")
        sys.exit(1)

    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Model: {model_size}")

    # Collect all WAV files (sorted for deterministic order)
    all_wavs = sorted(f for f in os.listdir(str(WAVS_DIR)) if f.endswith('.wav'))
    total_files = len(all_wavs)
    print(f"  WAV files: {total_files}")

    if total_files == 0:
        print("  No WAV files found.")
        return

    # Checkpoint handling
    if reset and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("  [Checkpoint cleared]")

    start_idx = 0
    if not reset:
        # On resume, count actual output lines to determine true progress
        # (handles crash between checkpoint saves)
        if OUTPUT_PATH.exists():
            with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                lines_written = sum(1 for line in f if line.strip())
            if lines_written > 0:
                start_idx = lines_written
                print(f"  [RESUME] Output has {lines_written} lines, "
                      f"resuming from file #{start_idx}")
        elif CHECKPOINT_PATH.exists():
            try:
                with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
                    ckpt = json.load(f)
                start_idx = ckpt.get('next_idx', 0)
                print(f"  [RESUME] From checkpoint #{start_idx}")
            except Exception:
                start_idx = 0

    if start_idx >= total_files:
        print("  All files already transcribed. Use --reset to redo.")
        return

    # Load Whisper model
    print(f"\n  Loading Whisper {model_size} on CUDA...")
    model = whisper.load_model(model_size, device="cuda")
    print("  Model loaded.\n")

    # Open output file in append mode (resume-safe)
    # On fresh start, write header comment
    if start_idx == 0:
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            pass  # Create/clear the file

    errors = []

    for idx in range(start_idx, total_files):
        wav_name = all_wavs[idx]
        wav_path = WAVS_DIR / wav_name

        # Progress display
        pct = (idx + 1) / total_files * 100
        print(f"  [{idx+1}/{total_files}] ({pct:.1f}%) {wav_name}", end=" ", flush=True)

        try:
            result = model.transcribe(
                str(wav_path),
                language="ko",
                verbose=False,
                fp16=True
            )
            text = result['text'].strip()
            print(f"OK ({len(text)} chars)")
        except Exception as e:
            text = f"[ERROR: {e}]"
            errors.append((wav_name, str(e)))
            print(f"ERROR: {e}")

        # Append to output file with retry (exFAT can intermittently lock)
        for attempt in range(5):
            try:
                with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
                    f.write(f"{wav_name}|{text}\n")
                break
            except PermissionError:
                if attempt < 4:
                    time.sleep(1)
                else:
                    print(f"  [WARN] Cannot write after 5 retries, saving checkpoint")
                    with open(CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
                        json.dump({'next_idx': idx}, f)
                    raise

        # Save checkpoint periodically
        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            for attempt in range(3):
                try:
                    with open(CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
                        json.dump({'next_idx': idx + 1}, f)
                    break
                except PermissionError:
                    time.sleep(1)
            print(f"  --- checkpoint saved at {idx+1}/{total_files} ---")

    # Clean up checkpoint on completion
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(f"\n{'='*60}")
    print("COMPLETED")
    print(f"{'='*60}")
    print(f"  Total transcribed: {total_files}")
    print(f"  Errors: {len(errors)}")
    print(f"  Output: {OUTPUT_PATH}")

    if errors:
        print(f"\n  Error files:")
        for wav_name, err in errors[:20]:
            print(f"    {wav_name}: {err}")
        if len(errors) > 20:
            print(f"    ... and {len(errors)-20} more")

    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Transcribe all WAVs in datasets/wavs/ with Whisper")
    parser.add_argument("--model", default="medium",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--reset", action="store_true",
                        help="Clear checkpoint and start fresh")
    args = parser.parse_args()

    transcribe_all(model_size=args.model, reset=args.reset)
