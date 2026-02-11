# -*- coding: utf-8 -*-
"""
TTS Dataset Pipeline Manager
=============================
End-to-end workflow for Korean TTS dataset preparation.

Pipeline Steps:
  1. DISCOVER  - Find raw audio files and matching scripts
  2. ALIGN     - Transcribe with Whisper and match to script lines
  3. SPLIT     - Extract matched segments as individual WAV files
  4. VALIDATE  - Verify dataset integrity (WAVs <-> metadata)
  5. ORPHANS   - Collect unmatched WAVs -> rawdata/missed audios and script/
  6. REPORT    - Generate timestamped report -> TaskLogs/

Usage:
  python pipeline_manager.py                          # Process all scripts
  python pipeline_manager.py --script 2               # Process Script_2 only
  python pipeline_manager.py --script 2 5             # Process Script_2 and Script_5
  python pipeline_manager.py --reset                  # Clear checkpoints, start fresh
  python pipeline_manager.py --validate-only          # Skip alignment, just validate
  python pipeline_manager.py --collect-orphans        # Validate + move orphan WAVs
  python pipeline_manager.py --model large            # Use larger Whisper model
  python pipeline_manager.py --start-line 201         # Start matching from line 201
"""

import os
import sys
import re
import json
import difflib
import shutil
import warnings
import datetime
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


# ============================================================
# ALIGNMENT CONFIGURATION (proven optimal for Korean TTS)
# ============================================================
SEG_SEARCH_WINDOW = 25      # Search up to N lines forward from current position
MATCH_THRESHOLD = 0.25      # Minimum similarity score to accept a match
SKIP_PENALTY = 0.01         # Penalty per skipped line (distance cost)
CONSEC_FAIL_LIMIT = 5       # Consecutive unmatched segments before advancing
AUDIO_PAD_MS = 50           # Milliseconds of padding on each side of extracted audio


def normalize_text(text):
    """Remove punctuation, keep Korean/English/numbers for comparison."""
    return re.sub(r'[^가-힣a-zA-Z0-9]', '', text)


def load_script(script_path, start_line=1):
    """Load script file with proper Korean encoding.
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


class PipelineManager:
    """Orchestrates the full TTS dataset alignment pipeline."""

    def __init__(self, base_dir, model_size="medium"):
        self.base_dir = Path(base_dir)
        self.model_size = model_size
        self.run_timestamp = datetime.datetime.now()

        # Directory structure
        self.audio_dir = self.base_dir / "rawdata" / "audio"
        self.scripts_dir = self.base_dir / "rawdata" / "Scripts"
        self.output_dir = self.base_dir / "datasets"
        self.wavs_dir = self.output_dir / "wavs"
        self.script_txt = self.output_dir / "script.txt"
        self.missed_dir = self.base_dir / "rawdata" / "missed audios and script"
        self.missed_targets_dir = self.missed_dir / "TargetScripts"
        self.tasklogs_dir = self.base_dir / "TaskLogs"

        # Runtime state
        self.model = None
        self.all_results = {}       # {script_id: result_dict}
        self.all_skipped = []       # List of skipped line log entries

    # ============================================================
    # STEP 1: DISCOVER
    # ============================================================
    def discover(self, script_ids=None):
        """Find all raw audio files grouped by script ID, with matching scripts.
        Returns (audio_groups, script_files) dicts keyed by script_id (int)."""
        pattern = re.compile(r'Script_(\d+)_(\d+)-(\d+)\.wav')
        audio_groups = {}

        if not self.audio_dir.exists():
            print(f"  [ERROR] Audio directory not found: {self.audio_dir}")
            return {}, {}

        for f in sorted(os.listdir(str(self.audio_dir))):
            match = pattern.match(f)
            if match:
                sid = int(match.group(1))
                if script_ids and sid not in script_ids:
                    continue
                start_line = int(match.group(2))
                end_line = int(match.group(3))
                if sid not in audio_groups:
                    audio_groups[sid] = []
                audio_groups[sid].append((start_line, end_line, self.audio_dir / f))

        # Sort each group by start line
        for sid in audio_groups:
            audio_groups[sid].sort(key=lambda x: x[0])

        # Find matching script text files
        script_files = {}
        if self.scripts_dir.exists():
            for script_file in sorted(os.listdir(str(self.scripts_dir))):
                match = re.match(r'Script_(\d+)_A0\.txt', script_file)
                if match:
                    sid = int(match.group(1))
                    if script_ids and sid not in script_ids:
                        continue
                    script_files[sid] = self.scripts_dir / script_file

        return audio_groups, script_files

    # ============================================================
    # STEP 2: LOAD MODEL
    # ============================================================
    def load_model(self):
        """Load Whisper model on CUDA. Exits if no GPU available."""
        if self.model is not None:
            return

        if not torch.cuda.is_available():
            print("\n[FATAL] CUDA is NOT available!")
            print("This pipeline requires a CUDA-capable GPU.")
            sys.exit(1)

        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Loading Whisper {self.model_size} on CUDA...")
        self.model = whisper.load_model(self.model_size, device="cuda")
        print("  Model loaded.")

    # ============================================================
    # CHECKPOINT MANAGEMENT
    # ============================================================
    def _checkpoint_path(self, script_id):
        return self.base_dir / f"checkpoint_pipeline_Script_{script_id}.json"

    def _load_checkpoint(self, script_id):
        path = self._checkpoint_path(script_id)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _save_checkpoint(self, script_id, data):
        path = self._checkpoint_path(script_id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _clear_checkpoint(self, script_id):
        path = self._checkpoint_path(script_id)
        if path.exists():
            path.unlink()

    # ============================================================
    # STEP 3: ALIGN & SPLIT (core algorithm)
    # ============================================================
    def align_script(self, script_id, audio_files, script_path,
                     start_line=1, reset=False):
        """Align and split all audio files for one script.

        Uses forward-only sequential matching:
        - 25-line search window from current position
        - 0.01 skip penalty per line distance
        - Segment merging (1, 2, 3 consecutive Whisper segments)
        - Checkpoint saved after each audio file

        Returns (matched_count, skipped_entries_list, total_script_lines).
        """
        # Load script text
        sentences, enc = load_script(str(script_path), start_line=start_line)
        if not sentences:
            print(f"  [ERROR] Cannot load script: {script_path}")
            return 0, [], 0

        total_sentences = max(sentences.keys())
        print(f"  Script_{script_id}: {len(sentences)} lines (encoding: {enc})")
        print(f"  Audio files: {len(audio_files)}")

        # Checkpoint handling
        checkpoint = None if reset else self._load_checkpoint(script_id)

        if checkpoint:
            resume_file_idx = checkpoint.get('next_file_idx', 0)
            resume_script_line = checkpoint.get('next_script_line', start_line)
            total_matched = checkpoint.get('total_matched', 0)
            print(f"  [RESUME] file #{resume_file_idx}, line {resume_script_line}, "
                  f"matched {total_matched}")
        else:
            resume_file_idx = 0
            resume_script_line = start_line
            total_matched = 0

        # Skipped lines log
        skipped_log_path = self.base_dir / f"skipped_Script_{script_id}.log"
        skipped_entries = []

        if resume_file_idx == 0:
            with open(skipped_log_path, 'w', encoding='utf-8') as f:
                f.write("AudioFile|ScriptLine|Reason|ScriptText|WhisperText\n")

        # Ensure output directory exists
        self.wavs_dir.mkdir(parents=True, exist_ok=True)

        # Load existing metadata filenames to avoid duplicates
        existing_meta = set()
        if self.script_txt.exists():
            with open(self.script_txt, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '|' in line:
                        existing_meta.add(line.split('|')[0])

        current_script_line = resume_script_line
        new_entries = []

        # Process each audio file sequentially
        for file_idx in range(resume_file_idx, len(audio_files)):
            start_ln, end_ln, audio_path = audio_files[file_idx]
            audio_filename = audio_path.name

            print(f"\n  [{file_idx+1}/{len(audio_files)}] {audio_filename}")
            print(f"    Expected lines: {start_ln}-{end_ln}")
            print(f"    Current script line: {current_script_line}")

            # Transcribe with Whisper
            print(f"    Transcribing...", end=" ", flush=True)
            result = self.model.transcribe(
                str(audio_path), language="ko", verbose=False, fp16=True
            )
            segments = result['segments']
            print(f"{len(segments)} segments")

            if not segments:
                print(f"    [WARN] No segments found, skipping file")
                continue

            # Load audio for slicing
            audio = AudioSegment.from_wav(str(audio_path))

            file_matched = 0
            file_skipped = []
            seg_idx = 0
            used_segments = set()
            consec_fails = 0

            while seg_idx < len(segments) and current_script_line <= total_sentences:
                if seg_idx in used_segments:
                    seg_idx += 1
                    continue

                seg = segments[seg_idx]
                seg_text = seg['text'].strip()
                norm_seg = normalize_text(seg_text)

                if len(norm_seg) < 2:
                    seg_idx += 1
                    continue

                # Try merge combinations: 1, 2, or 3 consecutive segments
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

                    # Forward-only search window
                    search_start = current_script_line
                    search_end = min(total_sentences + 1,
                                     current_script_line + SEG_SEARCH_WINDOW)

                    for line_num in range(search_start, search_end):
                        if line_num not in sentences:
                            continue
                        target_text = sentences[line_num]
                        norm_target = normalize_text(target_text)

                        if len(norm_target) < 2:
                            continue

                        score = difflib.SequenceMatcher(
                            None, norm_merged, norm_target
                        ).ratio()

                        # Penalize skipping lines (prefer closer matches)
                        skip_count = line_num - current_script_line
                        adjusted_score = score - (skip_count * SKIP_PENALTY)

                        if adjusted_score > best_score:
                            best_score = adjusted_score
                            best_line = line_num
                            best_line_text = target_text
                            best_merge_count = merge_count
                            best_end_time = merged_segs[-1]['end']

                if best_score >= MATCH_THRESHOLD and best_line is not None:
                    consec_fails = 0

                    # Log skipped lines between current position and match
                    for skip_line in range(current_script_line, best_line):
                        if skip_line in sentences:
                            entry = (f"{audio_filename}|{skip_line}|SKIPPED"
                                     f"|{sentences[skip_line]}|")
                            file_skipped.append(entry)

                    # Extract audio chunk with padding
                    start_ms = max(0, int(seg['start'] * 1000) - AUDIO_PAD_MS)
                    end_ms = min(len(audio),
                                 int(best_end_time * 1000) + AUDIO_PAD_MS)
                    chunk = audio[start_ms:end_ms]

                    # Save WAV file
                    out_filename = f"Script_{script_id}_{best_line:04d}.wav"
                    out_path = self.wavs_dir / out_filename

                    if out_path.exists():
                        try:
                            out_path.unlink()
                        except Exception:
                            pass
                    try:
                        chunk.export(str(out_path), format="wav")
                    except PermissionError:
                        print(f"    [WARN] Cannot write {out_filename}")

                    # Record metadata entry (deduplicate)
                    if out_filename not in existing_meta:
                        new_entries.append(f"{out_filename}|{best_line_text}")
                        existing_meta.add(out_filename)

                    file_matched += 1
                    total_matched += 1
                    current_script_line = best_line + 1

                    # Mark consumed segments
                    for i in range(best_merge_count):
                        used_segments.add(seg_idx + i)
                    seg_idx += best_merge_count
                else:
                    consec_fails += 1

                    if (consec_fails >= CONSEC_FAIL_LIMIT
                            and current_script_line <= total_sentences):
                        if current_script_line in sentences:
                            entry = (f"{audio_filename}|{current_script_line}"
                                     f"|NO_MATCH|{sentences[current_script_line]}"
                                     f"|{seg_text}")
                            file_skipped.append(entry)
                        current_script_line += 1
                        consec_fails = 0

                    seg_idx += 1

            # Write skipped lines to log
            if file_skipped:
                with open(skipped_log_path, 'a', encoding='utf-8') as f:
                    for line in file_skipped:
                        f.write(line + "\n")
            skipped_entries.extend(file_skipped)

            print(f"    Matched: {file_matched}, Skipped: {len(file_skipped)}")
            print(f"    Total matched so far: {total_matched}")

            # Save checkpoint after each file
            self._save_checkpoint(script_id, {
                'next_file_idx': file_idx + 1,
                'next_script_line': current_script_line,
                'total_matched': total_matched,
                'last_file': audio_filename
            })

        # Append new entries to script.txt
        if new_entries:
            with open(self.script_txt, 'a', encoding='utf-8') as f:
                for entry in new_entries:
                    f.write(entry + "\n")
            print(f"\n  Appended {len(new_entries)} new entries to script.txt")

        # Clear checkpoint on completion
        self._clear_checkpoint(script_id)

        match_rate = (total_matched / len(sentences) * 100
                      if sentences else 0)
        print(f"  Script_{script_id} done: {total_matched}/{len(sentences)} "
              f"({match_rate:.1f}%)")

        return total_matched, skipped_entries, len(sentences)

    # ============================================================
    # STEP 4: VALIDATE
    # ============================================================
    def validate(self):
        """Check dataset integrity: WAV files <-> metadata entries in script.txt.
        Returns a validation result dict."""
        print("\n" + "=" * 60)
        print("VALIDATION")
        print("=" * 60)

        # Load all metadata entries from script.txt
        meta_files = {}
        if self.script_txt.exists():
            with open(self.script_txt, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '|' in line:
                        parts = line.split('|', 1)
                        meta_files[parts[0]] = parts[1]

        # Get actual WAV files
        wav_files = set()
        if self.wavs_dir.exists():
            wav_files = {f for f in os.listdir(str(self.wavs_dir))
                         if f.endswith('.wav')}

        meta_set = set(meta_files.keys())
        missing_wav = sorted(meta_set - wav_files)
        orphan_wav = sorted(wav_files - meta_set)

        # Stats by script ID
        script_stats = {}
        for fname in meta_set | wav_files:
            match = re.match(r'Script_(\d+)_', fname)
            if match:
                sid = int(match.group(1))
                if sid not in script_stats:
                    script_stats[sid] = {
                        'meta': 0, 'wav': 0, 'orphan': 0, 'missing': 0
                    }
                if fname in meta_set:
                    script_stats[sid]['meta'] += 1
                if fname in wav_files:
                    script_stats[sid]['wav'] += 1
                if fname in set(orphan_wav):
                    script_stats[sid]['orphan'] += 1
                if fname in set(missing_wav):
                    script_stats[sid]['missing'] += 1

        print(f"\n  Metadata entries:  {len(meta_set)}")
        print(f"  WAV files:         {len(wav_files)}")
        print(f"  Missing WAVs:      {len(missing_wav)}")
        print(f"  Orphan WAVs:       {len(orphan_wav)}")

        print(f"\n  By Script:")
        for sid in sorted(script_stats.keys()):
            s = script_stats[sid]
            status = ""
            if s['orphan'] > 0:
                status += f" ({s['orphan']} orphans)"
            if s['missing'] > 0:
                status += f" ({s['missing']} missing)"
            print(f"    Script_{sid}: {s['meta']} entries, "
                  f"{s['wav']} wavs{status}")

        integrity = ("PASS" if not missing_wav and not orphan_wav
                     else "NEEDS ATTENTION")
        print(f"\n  Integrity: {integrity}")

        return {
            'meta_count': len(meta_set),
            'wav_count': len(wav_files),
            'missing_wav': missing_wav,
            'orphan_wav': orphan_wav,
            'script_stats': script_stats,
            'integrity': integrity
        }

    # ============================================================
    # STEP 5: COLLECT ORPHANS
    # ============================================================
    def collect_orphans(self, orphan_wavs):
        """Move orphan WAVs from datasets/wavs/ to rawdata/missed audios and script/.
        Returns number of files moved."""
        if not orphan_wavs:
            print("\n  No orphan WAVs to collect.")
            return 0

        print(f"\n{'='*60}")
        print(f"COLLECTING ORPHANS -> {self.missed_dir.name}/")
        print(f"{'='*60}")

        self.missed_dir.mkdir(parents=True, exist_ok=True)

        moved = 0
        for wav_name in sorted(orphan_wavs):
            src = self.wavs_dir / wav_name
            dst = self.missed_dir / wav_name
            if src.exists():
                try:
                    shutil.move(str(src), str(dst))
                    moved += 1
                except Exception as e:
                    print(f"  [WARN] Cannot move {wav_name}: {e}")

        print(f"  Moved {moved}/{len(orphan_wavs)} orphan WAVs")
        return moved

    def write_missed_lines(self, skipped_entries):
        """Write missed/skipped script line numbers to a report file
        in rawdata/missed audios and script/ for future recovery."""
        if not skipped_entries:
            return

        self.missed_dir.mkdir(parents=True, exist_ok=True)

        # Group by script ID
        by_script = {}
        for entry in skipped_entries:
            parts = entry.split('|')
            if len(parts) >= 4:
                match = re.match(r'Script_(\d+)_', parts[0])
                if match:
                    sid = int(match.group(1))
                    line_num = int(parts[1])
                    reason = parts[2]
                    text = parts[3]
                    if sid not in by_script:
                        by_script[sid] = []
                    by_script[sid].append((line_num, reason, text))

        for sid, lines in sorted(by_script.items()):
            report_path = (self.missed_dir /
                           f"missed_lines_Script_{sid}.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# Missed lines from Script_{sid} alignment\n")
                f.write(f"# Generated: {self.run_timestamp:%Y-%m-%d %H:%M}\n")
                f.write(f"# Format: LineNumber|Reason|Text\n")
                f.write(f"# Total: {len(lines)} lines\n\n")
                for line_num, reason, text in sorted(lines, key=lambda x: x[0]):
                    f.write(f"{line_num}|{reason}|{text}\n")
            print(f"  Script_{sid}: {len(lines)} missed lines -> "
                  f"{report_path.name}")

    # ============================================================
    # STEP 6: REPORT
    # ============================================================
    def generate_report(self, validation):
        """Generate a timestamped pipeline report to TaskLogs/."""
        self.tasklogs_dir.mkdir(parents=True, exist_ok=True)

        ts = self.run_timestamp.strftime("%Y%m%d_%H%M%S")
        report_path = self.tasklogs_dir / f"{ts}_pipeline_report.txt"

        lines = []
        lines.append("=" * 80)
        lines.append("TTS DATASET PIPELINE REPORT")
        lines.append("=" * 80)
        lines.append(f"Date: {self.run_timestamp:%Y-%m-%d %H:%M:%S}")
        lines.append(f"Tool: Whisper ASR ({self.model_size} model, CUDA"
                      f" - {torch.cuda.get_device_name(0)})")
        lines.append(f"Config: window={SEG_SEARCH_WINDOW}, "
                      f"threshold={MATCH_THRESHOLD}, "
                      f"penalty={SKIP_PENALTY}")
        lines.append("")

        # Alignment results
        if self.all_results:
            lines.append("=" * 80)
            lines.append("ALIGNMENT RESULTS")
            lines.append("=" * 80)

            total_matched = 0
            total_skipped = 0
            total_lines = 0

            for sid in sorted(self.all_results.keys()):
                r = self.all_results[sid]
                total_matched += r['matched']
                total_skipped += r['skipped']
                total_lines += r['total_lines']
                rate = (r['matched'] / r['total_lines'] * 100
                        if r['total_lines'] > 0 else 0)
                lines.append(f"  Script_{sid}:")
                lines.append(f"    Script lines: {r['total_lines']}")
                lines.append(f"    Matched:      {r['matched']}  ({rate:.1f}%)")
                lines.append(f"    Skipped:      {r['skipped']}")
                lines.append(f"    Audio files:  {r['audio_files']}")
                lines.append("")

            overall = (total_matched / total_lines * 100
                       if total_lines > 0 else 0)
            lines.append(f"  TOTAL: {total_matched}/{total_lines} "
                         f"matched ({overall:.1f}%)")
            lines.append(f"         {total_skipped} lines skipped")
            lines.append("")

        # Validation results
        if validation:
            lines.append("=" * 80)
            lines.append("DATASET VALIDATION")
            lines.append("=" * 80)
            lines.append(f"  Metadata entries:  {validation['meta_count']}")
            lines.append(f"  WAV files:         {validation['wav_count']}")
            lines.append(f"  Missing WAVs:      {len(validation['missing_wav'])}")
            lines.append(f"  Orphan WAVs:       {len(validation['orphan_wav'])}")
            lines.append(f"  Integrity:         {validation['integrity']}")

            if validation['orphan_wav']:
                lines.append(f"\n  Orphan WAVs (moved to missed audios):")
                for wav in validation['orphan_wav'][:20]:
                    lines.append(f"    - {wav}")
                if len(validation['orphan_wav']) > 20:
                    lines.append(f"    ... and {len(validation['orphan_wav'])-20} more")

            lines.append("")

        lines.append("=" * 80)

        report_text = "\n".join(lines)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\n  Report: {report_path}")
        return report_path

    # ============================================================
    # PIPELINE ORCHESTRATOR
    # ============================================================
    def run(self, script_ids=None, start_line=1, reset=False,
            validate_only=False, collect_orphans_only=False):
        """Run the full pipeline end-to-end."""
        print("=" * 60)
        print("TTS Dataset Pipeline Manager")
        print("=" * 60)
        print(f"  Base: {self.base_dir}")
        print(f"  Time: {self.run_timestamp:%Y-%m-%d %H:%M:%S}")

        # --validate-only / --collect-orphans: skip alignment
        if validate_only or collect_orphans_only:
            validation = self.validate()
            if collect_orphans_only and validation['orphan_wav']:
                self.collect_orphans(validation['orphan_wav'])
            self.generate_report(validation)
            return

        # Step 1: Discover
        print(f"\n[Step 1/6] Discovering audio files and scripts...")
        audio_groups, script_files = self.discover(script_ids)

        if not audio_groups:
            print("  No audio files found to process.")
            if script_ids:
                print(f"  (Filtered for script IDs: {script_ids})")
            return

        for sid in sorted(audio_groups.keys()):
            files = audio_groups[sid]
            has_script = "OK" if sid in script_files else "MISSING"
            print(f"  Script_{sid}: {len(files)} audio files "
                  f"(script: {has_script})")
            for start, end, path in files:
                print(f"    {path.name}  (lines {start}-{end})")

        # Step 2: Load model
        print(f"\n[Step 2/6] Loading Whisper model...")
        self.load_model()

        # Step 3: Align & Split
        print(f"\n[Step 3/6] Aligning and splitting audio...")

        for sid in sorted(audio_groups.keys()):
            if sid not in script_files:
                print(f"\n  [SKIP] Script_{sid}: no script file found")
                continue

            print(f"\n  {'='*50}")
            print(f"  Processing Script_{sid}")
            print(f"  {'='*50}")

            # Determine start line for this script
            sl = start_line if start_line > 1 else 1

            matched, skipped, total = self.align_script(
                sid, audio_groups[sid], script_files[sid],
                start_line=sl, reset=reset
            )

            self.all_results[sid] = {
                'matched': matched,
                'skipped': len(skipped),
                'total_lines': total,
                'audio_files': len(audio_groups[sid])
            }
            self.all_skipped.extend(skipped)

        # Step 4: Validate
        print(f"\n[Step 4/6] Validating dataset...")
        validation = self.validate()

        # Step 5: Collect orphans
        print(f"\n[Step 5/6] Collecting orphan WAVs...")
        if validation['orphan_wav']:
            self.collect_orphans(validation['orphan_wav'])
            self.write_missed_lines(self.all_skipped)
        else:
            print("  No orphans found. Dataset is clean.")

        # Step 6: Report
        print(f"\n[Step 6/6] Generating report...")
        self.generate_report(validation)

        # Final summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        for sid in sorted(self.all_results.keys()):
            r = self.all_results[sid]
            rate = (r['matched'] / r['total_lines'] * 100
                    if r['total_lines'] > 0 else 0)
            print(f"  Script_{sid}: {r['matched']}/{r['total_lines']} "
                  f"({rate:.1f}%)")

        total_m = sum(r['matched'] for r in self.all_results.values())
        total_l = sum(r['total_lines'] for r in self.all_results.values())
        if total_l > 0:
            print(f"\n  Overall: {total_m}/{total_l} "
                  f"({total_m/total_l*100:.1f}%)")
        print(f"  Orphans collected: {len(validation.get('orphan_wav', []))}")
        print(f"  Skipped lines report: "
              f"{'written' if self.all_skipped else 'none'}")
        print("=" * 60)


# ============================================================
# CLI ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TTS Dataset Pipeline Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_manager.py                    Process all scripts
  python pipeline_manager.py --script 2         Process Script_2 only
  python pipeline_manager.py --script 2 5       Process Script_2 and 5
  python pipeline_manager.py --reset            Clear checkpoints, fresh start
  python pipeline_manager.py --validate-only    Just validate dataset
  python pipeline_manager.py --collect-orphans  Validate + move orphan WAVs
  python pipeline_manager.py --model large      Use larger Whisper model
  python pipeline_manager.py --start-line 201   Start matching from line 201
        """
    )
    parser.add_argument(
        "--script", nargs="+", type=int, default=None,
        help="Script IDs to process (default: all)")
    parser.add_argument(
        "--model", default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: medium)")
    parser.add_argument(
        "--start-line", type=int, default=1,
        help="Start matching from this script line (default: 1)")
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear checkpoints and start fresh")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip alignment, just validate dataset integrity")
    parser.add_argument(
        "--collect-orphans", action="store_true",
        help="Validate and collect orphan WAVs to missed audios folder")

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    manager = PipelineManager(base_dir, model_size=args.model)
    manager.run(
        script_ids=args.script,
        start_line=args.start_line,
        reset=args.reset,
        validate_only=args.validate_only,
        collect_orphans_only=args.collect_orphans
    )
