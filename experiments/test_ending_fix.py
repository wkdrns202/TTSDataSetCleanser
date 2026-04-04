"""
Test the formal ending fix by reprocessing Script_2_1-162.wav.
Outputs to experiments/ending_fix_test/ — does NOT touch datasets/.

After processing, runs the truncation detector on the new WAVs
to compare with the original results.
"""
import os
import sys
import json
import re
import unicodedata

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

BASE = os.path.join(os.path.dirname(__file__), '..')
EXPERIMENT_DIR = os.path.join(BASE, 'experiments', 'ending_fix_test')
EXPERIMENT_WAV_DIR = os.path.join(EXPERIMENT_DIR, 'wavs')
EXPERIMENT_SCRIPT = os.path.join(EXPERIMENT_DIR, 'script.txt')

# Monkey-patch the output paths in align_and_split before importing
import align_and_split as aas

# Override output paths to experiment directory
aas.OUTPUT_WAV_DIR = EXPERIMENT_WAV_DIR
aas.METADATA_PATH = EXPERIMENT_SCRIPT
aas.CHECKPOINT_PATH = os.path.join(EXPERIMENT_DIR, 'checkpoint.json')

os.makedirs(EXPERIMENT_WAV_DIR, exist_ok=True)
os.makedirs(aas.LOG_DIR, exist_ok=True)

# Now run just Script_2 with only the first audio file (lines 1-162)
import whisper
from pydub import AudioSegment

def run_test():
    print("=" * 60)
    print("Testing formal ending fix on Script_2_1-162.wav")
    print("=" * 60)

    # Load model
    device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    print(f"\nLoading Whisper {aas.MODEL_SIZE} on {device}...")
    model = whisper.load_model(aas.MODEL_SIZE, device=device)

    # Load script
    script_path = os.path.join(aas.SCRIPT_DIR, "Script_2_A0.txt")
    all_sentences, enc = aas.load_script(script_path)
    print(f"Script loaded: {len(all_sentences)} sentences ({enc})")

    # Process only the first audio file
    audio_file = os.path.join(aas.RAW_AUDIO_DIR, "Script_2_1-176.wav")
    if not os.path.exists(audio_file):
        # Try alternate name
        audio_file = os.path.join(aas.RAW_AUDIO_DIR, "Script_2_1-162.wav")

    if not os.path.exists(audio_file):
        print(f"ERROR: Audio file not found. Looking for Script_2_1-*.wav...")
        import glob
        files = glob.glob(os.path.join(aas.RAW_AUDIO_DIR, "Script_2_*.wav"))
        print(f"Available: {[os.path.basename(f) for f in sorted(files)]}")
        # Use the first one
        if files:
            audio_file = sorted(files)[0]
            print(f"Using: {audio_file}")
        else:
            return

    print(f"\nProcessing: {os.path.basename(audio_file)}")

    # Transcribe with Whisper
    print("Transcribing (this may take a few minutes)...")
    result = model.transcribe(
        audio_file, language=aas.LANGUAGE, verbose=False,
        fp16=(device == "cuda"), word_timestamps=True
    )
    segments = result.get('segments', [])
    print(f"Got {len(segments)} Whisper segments")

    # Parse filename for line range
    script_no, start_line, end_line = aas.parse_audio_filename(
        os.path.basename(audio_file))
    print(f"Script {script_no}, lines {start_line}-{end_line}")

    # Load audio with pydub
    audio = AudioSegment.from_wav(audio_file)
    total_sentences = max(all_sentences.keys())

    # Run alignment (same logic as main process_audio_files)
    import difflib
    seg_idx = 0
    current_script_line = start_line if start_line else 1
    consec_fails = 0
    matched = 0
    used_segments = set()
    metadata_entries = []
    formal_ending_extensions = 0

    while seg_idx < len(segments) and current_script_line <= min(end_line or total_sentences, total_sentences):
        if seg_idx in used_segments:
            seg_idx += 1
            continue

        seg = segments[seg_idx]
        seg_text = seg['text'].strip()
        norm_seg = aas.normalize_text(seg_text)

        if len(norm_seg) < 2:
            seg_idx += 1
            continue

        # Try merges
        best_score = 0
        best_line = None
        best_line_text = ""
        best_merge_count = 1
        best_end_time = seg['end']

        for merge_count in range(1, aas.MAX_MERGE + 1):
            if seg_idx + merge_count > len(segments):
                break
            merged_segs = segments[seg_idx:seg_idx + merge_count]
            merged_text = " ".join(s['text'].strip() for s in merged_segs)
            norm_merged = aas.normalize_text(merged_text)
            if len(norm_merged) < 3:
                continue

            search_end = min(total_sentences + 1,
                           current_script_line + aas.SEG_SEARCH_WINDOW)
            for line_num in range(current_script_line, search_end):
                if line_num not in all_sentences:
                    continue
                target_text = all_sentences[line_num]
                norm_target = aas.normalize_text(target_text)
                if len(norm_target) < 2:
                    continue
                score = difflib.SequenceMatcher(
                    None, norm_merged, norm_target).ratio()
                skip_count = line_num - current_script_line
                adjusted_score = score - (skip_count * aas.SKIP_PENALTY)
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_line = line_num
                    best_line_text = target_text
                    best_merge_count = merge_count
                    best_end_time = merged_segs[-1]['end']

        if best_score >= aas.MATCH_THRESHOLD and best_line is not None:
            # Match confirmation
            confirmed = True
            if best_score < 0.80:
                confirm_seg_idx = seg_idx + best_merge_count
                confirm_line = best_line + 1
                if (confirm_seg_idx < len(segments) and confirm_line in all_sentences):
                    confirm_seg_text = aas.normalize_text(segments[confirm_seg_idx]['text'].strip())
                    confirm_line_text = aas.normalize_text(all_sentences[confirm_line])
                    if len(confirm_seg_text) >= 3 and len(confirm_line_text) >= 2:
                        confirm_score = difflib.SequenceMatcher(
                            None, confirm_seg_text, confirm_line_text).ratio()
                        if confirm_score < 0.25:
                            confirmed = False

            if not confirmed:
                consec_fails += 1
                seg_idx += 1
                continue

            # Word-level boundary refinement (WITH formal ending protection)
            raw_start_ms = int(seg['start'] * 1000)
            raw_end_ms = int(best_end_time * 1000)

            merged_segs = segments[seg_idx:seg_idx + best_merge_count]
            refined = aas.refine_boundaries_with_words(
                merged_segs, best_line_text, aas.normalize_text)
            if refined is not None:
                raw_start_ms = int(refined[0] * 1000)
                raw_end_ms = int(refined[1] * 1000)

            # Right padding with formal ending protection
            has_formal_ending = aas._gt_ends_with_formal(
                aas.normalize_text(best_line_text)) is not None

            next_seg_idx = seg_idx + best_merge_count
            if next_seg_idx < len(segments):
                next_start_ms = int(segments[next_seg_idx]['start'] * 1000)
                right_gap = next_start_ms - raw_end_ms

                if has_formal_ending:
                    # Extend aggressively past next segment boundary
                    # The ending syllables belong to THIS sentence
                    right_pad = 500
                elif right_gap < aas.MIN_GAP_FOR_PAD_MS:
                    right_pad = 0
                else:
                    safe_limit = max(0, right_gap - 20)
                    right_pad = min(aas.TAIL_EXTEND_MAX_MS, safe_limit)
            else:
                right_pad = aas.TAIL_EXTEND_MAX_MS

            if has_formal_ending and right_pad > 0:
                formal_ending_extensions += 1

            # Left padding
            if seg_idx > 0:
                prev_end_ms = int(segments[seg_idx - 1]['end'] * 1000)
                left_gap = raw_start_ms - prev_end_ms
                left_pad = aas.AUDIO_PAD_MS if left_gap >= aas.MIN_GAP_FOR_PAD_MS else 0
            else:
                left_pad = aas.AUDIO_PAD_MS

            start_ms = max(0, raw_start_ms - left_pad)
            end_ms = min(len(audio), raw_end_ms + right_pad)
            chunk = audio[start_ms:end_ms]
            chunk = aas.apply_fade(chunk)

            if chunk.channels > 1:
                chunk = chunk.set_channels(1)

            out_filename = f"Script_{script_no}_{best_line:04d}.wav"
            out_path = os.path.join(EXPERIMENT_WAV_DIR, out_filename)

            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except:
                    pass
            chunk.export(out_path, format="wav")

            metadata_entries.append(f"{out_filename}|{best_line_text}")
            matched += 1
            current_script_line = best_line + 1
            for i in range(best_merge_count):
                used_segments.add(seg_idx + i)
            seg_idx += best_merge_count
            consec_fails = 0
        else:
            consec_fails += 1
            if consec_fails >= aas.CONSEC_FAIL_LIMIT:
                current_script_line += 1
                consec_fails = 0
            seg_idx += 1

    # Save metadata
    with open(EXPERIMENT_SCRIPT, 'w', encoding='utf-8') as f:
        for entry in metadata_entries:
            f.write(entry + '\n')

    print(f"\nMatched: {matched}")
    print(f"Formal ending extensions applied: {formal_ending_extensions}")
    print(f"Output: {EXPERIMENT_WAV_DIR}")
    print(f"Metadata: {EXPERIMENT_SCRIPT}")

    return matched


def verify_results():
    """Run truncation detection on experiment output."""
    print("\n" + "=" * 60)
    print("Verifying results with truncation detector (NO GT-prompting)")
    print("=" * 60)

    import torch
    import whisper as w
    import soundfile as sf
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = w.load_model("medium", device=device)

    FORMAL_ENDINGS = [
        "것이었습니다", "것입니다",
        "었습니다", "였습니다", "겠습니다",
        "습니다", "습니까", "십시오",
        "합시다", "으세요", "하세요",
    ]

    def norm(t):
        t = unicodedata.normalize('NFC', t)
        return re.sub(r'[^가-힣a-zA-Z0-9]', '', t)

    # Load experiment metadata
    entries = {}
    with open(EXPERIMENT_SCRIPT, 'r', encoding='utf-8') as f:
        for line in f:
            if '|' in line:
                fn, tx = line.strip().split('|', 1)
                entries[fn.strip()] = tx.strip()

    # Filter to formal endings
    candidates = []
    for fn, gt in entries.items():
        ng = norm(gt)
        for e in FORMAL_ENDINGS:
            if ng.endswith(e):
                candidates.append((fn, gt, e))
                break

    print(f"Total segments: {len(entries)}")
    print(f"Formal ending segments to verify: {len(candidates)}")

    truncated = 0
    ok = 0

    for i, (fn, gt, ending) in enumerate(candidates):
        wav_path = os.path.join(EXPERIMENT_WAV_DIR, fn)
        if not os.path.exists(wav_path):
            continue

        samples, sr = sf.read(str(wav_path), dtype='float32')
        duration_sec = len(samples) / sr
        target_len = int(duration_sec * 16000)
        indices = np.linspace(0, len(samples) - 1, target_len).astype(int)
        audio_16k = samples[indices].astype(np.float32)

        # Strip envelope
        strip_lead = int(16000 * 350 / 1000)
        strip_tail = int(16000 * 700 / 1000)
        if len(audio_16k) > strip_lead + strip_tail + 1600:
            audio_16k = audio_16k[strip_lead:-strip_tail] if strip_tail > 0 else audio_16k[strip_lead:]

        try:
            result = w.transcribe(
                model, audio_16k, language="ko", verbose=False,
                fp16=(device == "cuda"), condition_on_previous_text=False,
            )
            whisper_text = result.get('text', '').strip()
        except:
            continue

        norm_whisper = norm(whisper_text)
        norm_ending = norm(ending)

        is_truncated = not norm_whisper.endswith(norm_ending)

        if is_truncated:
            truncated += 1
            ng = norm(gt)
            print(f"  [{i+1}/{len(candidates)}] {fn}: TRUNCATED")
            print(f"    GT:      ...{ng[-20:]}")
            print(f"    Whisper: ...{norm_whisper[-20:]}")
        else:
            ok += 1
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(candidates)}] ...{ok} OK so far")

    print(f"\n--- VERIFICATION RESULTS ---")
    print(f"Checked:   {ok + truncated}")
    print(f"OK:        {ok}")
    print(f"TRUNCATED: {truncated}")
    if ok + truncated > 0:
        print(f"Truncation rate: {truncated/(ok+truncated)*100:.1f}%")

    return truncated


if __name__ == "__main__":
    matched = run_test()
    if matched and matched > 0:
        truncated = verify_results()
        print("\n" + "=" * 60)
        if truncated == 0:
            print("SUCCESS: No truncation detected in experiment output!")
        else:
            print(f"WARNING: {truncated} truncated segments remain")
        print("=" * 60)
