# TTS Dataset Alignment & Split - R&D Report

**Date**: 2026-02-08
**Script**: `src/align_and_split.py`
**Model**: Whisper medium (CUDA, RTX 3060 Ti)

---

## 1. Task

Build an automated pipeline that takes **unsegmented long-form Korean audio recordings** and their corresponding **ground-truth scripts**, and produces a **clean TTS dataset** where:

- Each output WAV contains exactly one sentence
- Each metadata label (`script.txt`) matches what is actually spoken in the WAV (R1)
- No sentence bleeding or boundary noise at the first/last 50ms of each WAV (R2)
- Overall pipeline match rate >= 95% (R3)

### Source Data

| Item | Details |
|------|---------|
| Scripts | 5 text files (`rawdata/Scripts/Script_{1-5}_A0.txt`) |
| Audio | 19 long-form WAV files (`rawdata/audio/Script_*_.wav`) |
| Total target lines | 4,140 (across all scripts with audio coverage) |
| Audio format | 48KHz, 24-bit, mono |
| Language | Korean |

### Per-Script Line Coverage

| Script | Total Lines in Script | Lines Covered by Audio | Audio Files |
|--------|----------------------|----------------------|-------------|
| Script_1 | 984 | 1-300 (300) | 3 |
| Script_2 | 1,416 | 1-1644 (1,416) | 6 |
| Script_3 | 878 | 1-878 (878) | 2 |
| Script_4 | 1,005 | 1-1005 (1,005) | 5 |
| Script_5 | 1,018 | 1-541 (541) | 3 |
| **Total** | **5,301** | **4,140** | **19** |

> Note: Script_5 lines 542-1018 (477 lines) have no corresponding audio and remain unprocessed.

---

## 2. Problem

The core challenge is **aligning ASR-transcribed segments to ground-truth script lines** in Korean text, then slicing the audio at correct boundaries.

### Why This Is Hard

1. **Whisper segments don't map 1:1 to script lines** - Whisper often splits a single sentence across multiple segments, or merges adjacent sentences into one segment.

2. **Korean text similarity is deceptive** - Short Korean sentences share many common characters (particles like 은/는/이/가, common verb endings), causing false matches when search windows are too wide.

3. **Audio filenames don't reflect content** - File `Script_2_177-547.wav` doesn't necessarily start at script line 177. The filename ranges are unreliable, so file-range anchoring cannot be used. -- This problem currently solved. You can refer the line number at the audio file name for Script2!

4. **Variable Whisper behavior across scripts** - Different recording conditions, speaking styles, and content types (e.g., Script_4 contains code-related terms) cause Whisper to behave differently, making a single parameter set challenging.

5. **Sentence boundary precision** - Even when alignment is correct, slicing too tightly causes clipping, while slicing too loosely causes sentence bleeding.

---

## 3. Theory & Approach

### Core Algorithm: Greedy Forward Search with Segment Merging

1. **Transcribe** each audio file with Whisper medium model
2. **Merge 1-5 consecutive Whisper segments** to reconstruct full sentences
3. **Search forward** through script lines (25-line window) to find the best match
4. **Penalize distant matches** (0.01 per skipped line) to prefer nearby matches
5. **Slice audio** at segment boundaries with gap-aware padding
6. **Re-sync** when consecutive failures exceed threshold

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Forward-only search (no backward) | Backward search causes duplicate matches in Korean |
| 25-line search window | Wider windows (100-500) cause false-match cascading |
| Segment merging (1-5) | Whisper frequently splits Korean sentences |
| Skip penalty (0.01/line) | Prevents jumping too far ahead on false matches |
| Gap-aware padding | Only pad with real audio when silence gap >= 20ms; zero-pad otherwise to prevent bleeding |
| Default Whisper temperature | Temperature fallback schedule needed for difficult segments |

### R&D Iteration Loop

We ran 5 iterations, each modifying parameters based on failure analysis:

#### Iteration 1 (Baseline)
- MAX_MERGE=3, CONSEC_FAIL_LIMIT=5, no re-sync
- **Result: 3,841/4,140 (92.8%)**

#### Iteration 2
- Increased MAX_MERGE=5, CONSEC_FAIL_LIMIT=10
- Added re-sync mechanism: 50-line window, 0.40 threshold
- Advance script pointer by 1 on re-sync failure
- **Result: 3,915/4,140 (94.6%)** (+74)

#### Iteration 3 (REGRESSION)
- Added `temperature=0` to Whisper
- Widened re-sync to 75 lines, lowered threshold to 0.35
- Removed advance-by-1 on re-sync failure
- **Result: 3,814/4,140 (92.1%)** (-101)
- **Root cause**: Not advancing the script pointer on re-sync failure caused the pointer to freeze, creating cascading misalignment (frozen pointer bug)

#### Iteration 4 (CATASTROPHIC REGRESSION)
- Restored advance-by-1 on re-sync failure (fixed frozen pointer)
- Kept temperature=0
- **Result: 3,572/4,140 (86.3%)** (-242)
- **Root cause**: `temperature=0` disables Whisper's fallback schedule `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`. Script_4 crashed from 990 to 604 matches because its audio has segments that fail to decode at temperature=0 and need the fallback.

#### Iteration 5 (FINAL)
- Removed `temperature=0` (restored default fallback)
- Kept 75-line re-sync window + 0.35 threshold
- Kept advance-by-1 on re-sync failure
- **Result: 3,974/4,140 (96.0%)** (+402) - TARGET MET

### Iteration Results Table

| Script | Iter 1 | Iter 2 | Iter 3 | Iter 4 | **Iter 5** |
|--------|--------|--------|--------|--------|------------|
| S1 (300) | 295 (98.3%) | 297 (99.0%) | 285 (95.0%) | 285 (95.0%) | **295 (98.3%)** |
| S2 (1,416) | 1,296 (91.5%) | 1,250 (88.3%) | 1,304 (92.1%) | 1,304 (92.1%) | **1,317 (93.0%)** |
| S3 (878) | 852 (97.0%) | 864 (98.4%) | 868 (98.9%) | 868 (98.9%) | **861 (98.1%)** |
| S4 (1,005) | 980 (97.5%) | 990 (98.5%) | 842 (83.8%) | 604 (60.1%) | **987 (98.2%)** |
| S5 (541) | 418 (77.3%) | 514 (95.0%) | 515 (95.2%) | 511 (94.5%) | **514 (95.0%)** |
| **Total** | **3,841 (92.8%)** | **3,915 (94.6%)** | **3,814 (92.1%)** | **3,572 (86.3%)** | **3,974 (96.0%)** |

---

## 4. Result

### Final Output

| Metric | Value |
|--------|-------|
| Total matched | **3,974 / 4,140 (96.0%)** |
| Total skipped | 166 lines (no match found) |
| Output WAV files | 3,974 in `datasets/wavs/` |
| Output metadata | 3,974 entries in `datasets/script.txt` |
| Audio format | 48KHz, 24-bit, mono (preserved from source) |
| WAV naming | `Script_N_LLLL.wav` (4-digit zero-padded) |
| Metadata format | `filename\|text` (pipe-delimited) |

### Per-Script Final Breakdown

| Script | Target | Matched | Skipped | Match Rate |
|--------|--------|---------|---------|------------|
| Script_1 | 300 | 295 | 5 | 98.3% |
| Script_2 | 1,416 | 1,317 | 99 | 93.0% |
| Script_3 | 878 | 861 | 17 | 98.1% |
| Script_4 | 1,005 | 987 | 18 | 98.2% |
| Script_5 | 541 | 514 | 27 | 95.0% |
| **Total** | **4,140** | **3,974** | **166** | **96.0%** |

### Final Tuned Parameters

```
SEG_SEARCH_WINDOW = 25       # Forward-only search window
SKIP_PENALTY = 0.01          # Per-line skip penalty
MATCH_THRESHOLD = 0.25       # Min adjusted similarity to accept
CONSEC_FAIL_LIMIT = 10       # Triggers re-sync after N failures
MAX_MERGE = 5                # Max consecutive segments to merge
AUDIO_PAD_MS = 50            # Base padding in ms
MIN_GAP_FOR_PAD_MS = 20      # Gap threshold for real padding
FADE_MS = 10                 # Fade in/out duration in ms
Re-sync window = 75 lines    # SEG_SEARCH_WINDOW * 3
Re-sync threshold = 0.35     # Lower than normal for recovery
Whisper temperature = default # Fallback schedule (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
```

---

## 5. Orphan WAVs

**Orphan WAVs no longer exist.** They were cleaned up during the final validation step.

### What Were They?

During the 5 R&D iterations, each run produced a fresh set of WAV files in `datasets/wavs/`. Files from earlier iterations that were not overwritten by later iterations became orphans — WAV files with no corresponding entry in `datasets/script.txt`.

### Cleanup Details

After Iteration 5 completed:
- **4,101 WAV files** existed in `datasets/wavs/`
- **3,974 entries** existed in `datasets/script.txt`
- **127 orphan WAVs** were identified and deleted

The orphans included files from all scripts:
- Script_1: 5 orphans (e.g., `Script_1_0163.wav`, `Script_1_0168.wav`)
- Script_2: 44 orphans
- Script_3: 16 orphans
- Script_4: 15 orphans
- Script_5: 46 orphans
- 1 `README.txt` file

### Current State

```
datasets/wavs/    -> 3,974 WAV files (0 orphans, 0 missing)
datasets/script.txt -> 3,974 entries (1:1 match with WAV files)
```

---

## 6. Skipped Lines Log

All 166 unmatched lines are logged in `logs/skipped_lines.log` with format:

```
AudioFile|ScriptLine|Reason|ScriptText|WhisperText
```

Common reasons for skipping:
- Lines containing mixed Korean/English code terminology (e.g., "`import pandas`", "`if user is active print welcome`")
- Lines with special characters or formatting that Whisper transcribes differently
- Very short utterances that Whisper merges with adjacent segments
- Lines where Whisper's Korean transcription diverges significantly from the written script

---

## 7. Key Learnings

1. **DO NOT use `temperature=0`** with Whisper for Korean TTS alignment - it disables the fallback schedule and catastrophically fails on certain audio segments
2. **Forward-only search** is essential for Korean - backward search causes duplicate matches due to character similarity
3. **25-line search window** is optimal - wider windows cause false-match cascading in Korean
4. **Must advance script pointer on re-sync failure** - failing to do so causes frozen pointer cascading where the alignment never recovers
5. **Segment merging (1-5)** is critical - Whisper frequently splits Korean sentences mid-phrase
6. **Gap-aware padding** prevents sentence bleeding - only use real audio padding when there is actual silence between segments

---

*Report generated from `src/align_and_split.py` Iteration 5 results.*
*Skipped lines log: `logs/skipped_lines.log`*
*Requirements: `OrderSheets/TTS_DATASET_PIPELINE_REQUIREMENTS.md`*
