# TTS Dataset Pipeline - Requirements & Iterative R&D Plan

> **This document is the single source of truth.**
> After EVERY pipeline execution, re-read this file from the top and verify ALL requirements before reporting completion.
> **DO NOT mark the task as done until every requirement scores >= 95%.**

---

## 0. Mission Statement

Build and iteratively refine an audio-script alignment & splitting pipeline (`src/align_and_split.py`) that produces TTS training-ready WAV segments from Korean voice recordings. Each output WAV must correspond **exactly** to its paired transcript line - verified by re-transcribing the sliced WAV with Whisper and comparing against the ground-truth script. The pipeline must **autonomously repeat research -> implement -> evaluate -> improve cycles** until the quality gate (>= 95% match rate) is cleared.

---

## 1. Glossary

| Term | Definition |
|------|------------|
| **Ground-Truth Script** | The original Korean text scripts (`rawdata/Scripts/Script_{N}_A0.txt`) read aloud in the source audio. Each file has numbered lines in `idx\|text` format. |
| **Source Audio** | Long-form WAV recordings (`rawdata/audio/Script_{N}_{Start}-{End}.wav`) containing continuous speech covering a range of script lines. 19 files, ~6.6 GB total. |
| **Segment / Slice** | A single WAV file produced by the splitting logic (output to `datasets/wavs/`), intended to contain exactly one script line of speech. Named `Script_{N}_{LLLL}.wav` (4-digit zero-padded). |
| **Metadata** | The file `datasets/script.txt` mapping each WAV filename to its ground-truth text. Format: `Script_{N}_{LLLL}.wav\|Korean text here` |
| **Transcription** | The text Whisper STT returns when it processes a sliced Segment WAV. |
| **Match Rate** | `(segments whose Whisper transcription matches ground-truth at >= 95% character-level similarity) / (total segments) x 100` |
| **Similarity** | Normalized character-level similarity between ground-truth and Whisper transcription, after text normalization (strip punctuation, keep only Hangul/alphanumeric). |

---

## 2. Requirements Checklist

> **After each full pipeline run, fill in the Status column. If ANY requirement is FAIL, trigger a new R&D iteration.**

| ID | Requirement | Acceptance Criteria | Status | Score |
|---|---|---|---|---|
| **R1** | **Script <-> Segment Alignment Accuracy** | Whisper re-transcription of each sliced WAV must match its paired ground-truth script line (from `datasets/script.txt`) with >= 95% normalized character-level similarity. The metadata label MUST match what is actually spoken in the WAV file. | -- | --% |
| **R2** | **No Boundary Noise** | Every sliced WAV must have clean boundaries - no clicks, pops, silence artifacts, or **residual speech from adjacent lines** at the **first 50ms** and **last 50ms**. Measured by: (a) RMS energy of boundary regions < -40dB threshold, AND (b) no partial-word detection by Whisper in boundary windows. Zero sentence bleeding allowed. | -- | --% |
| **R3** | **Overall Pipeline Match Rate >= 95%** | Across ALL segments in the dataset, the percentage of segments satisfying R1 AND R2 simultaneously must be >= 95%. | -- | --% |
| **R4** | **Metadata Integrity** | Output `datasets/script.txt` must have exactly one entry per WAV file. Validation report must log: `filename`, `ground_truth_text`, `whisper_transcription`, `similarity_score`, `boundary_noise_pass`, `final_verdict (PASS/FAIL)` for every segment. Zero orphan WAVs (files without metadata), zero missing WAVs (metadata without files). | -- | -- |
| **R5** | **Reproducibility** | Pipeline must be deterministic given the same input. All parameters (model size, search window, thresholds, padding values) must be logged in the evaluation report. | -- | -- |

---

## 3. Pipeline Architecture

```
+-------------------------------------------------------------+
|                    ITERATIVE R&D LOOP                        |
|                                                              |
|   +--------------+    +-----------+    +-----------+         |
|   |   STAGE 1    |--->|  STAGE 2  |--->|  STAGE 3  |         |
|   | Align & Split|    | Clean &   |    | Validate  |         |
|   | (Whisper     |    | Post-     |    | (per-seg  |         |
|   |  matching)   |    | process   |    |  re-STT)  |         |
|   +--------------+    +-----------+    +-----+-----+         |
|                                              |               |
|                                         +----v----+          |
|                                         | STAGE 4 |          |
|                                         | Evaluate|          |
|                                         | (agg.)  |          |
|                                         +----+----+          |
|                                              |               |
|                                    +---------v----------+    |
|                                    |   Score >= 95% ?   |    |
|                                    +--+-------------+---+    |
|                                  YES  |             | NO     |
|                       +---------------v-+   +-------v-----+  |
|                       |    STAGE 5      |   |   STAGE 6   |  |
|                       |   Finalize &    |   |  Diagnose & |  |
|                       |    Report       |   |  Research & |  |
|                       +-----------------+   |  Improve    |  |
|                                             +------+------+  |
|                                                    v         |
|                                       Loop back to STAGE 1   |
+-------------------------------------------------------------+
```

---

## 4. Stage Specifications

### STAGE 1 - Align & Split (`src/align_and_split.py`)

**Goal:** For each source audio WAV, use Whisper to transcribe with timestamps, match each Whisper segment to the correct ground-truth script line, and extract the corresponding audio slice.

**Input (PRIMARY SOURCE - always process from rawdata/, not from datasets/):**
- Source audio: `rawdata/audio/Script_{N}_{Start}-{End}.wav` (19 files, ~6.6 GB)
- Ground-truth scripts: `rawdata/Scripts/Script_{N}_A0.txt` (5 files, numbered `idx|text` format)

**NOTE:** Any existing files in `datasets/` are from previous (potentially flawed) processing runs. They may be used as reference but must NOT be trusted as correct. The pipeline always processes fresh from `rawdata/`.

**Core Algorithm Requirements:**
1. **Whisper transcription** of each source audio file with timestamp extraction (model: medium recommended for Korean accuracy, configurable)
2. **Segment merging** - Try matching 1, 2, and 3 consecutive Whisper segments combined, because Whisper often splits a single Korean sentence into multiple segments
3. **Forward-only search window** (25 lines) - Only search forward from current script position to prevent false-match cascading in Korean text
4. **Skip penalty** (0.01 per line distance) - Penalize matches far from current position to keep alignment on track
5. **Text normalization** before comparison - Strip all punctuation, keep only Hangul + alphanumeric: `re.sub(r'[^ga-hia-zA-Z0-9]', '', text)`
6. **Consecutive fail limit** (5) - After N failed matches, advance script line to prevent infinite stalls
7. **Gap-aware dynamic padding** - Only pad audio edges when a real silence gap (>= 20ms) exists between segments; zero-pad on back-to-back segment boundaries to prevent sentence bleeding

**Output:**
- Sliced WAVs: `datasets/wavs/Script_{N}_{LLLL}.wav` (48KHz, 24-bit, mono — preserve source quality, NO downsampling)
- Metadata: `datasets/script.txt` (format: `Script_{N}_{LLLL}.wav|Korean text`)
- Skip log: `logs/skipped_lines.log` (format: `AudioFile|ScriptLine|Reason|ScriptText|WhisperText`)

**Key Parameters to Tune (R&D targets):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_SIZE` | `"medium"` | Whisper model size (tiny/base/small/medium/large) |
| `SEG_SEARCH_WINDOW` | `25` | Forward-only search window (lines) |
| `SKIP_PENALTY` | `0.01` | Similarity penalty per skipped line |
| `MATCH_THRESHOLD` | `0.25` | Minimum adjusted similarity to accept |
| `CONSEC_FAIL_LIMIT` | `5` | Consecutive failures before force-advance |
| `AUDIO_PAD_MS` | `50` | Base audio padding in milliseconds |
| `MIN_GAP_FOR_PAD_MS` | `20` | Minimum inter-segment gap to apply padding |
| `FADE_MS` | `10` | Fade-in/fade-out duration |

**Known Constraints (Korean-specific, proven from prior R&D):**
- 25-line forward window + 0.01 skip penalty is optimal for Korean text
- DO NOT use file-range anchoring - audio filenames don't reliably reflect actual content ranges
- Backward search windows cause duplicate matches - forward-only is safer
- Wider windows (100-500 lines) cause Korean false-match cascading
- Always deduplicate script.txt after processing
- Script file encoding: try UTF-8-sig / UTF-8 / CP949 / EUC-KR in order

---

### STAGE 2 - Clean & Post-process

**Goal:** Ensure every sliced WAV has clean audio boundaries with no artifacts or residual speech from adjacent lines.

**Process (applied to each sliced WAV):**
```
For each segment WAV in datasets/wavs/:
    1. Zero-crossing snap: shift cut point to nearest zero-crossing within +/-10ms
    2. Apply fade-in (5-20ms) and fade-out (5-20ms) with raised-cosine envelope
    3. Trim leading/trailing silence (RMS < -40dB threshold)
    4. Normalize peak amplitude to -1dB
    5. Re-export as WAV (48KHz, 24-bit, mono — preserve source quality)
```

**Boundary Noise Elimination Techniques:**
- Zero-crossing snap: shift cut point to nearest zero-crossing within +/-10ms to avoid clicks
- Fade envelope: raised-cosine fade-in/out (configurable 5-30ms)
- Spectral gating: apply lightweight noise gate to first/last 50ms if needed
- Double-check: re-run Whisper on first/last 100ms - must return empty or silence

---

### STAGE 3 - Validate (Per-Segment)

**Goal:** Re-transcribe each sliced WAV with Whisper and compare against ground-truth to verify alignment accuracy.

**Process:**
```
For each segment WAV:
    1. Transcribe with Whisper (same model size as alignment stage, language="ko")
    2. Normalize both ground-truth and Whisper texts:
       - Strip all punctuation (keep only Hangul + alphanumeric)
       - Lowercase English portions
       - Collapse multiple spaces
       - Unicode NFC normalization
    3. Compute similarity:
       - CER = levenshtein_distance(norm_gt, norm_whisper) / max(len(gt), len(whisper))
       - similarity = 1 - CER
    4. Boundary noise check:
       a. RMS of first 50ms < -40dB
       b. RMS of last 50ms < -40dB
       c. Whisper transcription of first 100ms == empty (no partial words)
       d. Whisper transcription of last 100ms == empty (no partial words)
    5. Verdict: PASS if similarity >= 0.95 AND all boundary checks pass
```

---

### STAGE 4 - Evaluate (Aggregate)

**Goal:** Compute the overall pipeline match rate and generate a detailed report.

**Output: `logs/evaluation_report.json`**

```json
{
  "timestamp": "ISO-8601",
  "iteration": 1,
  "total_segments": "<determined by pipeline>",
  "passed_segments": 0,
  "failed_segments": 0,
  "overall_match_rate": 0.0,
  "r1_alignment_accuracy": 0.0,
  "r2_boundary_noise_pass_rate": 0.0,
  "r3_combined_pass_rate": 0.0,
  "parameters_used": {
    "whisper_model": "medium",
    "search_window": 25,
    "skip_penalty": 0.01,
    "match_threshold": 0.25,
    "audio_pad_ms": 50,
    "min_gap_for_pad_ms": 20,
    "consec_fail_limit": 5,
    "fade_ms": 10,
    "silence_threshold_db": -40
  },
  "per_script_breakdown": {
    "Script_1": { "total": "<auto>", "passed": 0, "rate": 0.0 },
    "Script_2": { "total": "<auto>", "passed": 0, "rate": 0.0 },
    "Script_3": { "total": "<auto>", "passed": 0, "rate": 0.0 },
    "Script_4": { "total": "<auto>", "passed": 0, "rate": 0.0 },
    "Script_5": { "total": "<auto>", "passed": 0, "rate": 0.0 }
  },
  "failed_segment_details": [
    {
      "filename": "Script_1_0042.wav",
      "ground_truth": "example text",
      "whisper_transcription": "example whisper output",
      "similarity": 0.92,
      "boundary_noise_pass": false,
      "failure_reason": "similarity below threshold",
      "failure_type": "Type A"
    }
  ]
}
```

---

### STAGE 5 - Finalize (if score >= 95%)

**Actions:**

1. Verify all WAV files in `datasets/wavs/` have corresponding entries in `datasets/script.txt` (zero orphans, zero missing).
2. Generate final `logs/evaluation_report.json` with all scores.
3. Generate `datasets/validation_results.csv` with columns: `filename | ground_truth | whisper_text | similarity | boundary_pass | verdict`
4. **Re-read this requirements document and verify ALL checklist items.**
5. Update the Requirements Checklist (Section 2) with PASS and actual scores.
6. Print final summary to console.

---

### STAGE 6 - Diagnose, Research & Improve (if score < 95%)

> **THIS IS THE CRITICAL AUTONOMOUS R&D STAGE.**
> You MUST NOT skip this stage. You MUST NOT give up.
> Analyze failures, hypothesize causes, implement fixes, and re-run the entire pipeline.

**Diagnosis Protocol:**

```
1. CATEGORIZE all failed segments:
   - Type A: Alignment shift (transcription is correct text but from WRONG script line)
   - Type B: Merge/Split error (two lines merged into one WAV, or one line split across two)
   - Type C: Boundary noise / sentence bleeding (residual speech from adjacent line at edges)
   - Type D: Whisper hallucination / recognition error (Whisper misheard the audio)
   - Type E: Script mismatch (script text has errors, not an audio problem)

2. IDENTIFY the dominant failure type and its root cause.

3. RESEARCH & APPLY targeted fixes:

   For Type A (Alignment Shift - wrong script line matched):
   -> Tighten search window (reduce SEG_SEARCH_WINDOW)
   -> Increase skip penalty (increase SKIP_PENALTY)
   -> Try different Whisper model size for better Korean recognition
   -> Add anchor-point alignment (match high-confidence words first)
   -> Use dynamic time warping (DTW) on character-level alignment

   For Type B (Merge/Split - segments combined or split wrong):
   -> Adjust segment merge count (try 1-2 only, or extend to 1-4)
   -> Add silence detection between lines as secondary signal
   -> Use energy-based segmentation to detect sentence boundaries
   -> Adjust min/max segment duration constraints
   -> Implement sliding-window realignment for merged segments

   For Type C (Boundary Noise / Sentence Bleeding):
   -> Reduce AUDIO_PAD_MS (50 -> 30 -> 20 -> 0)
   -> Increase MIN_GAP_FOR_PAD_MS threshold
   -> Increase fade duration (10ms -> 20ms -> 30ms)
   -> Tighten zero-crossing snap window
   -> Apply spectral subtraction at boundaries
   -> Increase silence trim aggressiveness

   For Type D (Whisper Hallucination):
   -> Try larger Whisper model (medium -> large-v3)
   -> Ensure language="ko" is set
   -> Use condition_on_previous_text=False
   -> Apply VAD filter before transcription
   -> Use multiple Whisper runs and majority vote

   For Type E (Script Mismatch):
   -> Flag for manual review (do not auto-fix)
   -> Log and exclude from scoring denominator

4. IMPLEMENT the fix in src/align_and_split.py (primary target).
5. RETURN TO STAGE 1 and re-run the full pipeline.
```

**Iteration Safeguards:**
- Maximum 10 iterations before requesting human intervention.
- Log all changes made per iteration in `logs/iteration_log.md`.
- Never revert a fix that improved the score without trying an alternative first.
- If score plateaus for 3 consecutive iterations, try a fundamentally different approach.

---

## 5. Iteration Log Template

> Append to `logs/iteration_log.md` after each R&D cycle.

```markdown
## Iteration N - [YYYY-MM-DD HH:MM]

### Scores
- R1 (Alignment Accuracy): ___%
- R2 (Boundary Noise Pass): ___%
- R3 (Combined Match Rate): ___%

### Failure Analysis
- Type A (Alignment Shift): ___ segments
- Type B (Merge/Split): ___ segments
- Type C (Boundary Noise): ___ segments
- Type D (Whisper Error): ___ segments
- Type E (Script Error): ___ segments
- Dominant failure: Type __

### Changes Made
- [ ] Description of change 1
- [ ] Description of change 2

### Hypothesis
> Why this change should improve the score.

### Result
- Previous score: ___%
- New score: ___%
- Delta: +/-___%
```

---

## 6. File & Directory Structure (Actual Project)

```
G:\Projects\AI_Research\TTSDataSetCleanser\
|
+-- OrderSheets/
|   +-- TTS_DATASET_PIPELINE_REQUIREMENTS.md   <-- THIS FILE
|
+-- src/
|   +-- align_and_split.py    # CORE: Stages 1-2 (PRIMARY R&D TARGET)
|   +-- pipeline_manager.py   # Reference: proven algorithms to port from
|   +-- batch_align_whisper.py # Reference: segment merging patterns
|   +-- process_script2.py    # Reference: Script_2 processor
|   +-- process_script5.py    # Reference: Script_5 processor
|   +-- process_missed.py     # Reference: orphan WAV recovery
|   +-- validate_dataset.py   # Utility: dataset integrity checks
|   +-- transcribe_wavs.py    # Utility: batch Whisper transcription
|   +-- clean_scripts.py      # Utility: script text cleaning
|   +-- compare_scripts.py    # Utility: script comparison
|
+-- rawdata/
|   +-- audio/                # INPUT: 19 source WAV files (~6.6 GB)
|   |   +-- Script_1_1-122.wav      (lines 1-122)
|   |   +-- Script_1_123-220.wav    (lines 123-220)
|   |   +-- Script_1_221-300.wav    (lines 221-300)
|   |   +-- Script_2_1-176.wav      (lines 1-176)
|   |   +-- Script_2_177-547.wav    ...
|   |   +-- Script_2_548-626.wav
|   |   +-- Script_2_627-1000.wav
|   |   +-- Script_2_1001-1505.wav
|   |   +-- Script_2_1506-1644.wav
|   |   +-- Script_3_1-404.wav
|   |   +-- Script_3_405-878.wav
|   |   +-- Script_4_1-100.wav
|   |   +-- Script_4_101-621.wav
|   |   +-- Script_4_622-650.wav
|   |   +-- Script_4_651-980.wav
|   |   +-- Script_4_981-1005.wav
|   |   +-- Script_5_1-43.wav
|   |   +-- Script_5_44-200.wav
|   |   +-- Script_5_201-541.wav
|   |
|   +-- Scripts/              # INPUT: 5 ground-truth script files
|       +-- Script_1_A0.txt   (300 lines)
|       +-- Script_2_A0.txt   (1644 lines)
|       +-- Script_3_A0.txt   (878 lines)
|       +-- Script_4_A0.txt   (1005 lines)
|       +-- Script_5_A0.txt   (1018 lines)
|
+-- datasets/                  # OUTPUT (REGENERATED by pipeline — do not treat as source of truth)
|   +-- script.txt            # OUTPUT: metadata (filename|ground_truth_text)
|   +-- whisper_transcribed.txt  # OUTPUT: Whisper re-transcription results
|   +-- validation_results.csv   # OUTPUT: per-segment validation details
|   +-- wavs/                 # OUTPUT: sliced WAV segments (48KHz, 24-bit, mono)
|       +-- Script_{N}_{LLLL}.wav  (quantity determined by pipeline results)
|
+-- logs/
|   +-- iteration_log.md      # R&D iteration history
|   +-- evaluation_report.json # Latest evaluation results
|   +-- skipped_lines.log     # Failed/skipped line details
|   +-- align_and_split.log   # Pipeline execution log
|
+-- TaskLogs/                 # Historical pipeline execution logs
+-- requirements.txt          # Python dependencies
```

---

## 7. Execution Protocol for Claude Code

> **Read this section as your operating instructions.**

```
REPEAT {
    1. Read TTS_DATASET_PIPELINE_REQUIREMENTS.md (this file) completely.

    2. Run the full pipeline:
       a. STAGE 1-2: Execute src/align_and_split.py (align + split + clean)
       b. STAGE 3: Validate every segment (re-transcribe with Whisper, compare to ground-truth)
       c. STAGE 4: Aggregate scores into logs/evaluation_report.json

    3. Check evaluation_report.json:
       - If R1 >= 95% AND R2 >= 95% AND R3 >= 95%:
           -> Execute STAGE 5 (Finalize).
           -> Re-read this file.
           -> Update the Requirements Checklist (Section 2) with PASS and actual scores.
           -> STOP. Report success.
       - Else:
           -> Execute STAGE 6 (Diagnose, Research, Improve).
           -> Log the iteration in logs/iteration_log.md.
           -> CONTINUE loop.

    4. Safety: If iteration count > 10, STOP and report:
       "Maximum iterations reached. Best score: X%.
        Manual intervention required for the following segments: [list]"

} UNTIL (all requirements met) OR (iteration > 10)
```

---

## 8. Quality Metrics - Scoring Formula

```
Text Normalization (applied to BOTH ground-truth and Whisper output):
  normalized = re.sub(r'[^ga-hia-zA-Z0-9]', '', text)
  normalized = unicodedata.normalize('NFC', normalized)

Per-Segment Score:
  CER = levenshtein_distance(normalized_gt, normalized_whisper)
        / max(len(normalized_gt), len(normalized_whisper))
  similarity = 1 - CER
  boundary_clean = (rms_first_50ms < -40dB) AND (rms_last_50ms < -40dB)
  segment_pass = (similarity >= 0.95) AND boundary_clean

Overall Score:
  R1_score = (count(similarity >= 0.95) / total_segments) x 100
  R2_score = (count(boundary_clean == True) / total_segments) x 100
  R3_score = (count(segment_pass == True) / total_segments) x 100

  ALL THREE must be >= 95% to pass.
```

---

## 9. Critical Reminders

> **DO NOT** report the task as complete if any score is below 95%.
> **DO NOT** skip STAGE 6 - autonomous R&D is the core of this pipeline.
> **DO NOT** manually edit output WAV files - all fixes must be algorithmic.
> **DO** log every iteration with before/after scores.
> **DO** re-read this document after every iteration to ensure no requirement drift.
> **DO** prioritize the dominant failure type in each R&D cycle.
> **DO** preserve all intermediate outputs for debugging.
> The `align_and_split.py` file is the most critical code - iterate on it relentlessly.
W
---

## 10. Environment & Constraints

| Item | Value |
|------|-------|
| **OS** | Windows (Git Bash / MSYS2 for shell) |
| **Filesystem** | G: drive is exFAT (no NTFS ACLs, possible PermissionError on file overwrites) |
| **Python** | 3.11 at `C:\Users\Polymath\AppData\Local\Programs\Python\Python311\python.exe` |
| **GPU** | NVIDIA GeForce RTX 3060 Ti (CUDA available) |
| **Whisper** | openai-whisper (pip), language="ko" |
| **Audio output format** | 48000Hz, 24-bit, mono WAV |
| **WAV naming** | `Script_{N}_{LLLL}.wav` - always 4-digit zero-padded line number |
| **Metadata format** | Pipe-delimited: `filename\|text` in `datasets/script.txt` |
| **Script encoding** | Try: UTF-8-sig / UTF-8 / CP949 / EUC-KR (in order) |
| **Source data** | 19 audio files covering Script_1 (lines 1-300), Script_2 (lines 1-1644), Script_3 (lines 1-878), Script_4 (lines 1-1005), Script_5 (lines 1-541). Script_5 lines 542-1018 have no audio yet. |
| **Previous run** | datasets/ may contain ~4,063 entries from a prior flawed run. Do NOT depend on these — reprocess from rawdata/. Output count will be determined by new pipeline results. |

---

## 11. Success Criteria (Final Gate)

```
R1 (Alignment Accuracy)      >= 95%   -- Whisper re-transcription matches ground-truth
R2 (Boundary Noise Clean)    >= 95%   -- No residual speech or artifacts at edges
R3 (Combined Match Rate)     >= 95%   -- R1 AND R2 both pass per segment
R4 (Metadata Complete)       = 95%   -- Every WAV should has metadata, only 5% or lessthan orphans are acceptable.
R5 (Reproducible)            = YES    -- All parameters logged, deterministic

-> ALL FIVE must be satisfied. No exceptions.
```

---

*End of Requirements Document*
