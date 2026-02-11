# TTS Dataset Transcription Workflow

Proven workflow for generating script.txt from pre-segmented WAV files.
Established from the Script_5 Experiment (2026-02-08).

## Prerequisites

| Item | Description |
|------|-------------|
| Pre-segmented WAVs | One WAV file per script line, named `Script_N_LLLL.wav` (4-digit zero-padded) |
| Answer key script | The exact script sheet the voiceover actor read from (`Script_N_A0.txt`) |
| Python 3.11 | With `whisper`, `soundfile`, `librosa`, `torch` installed |
| GPU | CUDA-capable GPU (tested on RTX 3060 Ti) |

## Scoring Metric: WER Score

### Formula

```
WER Score = Matched Words / ASR Recognized Words
```

| Term | Definition |
|------|------------|
| **ASR Recognized Words** | Total word count of the Whisper transcription output for that line |
| **Matched Words** | Count of ASR-recognized words that appear in the Correct Key script line |
| **Normalization** | Remove punctuation, collapse whitespace before word comparison |

### Example

```
ASR output (10 words):  "따뜻한 차를 한 잔 끓여 베란다로 나간다 시원한 밤공기가"
Correct Key (10 words): "따뜻한 차를 한 잔 끓여 베란다로 나간다. 시원한 밤공기가 얼굴을 스친다."

Matched: 따뜻한, 차를, 한, 잔, 끓여, 베란다로, 나간다 = 7 words found in Correct Key
WER Score = 7 / 10 = 0.70 (70)
```

### Target: >= 90 (0.9)

| Score Range | Meaning | Action |
|-------------|---------|--------|
| 95-100 | Excellent | Accept as-is |
| 90-94 | Good — minor Whisper differences (particle spacing, punctuation) | Accept |
| 80-89 | Marginal | Investigate audio quality or Whisper misrecognition |
| < 80 | Fail | Check for segmentation error or wrong audio content |
| Consecutive 0% block | WAV segmentation cascade | Re-segment the source audio |

## Workflow

### Step 1: Receive & Organize Source Files

```
rawdata/{SessionName}/
    wavs/              <- pre-segmented WAV files (1 file = 1 script line)
    Script_N_A0.txt    <- original script (answer key)
```

Rules:
- WAV filename number MUST correspond to the script line number
- WAV filenames MUST be 4-digit zero-padded (`Script_5_0542.wav`, not `Script_5_542.wav`)
- WAV files must be mono audio (stereo will be auto-converted)
- Any sample rate is accepted (auto-resampled to 16kHz for Whisper)

### Step 2: Transcribe

Run the transcription script:

```bash
python src/transcribe_experiment.py
```

What it does:
1. Loads Whisper `medium` model on GPU
2. Reads each WAV with `soundfile` (no ffmpeg dependency)
3. Resamples to 16kHz with `librosa` if needed
4. Transcribes with Whisper (Korean, default temperature fallback)
5. Writes output in `filename.wav|transcribed text` format

Performance: ~1 file/second on RTX 3060 Ti (~4.3 min per 260 files).

Configuration (in script):
- `WHISPER_MODEL = "medium"` — DO NOT use temperature=0
- `WHISPER_LANGUAGE = "ko"`
- Audio loaded via soundfile+librosa (no ffmpeg needed)

### Step 3: Evaluate Against Answer Key (WER Score)

The script automatically scores each line using the WER Score formula:

```
WER Score = Matched Words / ASR Recognized Words
```

For each line:
1. Normalize both ASR output and Correct Key (strip punctuation, collapse whitespace)
2. Split both into word lists
3. Count how many ASR words appear in the Correct Key word list
4. Divide by total ASR word count

Reports:
- Per-line WER Score
- Overall average WER Score (target: >= 90)
- Per-range breakdown
- Worst performers list

### Step 3.5: Detect Split Lines (CRITICAL)

**Before concluding evaluation, check every low-scoring line for split-line cases.**

A "split line" occurs when one answer key line was segmented into multiple
consecutive WAV files. This is a source segmentation issue, not a transcription error.

**Detection procedure:**

1. For each line scoring < 80:
   - Check if the ASR transcription matches the **beginning** of the answer key line
   - Check if the **next** WAV file's ASR transcription matches the **remaining part**
   - Continue checking WAV N+2, N+3... for multi-way splits
2. Confirm by concatenating the split WAVs' transcriptions and matching against the full answer key line

**Example:**

```
Answer Key line 10: "안녕하세요 오늘 날씨가 좋네요"

Script_1_0010.wav → ASR: "안녕하세요"           ← first part only → low WER Score
Script_1_0011.wav → ASR: "오늘 날씨가 좋네요"   ← remaining part

Combined: "안녕하세요 오늘 날씨가 좋네요" → matches answer key line 10 fully
Verdict: 1-to-2 split detected
```

**Rules for split lines:**
- Transcribe each WAV as-is — do NOT merge or skip any file
- Do NOT re-assign answer key lines to different WAV numbers
- Note split-line WAVs separately so they don't unfairly drag down the overall WER average
- Record every case in the Split-Line Report (see Step 6)

### Step 4: Review & Fix Source Issues

If evaluation reveals problems:

| Pattern | Diagnosis | Fix |
|---------|-----------|-----|
| Consecutive 0% block (e.g., files 573-640) | WAV segmentation offset — one file was cut too early, causing cascade | Re-segment the source audio for the affected range |
| Isolated scores 80-89 | Whisper interpretation differences | Acceptable if overall average >= 90 |
| Single 0% scattered randomly | WAV may contain wrong audio entirely | Re-check source segmentation for that specific file |
| Low score + ASR matches partial answer key | **Split line** — one script line cut across multiple WAVs | Record in Split-Line Report, transcribe each WAV as-is |
| Overall average < 90 | Systematic issue | Investigate source data quality, check for offset errors |

### Step 5: Generate Final Dataset

Once overall WER Score >= 90 (excluding identified split-line cases), the output
`transcribed_script.txt` becomes the production `script.txt`:

```
datasets/{DatasetName}/
    wavs/          <- copy verified WAV files here
    script.txt     <- copy transcribed_script.txt here
```

### Step 6: Split-Line Report (End of Session — MANDATORY)

At the end of every transcription session, produce a Split-Line Report listing
all detected cases where one answer key line was split across multiple WAVs.

**Format:**

```
===============================================================
  SPLIT-LINE REPORT
  Session: {SessionName}
  Date: {date}
===============================================================

  Total split lines detected: N
  Total affected WAV files: M

  --- Details ---

  Answer Key Line 10: "안녕하세요 오늘 날씨가 좋네요"
    → Script_1_0010.wav: "안녕하세요"
    → Script_1_0011.wav: "오늘 날씨가 좋네요"
    Type: 1-to-2 split

  Answer Key Line 572: "부엌에서 구수한 커피향이 난다 남편이 벌써 일어난 모양이다"
    → Script_5_0572.wav: "부엌에서 구수한 커피향이 난다"
    → Script_5_0573.wav: "남편이 벌써 일어난 모양이다"
    Type: 1-to-2 split

===============================================================
```

This report is **mandatory** — even if zero split lines are detected, report
"Total split lines detected: 0". The dataset manager relies on this report
to assess segmentation quality and plan corrective actions.

## Key Lessons (from Script_5 Experiment)

1. **Pre-segmented WAVs + correct script = fast & accurate.**
   No alignment algorithm needed. No re-sync. No search windows.
   260 files done in 6.4 minutes with perfect audio-to-text match.

2. **The only failure mode is source segmentation errors.**
   If a WAV file is cut at the wrong point, it cascades through
   all subsequent files. This is detectable by consecutive 0% scores
   in the evaluation step.

3. **Whisper medium scores ~92% WER Score on Korean.**
   The ~8% gap is particle spacing and punctuation — not actual
   speech content errors. For TTS training purposes, the Whisper
   transcription faithfully represents what is spoken in the audio.

4. **DO NOT set Whisper temperature=0.**
   This was proven catastrophic in earlier R&D iterations
   (604/1005 vs 987/1005 on Script_4).

## Comparison: This Workflow vs Align-and-Split Pipeline

| | This Workflow | Align-and-Split (align_and_split.py) |
|---|---|---|
| Input | Pre-segmented WAVs + script | Long-form audio + script |
| Speed | ~1 file/sec (260 files = 6 min) | Hours of processing |
| Complexity | Single Whisper pass | Alignment + search window + re-sync + splitting |
| Scoring | WER Score (matched/ASR words) >= 90 | Similarity threshold matching |
| Failure modes | Only source segmentation errors | Many (false matches, cascading skips, merge errors) |
| Tuning needed | None | 10+ parameters across 5 R&D iterations |
| Best for | Audio already cut per line | Raw uncut recording sessions |
