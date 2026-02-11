# Transcription Task Order: Pre-Segmented WAV Labeling

## Objective

Generate a new `script.txt` by transcribing the already-segmented WAV files using Whisper ASR. This is **NOT** an alignment or splitting task — the audio files are already cut and finalized. The only job is to produce an accurate text transcription for each WAV file as-is.

## Role & Responsibility

| Item | Description |
|------|-------------|
| **Role** | Dataset Transcriber |
| **Input** | Pre-segmented WAV files (1 file = 1 script line) |
| **Output** | `script.txt` in `filename.wav\|transcribed text` format |
| **Answer Key** | The original script sheet the voiceover actor read from |
| **Task** | Transcribe each WAV file using Whisper ASR, evaluate against answer key, achieve WER Score >= 90 |
| **NOT in scope** | Alignment, splitting, audio modification, re-segmentation |

## What This Is

- Accept each WAV file as a complete, final audio segment
- Run Whisper transcription on each file
- Write whatever Whisper hears — that IS the ground truth for this task
- Evaluate transcription quality using WER Score against the answer key
- **Detect split lines** — when one answer key line was split across multiple WAVs
- Report per-line scores, overall scores, and all detected split-line cases

## What This Is NOT

- NOT an align-and-split pipeline (audio is already segmented)
- NOT a QC/validation pass (no good/bad classification)
- NOT editing or modifying any audio files

## Scoring: WER (Word Error Rate) Score

The key quality metric is **WER Score** — how accurately the ASR transcription matches the correct answer key.

### Formula

```
WER Score = Matched Words / ASR Recognized Words
```

Where:
- **ASR Recognized Words** = total word count of the Whisper transcription output
- **Matched Words** = count of ASR-recognized words that appear in the Correct Key script line
- Words are compared after normalization (remove punctuation, collapse whitespace)

### Example

| | Value |
|---|---|
| ASR output | 10 words |
| Correct Key line | contains 7 of those 10 words |
| Matched Words | 7 |
| **WER Score** | 7 / 10 = **0.70 (70)** |

### Target

| Metric | Target |
|--------|--------|
| Per-line WER Score | >= 90 (0.9) for well-segmented lines |
| Overall average | >= 90 (0.9) |

### Score Interpretation

| Score Range | Meaning |
|-------------|---------|
| 95-100 | Excellent — near-perfect transcription |
| 90-94 | Good — minor Whisper interpretation differences (particle spacing, punctuation) |
| 80-89 | Marginal — check for audio quality issues or Whisper misrecognition |
| < 80 | Fail — likely source segmentation error, wrong audio content, or **split line** |
| Consecutive 0% block | WAV segmentation cascade — one file cut too early, offsetting all subsequent |

## CRITICAL: Split-Line Detection

**This is a core responsibility of the transcriber.**

When a WAV file scores low against its corresponding answer key line, you MUST check whether the audio content is a **partial match** — meaning one answer key line was split across multiple consecutive WAV files during segmentation.

### How to Detect

1. WAV file N scores low (< 80) against answer key line N
2. The ASR transcription of WAV N matches the **first part** of answer key line N
3. The ASR transcription of WAV N+1 matches the **remaining part** of answer key line N
4. (Possibly WAV N+2, N+3... for multi-way splits)

### Example

```
Answer Key line 10: "안녕하세요 오늘 날씨가 좋네요"

WAV Script_1_0010.wav → ASR: "안녕하세요"              (only first part)
WAV Script_1_0011.wav → ASR: "오늘 날씨가 좋네요"      (remaining part)
```

In this case, answer key line 10 was split into 2 WAV files.

### What to Do

1. **Transcribe each WAV as-is** — each WAV gets its own line in the output with what Whisper actually heard
2. **Do NOT merge** the WAVs or skip any file
3. **Do NOT re-assign** answer key lines to different WAV numbers
4. **Record every detected split** in a Split-Line Report at the end of the session

### Impact on Scoring

- Split-line WAVs will naturally score low individually against their answer key line
- When calculating the overall WER Score, split-line cases should be **noted separately** so they don't drag down the average unfairly
- The combined transcription of the split WAVs should cover the full answer key line

### Split-Line Report (End of Session)

At the end of every transcription session, produce a report listing all detected split lines:

```
=== SPLIT-LINE REPORT ===
Answer Key Line 10: "안녕하세요 오늘 날씨가 좋네요"
  → Script_1_0010.wav: "안녕하세요"
  → Script_1_0011.wav: "오늘 날씨가 좋네요"
  Type: 1-to-2 split

Answer Key Line 572: "부엌에서 구수한 커피향이 난다 남편이 벌써 일어난 모양이다 사랑하는 사람"
  → Script_5_0572.wav: "부엌에서 구수한 커피향이 난다"
  → Script_5_0573.wav: "남편이 벌써 일어난 모양이다 사랑하는 사람"
  Type: 1-to-2 split

Total split lines detected: N
Affected WAV files: M
```

This report is critical for the dataset manager to understand segmentation quality and decide on corrective actions.

## Output Format

Must match the existing `script.txt` format exactly:

```
{filename}|{transcribed text}
```

Example:
```
Script_1_0001.wav|어서 오십시오 환상의 도서관에 오신 것을 진심으로 환영합니다 손님
Script_1_0003.wav|이 대목에서 잠시 깊은 숨을 들이마시고 문장의 향기를 느껴보시길 바랍니다
```

Rules:
- One line per WAV file
- Pipe `|` delimiter between filename and transcription
- Filename includes `.wav` extension, 4-digit zero-padded (`Script_N_LLLL.wav`)
- Sorted by filename (Script_1 → Script_2 → ... → Script_5, ascending line number)
- UTF-8 encoding
- No header line

## Technical Specification

| Parameter | Value |
|-----------|-------|
| Whisper model | `medium` |
| Language | `ko` (Korean) |
| Device | CUDA (RTX 3060 Ti) |
| Audio loading | `soundfile` + `librosa` resample to 16kHz (no ffmpeg dependency) |
| Temperature | Default fallback (DO NOT set temperature=0) |

## Source & Destination

```
Source WAVs:  datasets/PreviousVersion/cleansed/cleansed_wavs/  (1,719 files)
Output file:  datasets/PreviousVersion/cleansed/transcribed_script.txt
Answer key:   The original script sheet per session (Script_N_A0.txt)
```
