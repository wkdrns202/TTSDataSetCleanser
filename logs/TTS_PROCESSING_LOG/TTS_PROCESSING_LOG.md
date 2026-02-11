# TTS Dataset Labeling — Processing Log

> **Project:** Korean TTS Dataset Pipeline (Soft Story)
> **Started:** 2026-02-08
> **Owner:** Young Jae
> **Pipeline Spec:** `TTS_DATASET_PIPELINE_REQUIREMENTS.md`
> **Status:** 🔄 In Progress

---

## Current Objectives

Two parallel workstreams are running simultaneously:

| Track | Goal | Status |
|---|---|---|
| **Track A** | Validate & sort the latest batch of labeled TTS data — determine which existing WAV segments are usable vs. must be re-processed | 🔄 In Progress |
| **Track B** | Build a new pipeline with pre-attack padding & tail silence guards to prevent over-tight cuts that cause sentence bleeding in output audio | 🔄 In Progress |

---

## Track A — Validate & Sort Existing Results

### Problem Statement

The latest round of TTS data labeling produced WAV segments, but their quality has not been systematically verified. Before using any of these files for TTS training, we need to confirm that each segment's audio actually matches its paired script line — and separate the usable files from the ones that need re-processing.

### Approach

1. Run Whisper transcription on every existing WAV segment.
2. Compare each transcription against the ground-truth script line (character-level similarity, Korean-normalized).
3. Classify each segment:
   - **PASS (≥ 95% similarity)** → move to `validated/usable/`
   - **MARGINAL (85–94%)** → move to `validated/review/` — may be recoverable with boundary adjustments
   - **FAIL (< 85%)** → move to `validated/reject/` — must be re-processed in Track B
4. Generate a validation report with per-segment scores and aggregate statistics.

### Decisions Made

| # | Decision | Rationale |
|---|---|---|
| 1 | Use the same Whisper model (large-v3) for validation as will be used in the new pipeline | Ensures consistency — a segment that passes validation will also pass the new pipeline's quality gate |
| 2 | Korean text normalization: strip punctuation, NFC normalize, collapse whitespace | Prevents false negatives from trivial formatting differences |
| 3 | Three-tier classification (PASS / MARGINAL / FAIL) instead of binary | MARGINAL segments may only need boundary re-trimming, not full re-alignment — saves processing time |

### Progress

- [ ] Inventory all existing WAV segments and script files
- [ ] Run batch Whisper transcription
- [ ] Compute similarity scores
- [ ] Sort into PASS / MARGINAL / FAIL directories
- [ ] Generate validation report
- [ ] Determine re-processing volume for Track B

### Blockers

- *(none yet — update as issues arise)*

---

## Track B — New Pipeline with Boundary Padding

### Problem Statement

The previous slicing logic cut segments too tightly at sentence boundaries. This caused two critical issues:

1. **Sentence bleeding** — the beginning or end of an adjacent sentence leaks into the current segment, corrupting the transcript match.
2. **Clipped speech** — the first or last phoneme of the intended sentence is partially cut off, making the audio sound unnatural for TTS training.

Both issues stem from the same root cause: insufficient padding around the alignment timestamps before applying silence trimming.

### Key Design Change: Pre-Attack & Tail Silence Guards

```
BEFORE (old logic):
  cut exactly at [start_sec, end_sec] → too tight → bleeding/clipping

AFTER (new logic):
  1. Add pre-attack padding (configurable, default 80–150ms before start_sec)
  2. Add tail silence padding (configurable, default 80–150ms after end_sec)
  3. Within the padded region, detect actual speech onset/offset using:
     - Energy-based voice activity detection
     - Zero-crossing rate analysis
  4. Snap cut points to the detected speech boundaries (not the raw timestamps)
  5. Apply fade-in/fade-out envelope
  6. Trim only true silence (RMS < threshold), never speech
```

### Why This Matters

The pre-attack guard ensures we never cut into the onset of a consonant (especially critical for Korean plosives like ㄱ, ㄷ, ㅂ, ㅈ which have short burst durations). The tail silence guard ensures the final syllable's release and any natural trailing resonance are fully captured before trimming.

### Technical Parameters (Initial Values — Subject to R&D Tuning)

| Parameter | Initial Value | Range to Explore | Notes |
|---|---|---|---|
| `pre_attack_padding_ms` | 100 | 50–200 | Time added before alignment start |
| `tail_silence_padding_ms` | 120 | 50–200 | Time added after alignment end |
| `silence_threshold_db` | -40 | -50 to -30 | RMS threshold for silence detection |
| `fade_in_ms` | 10 | 5–30 | Raised-cosine fade-in duration |
| `fade_out_ms` | 15 | 5–30 | Raised-cosine fade-out duration |
| `zero_crossing_snap_ms` | 10 | 5–20 | Window for snapping cuts to zero-crossings |
| `min_segment_duration_ms` | 300 | 200–500 | Reject segments shorter than this |
| `max_segment_duration_ms` | 15000 | 10000–20000 | Flag segments longer than this for review |

### Decisions Made

| # | Decision | Rationale |
|---|---|---|
| 1 | Default pre-attack padding = 100ms | Korean speech onset (especially aspirated consonants) can have 30–60ms VOT; 100ms gives safe margin |
| 2 | Default tail padding = 120ms | Slightly longer than pre-attack because sentence-final vowel lengthening is common in Korean |
| 3 | Energy-based VAD as primary, zero-crossing as secondary | Energy catches most cases; zero-crossing helps with voiceless consonants that have low energy but high ZCR |
| 4 | Implement the padding logic in `align_and_split.py` as the central module | This is the file identified in the requirements doc as the most critical — all boundary logic lives here |
| 5 | Iterative R&D loop per `TTS_DATASET_PIPELINE_REQUIREMENTS.md` Section 7 | Pipeline auto-repeats until 95% match rate is achieved or 10 iterations exhausted |

### Progress

- [ ] Implement pre-attack & tail silence padding in `align_and_split.py`
- [ ] Integrate energy-based VAD for speech onset/offset detection
- [ ] Add zero-crossing snap logic
- [ ] Run first pipeline iteration on test subset
- [ ] Evaluate scores (target: R1 ≥ 95%, R2 ≥ 95%, R3 ≥ 95%)
- [ ] If score < 95%: enter STAGE 6 R&D loop (auto)
- [ ] Finalize when all requirements met

### Blockers

- *(none yet — update as issues arise)*

---

## R&D Iteration History

> Each iteration of the autonomous R&D loop gets logged here with scores and changes.

*(Entries will be appended as the pipeline runs — see template below)*

### Template

```
### Iteration N — [YYYY-MM-DD HH:MM]

**Scores:**
- R1 (Alignment Accuracy): ___%
- R2 (Boundary Noise Pass): ___%
- R3 (Combined Match Rate): ___%

**Failure Breakdown:**
- Type A (Alignment shift): ___
- Type B (Merge/Split error): ___
- Type C (Boundary noise): ___
- Type D (Whisper hallucination): ___
- Type E (Script mismatch): ___

**Root Cause:** ...

**Changes Applied:** ...

**Hypothesis:** ...

**Result:** Previous ___% → New ___% (Δ ±___%)
```

---

## Architecture Notes

### Relationship Between Track A and Track B

```
Track A (Validate Existing)          Track B (New Pipeline)
         │                                    │
         ▼                                    ▼
  Sort existing WAVs              Build improved pipeline
  into PASS/MARGINAL/FAIL         with boundary padding
         │                                    │
         │    FAIL + MARGINAL segments        │
         └──────────────────┬─────────────────┘
                            ▼
              Re-process through Track B pipeline
                            │
                            ▼
                  Final validated dataset
```

Track A's FAIL and MARGINAL segments become input for Track B's pipeline. PASS segments from Track A are carried forward directly. This means Track A must finish (or at least produce the sort results) before Track B processes the re-do batch.

---

## Next Steps

1. **Immediate:** Complete Track A validation scan to understand the current quality baseline.
2. **Immediate:** Implement the pre-attack/tail silence padding in `align_and_split.py`.
3. **After Track A results:** Quantify how many segments need re-processing — this determines Track B's workload.
4. **Ongoing:** Monitor the R&D iteration loop scores and intervene if the pipeline plateaus.

---

## Open Questions

- [ ] What Whisper model size is optimal for Korean accuracy vs. speed tradeoff? (large-v3 is accurate but slow; medium might be sufficient for validation passes)
- [ ] Should MARGINAL segments (85–94%) be re-aligned from scratch, or just boundary-adjusted?
- [ ] Is the current script file 100% accurate, or are there known typos/errors that should be corrected before scoring?

---

*Last updated: 2026-02-08*
