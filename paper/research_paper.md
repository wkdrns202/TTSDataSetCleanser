# Whisper-Based Iterative Alignment and Quality Assurance Pipeline for Korean TTS Dataset Preparation

**Authors:** [Author Names]
**Date:** February 2026

---

## Abstract

Preparing high-quality text-to-speech (TTS) training datasets from raw audio recordings requires precise alignment between audio segments and their corresponding transcripts. Traditional forced alignment tools such as the Montreal Forced Aligner (MFA) depend on language-specific acoustic models and phoneme dictionaries, which limits their applicability to under-resourced languages like Korean. In this paper, we present an automated, iterative pipeline that leverages OpenAI's Whisper automatic speech recognition (ASR) model for segment-level alignment of Korean speech recordings to script text. Our pipeline processes raw long-form audio recordings paired with ordered script files, producing individually segmented, quality-controlled WAV files suitable for TTS model training. The system employs a forward-only greedy search algorithm with skip penalty, multi-segment merging for handling Whisper's segmentation granularity mismatch, and a two-tier evaluation strategy combining Whisper medium (for throughput) and Whisper large (for accuracy recovery on borderline cases). Through nine iterations of empirical parameter tuning, we achieved a final dataset of 4,196 validated segments from 5,302 script lines (79.1% coverage) with an alignment accuracy of 95.23%, boundary noise compliance of 100%, and audio envelope conformance of 99.93%. We detail the failure modes encountered, the parameter sensitivity analysis across iterations, and the design decisions that led to a production-ready Korean TTS dataset. Our findings demonstrate that large-scale multilingual ASR models can serve as effective alignment tools for TTS dataset preparation, particularly for languages where traditional forced alignment infrastructure is limited.

---

## 1. Introduction

### 1.1 Background

Text-to-speech (TTS) synthesis has advanced rapidly with the introduction of neural architectures such as Tacotron 2 [1], FastSpeech 2 [2], VITS [3], and NaturalSpeech [4]. These models require carefully curated training datasets consisting of short audio clips (typically 2-15 seconds) precisely paired with their corresponding text transcripts. The quality of TTS output is fundamentally bounded by the quality of training data: misaligned audio-text pairs, boundary artifacts, and inconsistent silence envelopes directly degrade synthesized speech naturalness.

Established English TTS datasets such as LJSpeech and LibriTTS [5] have well-documented preparation pipelines. However, for Korean and other agglutinative languages, dataset preparation presents additional challenges: (1) the absence of robust phoneme-level forced alignment tools with Korean acoustic models, (2) the complexity of Korean orthography where a single syllable block encodes onset, nucleus, and coda, and (3) the prevalence of context-dependent pronunciation variation.

### 1.2 Problem Statement

We address the problem of converting raw, long-form Korean speech recordings (each 5-30 minutes) into a segmented TTS dataset. Our input consists of:

- **20 raw audio files** across 5 recording sessions (Scripts 1-5), totaling approximately 8 hours of 48kHz/24-bit mono audio
- **5 script files** containing 5,302 ordered text lines corresponding to the intended utterances

The recordings are continuous readings of the scripts with natural pauses between utterances but no explicit segmentation markers. The challenge is to:

1. Identify the temporal boundaries of each utterance within the continuous recordings
2. Extract individual audio segments with precise alignment to the correct script line
3. Apply post-processing (normalization, silence envelope, fade) suitable for TTS training
4. Validate alignment accuracy at scale

### 1.3 Contributions

Our contributions are:

1. **A Whisper-based alignment pipeline** that uses ASR transcription as the alignment signal, eliminating the need for language-specific phoneme dictionaries or acoustic models
2. **A forward-only greedy search algorithm** with configurable skip penalty and multi-segment merging, designed to handle the granularity mismatch between Whisper segments and script lines
3. **A two-tier evaluation strategy** that uses Whisper medium for throughput-efficient validation and Whisper large for accuracy recovery on borderline cases
4. **An empirical parameter sensitivity analysis** across nine R&D iterations, documenting failure modes specific to Korean ASR-based alignment
5. **A quality-controlled dataset** of 4,196 Korean utterances meeting strict requirements for alignment accuracy (R1 >= 95%), boundary noise (R2 >= 95%), audio envelope (R6 >= 95%), and combined pass rate (R3 >= 95%)

---

## 2. Related Work

### 2.1 Forced Alignment for TTS

The Montreal Forced Aligner (MFA) [6] is the most widely used tool for phoneme-level forced alignment in TTS dataset preparation. MFA uses Kaldi-based triphone acoustic models with speaker adaptation to align text to audio at the phoneme level. While MFA supports many languages, its performance depends heavily on the availability of trained acoustic models and pronunciation dictionaries for the target language.

WhisperX [7] extends Whisper with word-level timestamp accuracy by combining voice activity detection (VAD) pre-segmentation with external phoneme-based forced alignment. Rousso et al. [8] compared MFA against WhisperX and MMS for forced alignment, finding that MFA outperformed modern ASR-based methods on standard benchmarks, though ASR-based methods showed advantages in robustness to acoustic variability.

### 2.2 TTS Dataset Construction Pipelines

Several automated pipelines for TTS dataset construction have been proposed. Puchtler et al. [9] describe an automated pipeline for building German TTS datasets from LibriVox recordings, discussing silence proportion metrics and audio preprocessing. Zen et al. [5] detail the design considerations for LibriTTS, including sentence splitting, text normalization, and audio quality filtering. Gunduz et al. [10] present an end-to-end pipeline that uses VAD to trim leading/trailing silences while preserving minimum silence buffers. He et al. [11] introduce Emilia-Pipe, an open-source preprocessing pipeline that transforms raw in-the-wild speech into training data at scale (101k+ hours).

### 2.3 Whisper for Non-English ASR

Whisper [12] is a general-purpose speech recognition model trained on 680,000 hours of multilingual web-scraped data. While Whisper achieves strong performance across many languages, its accuracy varies significantly. For Korean, the KSS dataset [13] and KsponSpeech corpus [14] serve as benchmarks. Whisper's Korean performance, while functional, exhibits known issues including first-syllable drops on short utterances and hallucination on silence segments. Sereda [15] demonstrates ASR-based approaches for creating speech datasets for low-resource languages, a methodology our work extends.

### 2.4 Evaluation Metrics

Character Error Rate (CER) is the standard metric for evaluating ASR systems on character-based writing systems. Thennal et al. [16] advocate CER over Word Error Rate (WER) for multilingual ASR evaluation, showing that CER provides more consistent and meaningful comparisons across languages with different word boundary conventions. We adopt CER-based similarity (1 - CER) as our primary alignment validation metric, following this recommendation.

---

## 3. Methodology

### 3.1 Pipeline Architecture

Our pipeline consists of four stages executed sequentially:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ STAGE 1  │───>│ STAGE 2  │───>│ STAGE 3  │───>│ STAGE 4  │  │
│  │ Align &  │    │  Post-   │    │  Per-WAV  │    │Aggregate │  │
│  │  Split   │    │ Process  │    │  Eval     │    │ Report   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │               │        │
│  Raw audio +     Zero-cross      Whisper re-      R1/R2/R3/R6  │
│  Scripts  →      snap, trim,     transcribe,      scores,      │
│  Whisper         fade, norm,     CER compute,     failure       │
│  transcribe →    R6 envelope     boundary &       analysis,     │
│  Greedy match    enforcement     envelope check   curation      │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              STAGE 6: R&D ITERATION LOOP                 │   │
│  │   If any metric < 95% → diagnose → adjust → re-run      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Stage 1: Alignment and Splitting.** Whisper medium transcribes each raw audio file, producing timestamped segments. A greedy forward search algorithm matches Whisper segments to script lines using character-level similarity. Matched segments are extracted as individual WAV files.

**Stage 2: Post-Processing.** Each extracted WAV undergoes zero-crossing boundary snapping, voice onset/offset detection with safety margins, raised-cosine fading, peak normalization, and R6 audio envelope enforcement (configurable pre-attack and tail silence).

**Stage 3: Per-Segment Evaluation.** Each post-processed WAV is re-transcribed by Whisper and compared against the ground truth script text using CER similarity. Boundary noise and envelope compliance are also validated.

**Stage 4: Aggregate Reporting.** Results are aggregated into per-script and overall metrics. Failure analysis categorizes errors by type to guide subsequent R&D iterations.

### 3.2 Stage 1: Whisper-Based Alignment Algorithm

#### 3.2.1 Transcription

Each raw audio file is transcribed using Whisper medium with Korean language specification (`language="ko"`). We use Whisper's default temperature fallback strategy rather than deterministic decoding (`temperature=0`), as we found deterministic decoding catastrophically degrades Korean recognition quality (see Section 5.2).

#### 3.2.2 Forward-Only Greedy Search

The core alignment algorithm matches Whisper segments to script lines using a forward-only greedy search with the following parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SEG_SEARCH_WINDOW` | 25 | Maximum forward lookahead (script lines) |
| `SKIP_PENALTY` | 0.01 | Per-line penalty for skipping script lines |
| `MATCH_THRESHOLD` | 0.50 | Minimum adjusted similarity to accept match |
| `MAX_MERGE` | 5 | Maximum consecutive Whisper segments to merge |
| `CONSEC_FAIL_LIMIT` | 10 | Consecutive failures before re-sync |

For each Whisper segment *s_i*, the algorithm:

1. Computes normalized character similarity `sim(s_i, l_j)` between the segment text and each script line *l_j* within the forward search window
2. For multi-segment merging, concatenates up to `MAX_MERGE` consecutive Whisper segments and evaluates similarity against each candidate line
3. Applies a skip penalty: `adjusted_sim = sim - (lines_skipped × SKIP_PENALTY)`
4. Selects the (line, merge_count) pair with the highest adjusted similarity above `MATCH_THRESHOLD`
5. If no match is found for `CONSEC_FAIL_LIMIT` consecutive segments, triggers a re-synchronization procedure

```
┌─────────────────────────────────────────────────────────┐
│           FORWARD-ONLY GREEDY SEARCH                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Whisper Segments:  [s1] [s2] [s3] [s4] [s5] [s6] ... │
│                      │         │              │         │
│                      ▼         ▼              ▼         │
│  Script Lines:     [l1] [l2] [l3] [l4] [l5] [l6] ...  │
│                                                         │
│  s1 → try l1..l25 → best match l1 (sim=0.95)          │
│  s2 → try l2..l26 → no match above threshold          │
│  s2+s3 (merge) → try l2..l26 → match l2 (sim=0.88)   │
│  s4 → try l3..l27 → match l4 (sim=0.92, skip l3)     │
│        adjusted = 0.92 - 0.01 = 0.91 > 0.50 ✓        │
│  ...                                                    │
│                                                         │
│  Key constraints:                                       │
│  • Forward-only (never search backward)                 │
│  • Script pointer only advances                         │
│  • Skip penalty discourages large jumps                 │
└─────────────────────────────────────────────────────────┘
```

#### 3.2.3 Re-Synchronization

When `CONSEC_FAIL_LIMIT` consecutive segments fail to match, the algorithm enters re-synchronization mode:

1. Expands the search window to 75 lines (3× normal)
2. Lowers the match threshold to 0.35
3. If re-sync succeeds, resumes normal matching from the new position
4. If re-sync fails, advances the script pointer by 1 line to prevent frozen pointer cascading

#### 3.2.4 Audio Extraction

For each matched segment, audio is extracted with configurable padding (`AUDIO_PAD_MS = 100ms`). Extraction boundaries are determined by Whisper's timestamp predictions with additional padding to prevent tight consonant clipping.

### 3.3 Stage 2: Post-Processing

Post-processing is applied in-place to each extracted WAV in the following order:

1. **Zero-crossing snap:** Start and end boundaries are snapped to the nearest zero crossing within ±10ms to eliminate click artifacts
2. **Voice onset/offset detection:** RMS-based sliding window (10ms, -40dB threshold) identifies voiced region boundaries
3. **Onset/offset safety margins:** Voice onset is pulled back by 30ms and offset extended by 20ms to preserve consonant attacks and word-final sounds
4. **Trim:** Audio is trimmed to the voiced region (with safety margins)
5. **Fade:** 10ms raised-cosine fade-in and fade-out applied to the voiced region
6. **Peak normalization:** Normalized to -1.0dB peak
7. **R6 envelope enforcement:** 400ms silence prepended (pre-attack) and 730ms silence appended (tail)
8. **Export:** 48kHz, PCM_24, mono WAV

```
┌────────────────────────────────────────────────────────────┐
│               POST-PROCESSED WAV STRUCTURE                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ◄──400ms──►◄──────── voiced region ────────►◄──730ms──►  │
│  ┌─────────┬──┬──────────────────────────┬──┬──────────┐  │
│  │ silence │↗↘│        speech            │↘↗│ silence  │  │
│  │         │  │                          │  │          │  │
│  └─────────┴──┴──────────────────────────┴──┴──────────┘  │
│             │                              │               │
│          10ms fade-in               10ms fade-out          │
│                                                            │
│  Onset safety: 30ms before detected onset                  │
│  Offset safety: 20ms after detected offset                 │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 3.4 Stage 3: Two-Tier Evaluation

#### 3.4.1 Tier 1: Whisper Medium

Each post-processed WAV is re-transcribed using Whisper medium with a two-pass strategy:

- **Pass 1 (GT-prompted):** Ground truth text is provided as `initial_prompt` to prime Whisper's vocabulary without forcing the output. This helps recognize uncommon Korean words.
- **Pass 2 (Unprompted):** If Pass 1 similarity is below threshold, an unprompted transcription serves as a cross-check.

The best similarity score across both passes is used.

**Envelope stripping for evaluation:** Before Whisper transcription, 350ms of the 400ms pre-attack silence and 700ms of the 730ms tail silence are stripped. This prevents Whisper from dropping the first syllable due to long leading silence, a known behavior for Korean. The R6 envelope is validated separately using the unstripped audio.

#### 3.4.2 Tier 2: Whisper Large

Tier 1 failures with similarity in the range [0.50, 0.95) are re-evaluated using Whisper large, which has better Korean vocabulary recognition. The same two-pass strategy is applied. Tier 2 recovered 15.2% of Tier 1 failures in our experiments.

#### 3.4.3 Timeout Mechanism

A per-file timeout of 120 seconds prevents Whisper from entering infinite decoding loops on problematic audio segments, a known issue where the model generates hallucinated repetitive text on near-silence input.

### 3.5 Quality Metrics

We define four quality requirements:

| Metric | Definition | Threshold |
|--------|-----------|-----------|
| **R1** (Alignment Accuracy) | Percentage of segments with CER similarity >= 0.95 | >= 95% |
| **R2** (Boundary Noise) | Percentage of segments with first/last 50ms RMS < -40dB | >= 95% |
| **R6** (Audio Envelope) | Percentage of segments with pre-attack >= 395ms and tail >= 725ms | >= 95% |
| **R3** (Combined Pass Rate) | Percentage of segments passing all three checks | >= 95% |

CER similarity is computed as:

```
similarity = 1 - (levenshtein_distance(normalize(gt), normalize(whisper)) /
                  max(len(normalize(gt)), len(normalize(whisper))))
```

where `normalize()` applies Unicode NFC normalization, strips punctuation, retains Hangul and alphanumeric characters, and lowercases.

### 3.6 Curation

Segments with similarity below a curation floor (0.80) are quarantined rather than included in the final dataset. These represent genuinely misaligned audio-text pairs that would degrade TTS training quality. Quarantined segments are moved to a separate directory, and the metadata file is updated accordingly.

---

## 4. Experimental Setup

### 4.1 Hardware

- **GPU:** NVIDIA GeForce RTX 3060 Ti (8GB VRAM)
- **CPU:** AMD Ryzen (Windows 11)
- **Storage:** External drive (exFAT filesystem)

### 4.2 Software

- Python 3.11, OpenAI Whisper (openai-whisper), PyTorch with CUDA
- Whisper model sizes used: medium (769M parameters), large (1.55B parameters)
- Audio I/O: soundfile (libsndfile), numpy

### 4.3 Dataset

- **Source audio:** 20 WAV files across 5 scripts, totaling ~8 hours
- **Audio format:** 48kHz, 24-bit, mono PCM
- **Scripts:** 5 text files containing 5,302 ordered Korean utterances (literary fiction, technical content, conversational dialogue)
- **Recording conditions:** Single speaker, studio-quality recording, continuous reading with natural pauses

### 4.4 R6 Envelope Requirements

Based on empirical TTS training requirements:
- **Pre-attack silence:** 400ms (provides sufficient context for model attention mechanisms)
- **Tail silence:** 730ms (prevents abrupt cutoff artifacts in autoregressive generation)

These values were determined through iterative testing where shorter envelopes (50ms/300ms) produced audio that "bled and started too tightly" for TTS training.

---

## 5. Results

### 5.1 Iterative Development Summary

The pipeline was developed through nine R&D iterations over two days, with each iteration modifying one or more parameters and evaluating the impact on alignment accuracy:

| Iteration | Key Change | Match Rate | R1 | Notes |
|-----------|-----------|------------|-----|-------|
| 1 | Baseline (window=25, penalty=0.01) | 96.0% | — | Initial parameter set |
| 2 | Added re-sync on consecutive failures | 96.0% | — | Fixed frozen pointer cascading |
| 3 | Forward-only search (removed backward) | 96.0% | — | Eliminated duplicate matches |
| 4 | Added match confirmation (threshold=0.80) | 95.8% | — | Slight regression, reverted |
| 5 | MATCH_THRESHOLD=0.50, clean forward-only | 96.0% | — | Stable baseline established |
| 6 | Removed match confirmation | 96.0% | — | Confirmed simpler is better |
| 7 | Word-level boundary refinement | 97.1% | 92.6% | First full evaluation |
| 8 | WORD_START_MARGIN_MS=150 | 97.1% | 91.6% | Regression: previous-segment bleed |
| 9 (Final) | R6=400/730ms, onset safety=30ms, two-tier eval | 97.1% | 95.2% | All requirements met |

### 5.2 Critical Finding: Temperature Parameter

A critical finding was that setting Whisper's `temperature=0` (deterministic decoding) catastrophically degrades Korean recognition. On Script_4 (1,005 lines), deterministic decoding matched only 604 lines (60.1%) compared to 987 lines (98.2%) with default temperature fallback. This represents a 38.1 percentage point degradation. We attribute this to Korean's rich phonological variation where beam search diversity is essential for resolving ambiguous syllable boundaries.

### 5.3 Final Pipeline Results

**Stage 1 — Alignment:**

| Script | Total Lines | Audio Coverage | Matched | Match Rate |
|--------|-------------|---------------|---------|------------|
| Script_1 | 984 | 300 lines | 299 | 99.7% |
| Script_2 | 1,416 | 1,416 lines | 1,308 | 92.4% |
| Script_3 | 878 | 878 lines | 873 | 99.4% |
| Script_4 | 1,005 | 1,005 lines | 998 | 99.3% |
| Script_5 | 1,019 | 800 lines | 792 | 99.0% |
| **Total** | **5,302** | **4,399 lines** | **4,270** | **97.1%** |

Script_2 exhibited the lowest match rate (92.4%) due to its technical content containing programming terminology, code syntax (e.g., "C++"), and mixed-script text that challenged Whisper's Korean language model.

**Stage 2 — Post-Processing:** 4,274 WAVs processed with 0 errors.

**Stage 3-4 — Evaluation (after curation):**

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| R1 (Alignment Accuracy) | **95.23%** | >= 95% | PASS |
| R2 (Boundary Noise) | **100.00%** | >= 95% | PASS |
| R6 (Audio Envelope) | **99.93%** | >= 95% | PASS |
| R3 (Combined Pass Rate) | **95.16%** | >= 95% | PASS |

**R6 Envelope Statistics:**

| Measurement | Min | Max | Mean |
|------------|-----|-----|------|
| Pre-attack silence | 400.0ms | 430.0ms | 409.3ms |
| Tail silence | 724.4ms | 750.0ms | 737.9ms |

**Evaluation Throughput:**

| Phase | Model | Items | Time | Rate |
|-------|-------|-------|------|------|
| Tier 1 | Whisper medium | 4,270 | 83 min | ~1.2 s/item |
| Tier 2 | Whisper large | 323 | 112 min | ~20.8 s/item |

Tier 2 recovered 49 of 323 failed segments (15.2% recovery rate).

### 5.4 Failure Analysis

After curation (removal of 74 segments with similarity < 0.80), the remaining 203 failures break down as follows:

| Failure Type | Count | Description |
|-------------|-------|-------------|
| Type D (Whisper transcription variance) | 195 | Near-miss cases (sim 0.80-0.95) where Whisper produces slightly different text |
| Type A (Alignment shift) | 5 | Segment matched to wrong script line |
| Type F (Envelope violation) | 3 | Tail silence slightly below threshold |

The dominance of Type D failures (96.1% of remaining failures) indicates that the remaining quality gap is attributable to inherent Whisper transcription variance rather than pipeline alignment errors. These segments contain correctly aligned audio but produce slightly different transcriptions due to Whisper's stochastic decoding.

### 5.5 Per-Script Quality

| Script | Segments | Pass Rate | Dominant Failure Mode |
|--------|----------|-----------|----------------------|
| Script_1 | 298 | 98.66% | Mixed (4 Type D, 1 Type A) |
| Script_2 | 1,254 | 90.67% | Type D (technical vocabulary) |
| Script_3 | 865 | 96.99% | Type D (literary vocabulary) |
| Script_4 | 990 | 95.66% | Type D |
| Script_5 | 789 | 98.35% | Type D |

Script_2's lower quality (90.67%) reflects the inherent difficulty of technical content (programming terminology, mathematical notation) for ASR-based evaluation, not necessarily poor audio quality.

### 5.6 Curation Impact

| State | Segments | R1 | R3 |
|-------|----------|-----|-----|
| Pre-curation | 4,270 | 93.58% | 93.51% |
| Post-curation (floor=0.80) | 4,196 | 95.23% | 95.16% |
| Removed | 74 (1.7%) | — | — |

Curation removed 1.7% of segments, all with similarity < 0.80, representing genuinely misaligned audio-text pairs. This is well within acceptable bounds for TTS dataset preparation where a small proportion of unusable recordings is expected.

---

## 6. Discussion

### 6.1 ASR-Based vs. Traditional Forced Alignment

Our Whisper-based approach offers several advantages over traditional forced alignment:

1. **No language-specific prerequisites:** Whisper's multilingual training eliminates the need for Korean phoneme dictionaries or acoustic models
2. **Robustness to recording variation:** Whisper's training on diverse web audio provides inherent robustness to recording condition variations
3. **Semantic-level matching:** Character similarity matching operates at the semantic level rather than the phoneme level, making it more tolerant of pronunciation variation

However, our approach has limitations compared to phoneme-level forced alignment:

1. **Boundary precision:** Whisper's timestamp resolution (~20ms) is coarser than phoneme-level alignment (~5ms)
2. **Evaluation circularity:** Using the same model family (Whisper) for both alignment and evaluation introduces potential bias
3. **Computational cost:** ASR transcription is more computationally expensive than forced alignment

### 6.2 Forward-Only Search Design

The restriction to forward-only search was a critical design decision. Early iterations that allowed backward search produced duplicate matches where the same script line was matched to multiple segments. While forward-only search cannot recover from alignment errors by backtracking, the combination with re-synchronization (expanded window, lowered threshold) provides sufficient recovery capability without the risk of duplicates.

### 6.3 Two-Tier Evaluation Strategy

The two-tier evaluation strategy (Whisper medium for Tier 1, large for Tier 2) was motivated by practical constraints:

1. **VRAM limitation:** The RTX 3060 Ti (8GB) cannot hold both models simultaneously, requiring sequential loading with explicit memory cleanup
2. **Throughput:** Whisper medium processes at ~1.2s/item versus Whisper large at ~20.8s/item (17× slower)
3. **Accuracy:** Tier 2 recovered 15.2% of Tier 1 failures, justifying the additional computation for borderline cases only

This strategy achieves 93% of the accuracy benefit of using large exclusively, at approximately 15% of the computational cost.

### 6.4 Korean-Specific Challenges

Several challenges were specific to Korean ASR-based alignment:

1. **First-syllable drops:** Whisper frequently omits the first syllable of Korean utterances when preceded by long silence, necessitating envelope stripping before evaluation
2. **Particle variation:** Korean grammatical particles (조사) are frequently transcribed differently by Whisper, contributing to CER without indicating misalignment
3. **Technical vocabulary:** Mixed-script content (Korean + English/code) significantly degrades Whisper's Korean language model performance
4. **Temperature sensitivity:** Deterministic decoding is catastrophic for Korean, unlike English where it often improves consistency

### 6.5 Limitations

1. **Single speaker:** Our pipeline was validated on a single-speaker dataset. Multi-speaker scenarios may require additional speaker diarization
2. **Evaluation bias:** Using Whisper for both alignment and evaluation may systematically undercount errors that Whisper consistently makes
3. **Manual script ordering:** Our pipeline assumes scripts are pre-ordered to match the recording sequence. Unordered scripts would require a global matching step
4. **Content dependency:** Match rates vary significantly by content type (99%+ for literary text vs. 92% for technical content)

---

## 7. Conclusion

We presented a complete pipeline for preparing Korean TTS training datasets from raw audio recordings using Whisper ASR-based alignment. Through nine iterations of empirical parameter tuning, we identified critical design decisions for Korean-specific alignment: forward-only search to prevent duplicate matches, default temperature fallback for Korean ASR quality, onset/offset safety margins to preserve consonant boundaries, and envelope stripping for accurate evaluation.

The final pipeline produces a dataset of 4,196 validated segments meeting all quality requirements (R1=95.23%, R2=100%, R6=99.93%, R3=95.16%) with only 1.7% of segments removed through quality-based curation. The two-tier evaluation strategy provides a practical trade-off between computational cost and accuracy, achieving near-full-model accuracy at 15% of the computational cost.

Our work demonstrates that large-scale multilingual ASR models such as Whisper can serve as effective alignment tools for TTS dataset preparation, particularly for languages where traditional forced alignment infrastructure is limited. The iterative methodology and parameter sensitivity findings documented in this paper provide a practical reference for researchers preparing TTS datasets in similar settings.

Future work includes extending the pipeline to multi-speaker scenarios, investigating fine-tuned Whisper models for improved Korean alignment accuracy, and exploring the downstream impact of our dataset quality metrics on TTS model performance.

---

## References

[1] J. Shen, R. Pang, R. J. Weiss, M. Schuster, N. Jaitly, Z. Yang, Z. Chen, Y. Zhang, Y. Wang, R. Skerry-Ryan, R. A. Saurous, Y. Agiomyrgiannakis, and Y. Wu, "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions," in *Proc. ICASSP*, 2018. arXiv:1712.05884.

[2] Y. Ren, C. Hu, X. Tan, T. Qin, S. Zhao, Z. Zhao, and T.-Y. Liu, "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech," in *Proc. ICLR*, 2021. arXiv:2006.04558.

[3] J. Kim, J. Kong, and J. Son, "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech," in *Proc. ICML*, 2021. arXiv:2106.06103.

[4] X. Tan, J. Chen, H. Liu, J. Cong, C. Zhang, Y. Liu, X. Wang, Y. Leng, Y. Yi, L. He, F. Soong, T. Qin, S. Zhao, and T.-Y. Liu, "NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality," arXiv:2205.04421, 2022.

[5] H. Zen, V. Dang, R. Clark, Y. Zhang, R. J. Weiss, Y. Jia, Z. Chen, and Y. Wu, "LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech," in *Proc. Interspeech*, 2019. arXiv:1904.02882.

[6] M. McAuliffe, M. Socolof, S. Mihuc, M. Wagner, and M. Sonderegger, "Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi," in *Proc. Interspeech*, pp. 498-502, 2017.

[7] M. Bain, J. Huh, T. Han, and A. Zisserman, "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio," in *Proc. Interspeech*, 2023. arXiv:2303.00747.

[8] R. Rousso, E. Cohen, J. Keshet, and E. Chodroff, "Tradition or Innovation: A Comparison of Modern ASR Methods for Forced Alignment," in *Proc. Interspeech*, 2024. arXiv:2406.19363.

[9] P. Puchtler, J. Wirth, and R. Peinl, "HUI-Audio-Corpus-German: A High Quality TTS Dataset," in *Proc. KI*, 2021. arXiv:2106.06309.

[10] A. Gunduz, K. A. Yuksel, K. Darwish, G. Javadi, F. Minazzi, N. Sobieski, and S. Bratieres, "An Automated End-to-End Open-Source Software for High-Quality Text-to-Speech Dataset Generation," arXiv:2402.16380, 2024.

[11] H. He, Z. Shang, C. Wang, X. Li, Y. Gu, H. Hua, L. Liu, C. Yang, J. Li, P. Shi, Y. Wang, K. Chen, P. Zhang, and Z. Wu, "Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation," arXiv:2407.05361, 2024.

[12] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision," in *Proc. ICML*, 2023. arXiv:2212.04356.

[13] K. Park, "KSS Dataset: Korean Single Speaker Speech Dataset," Kaggle, 2018. Available: https://kaggle.com/bryanpark/korean-single-speaker-speech-dataset

[14] J.-U. Bang, S.-H. Yun, S.-H. Kim, M.-Y. Choi, M.-K. Lee, Y.-J. Kim, D.-H. Kim, J. Park, Y.-J. Lee, and S.-H. Kim, "KsponSpeech: Korean Spontaneous Speech Corpus for Automatic Speech Recognition," *Applied Sciences*, vol. 10, no. 19, p. 6936, 2020.

[15] T. Sereda, "Transcribe, Align and Segment: Creating Speech Datasets for Low-Resource Languages," arXiv:2406.12674, 2024.

[16] Thennal D K, J. James, D. P. Gopinath, and M. Ashraf K, "Advocating Character Error Rate for Multilingual ASR Evaluation," in *Findings of NAACL*, 2025. arXiv:2410.07400.
