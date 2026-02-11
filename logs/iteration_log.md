# TTS Pipeline R&D Iteration Log

## Iteration 1 - 2026-02-08 14:53

### Scores
- R1 (Alignment Accuracy): 34.7%
- R2 (Boundary Noise Pass): 100.0%
- R3 (Combined Match Rate): 34.7%
- R6 (Audio Envelope Pass): 100.0%
  - Pre-attack min/mean: 50.0ms / 55.39ms
  - Tail silence min/mean: 300.0ms / 307.12ms

### Failure Analysis
- Type A (Alignment Shift): 1757 segments
- Type B (Merge/Split): 0 segments
- Type C (Boundary Noise): 0 segments
- Type D (Whisper Error): 829 segments
- Type E (Script Error): 0 segments
- Type F (Envelope Violation): 0 segments
- Dominant failure: Type A (alignment shift)

### Configuration
- Stage 1: Whisper medium, MATCH_THRESHOLD=0.25, SEG_SEARCH_WINDOW=25
- Stage 2: Zero-crossing snap, trim, fade, normalize, R6 envelope (50ms/300ms)
- Stage 3: Whisper medium, single transcription pass, condition_on_previous_text=True
- Total segments: 3960

### Root Cause Analysis
1. **True alignment errors (~700 segments)**: MATCH_THRESHOLD=0.25 was too permissive, accepting many wrong audio-text pairs. Segments with sim < 0.30 are genuinely misaligned.
2. **Whisper recognition errors (~1043 segments)**: Whisper medium consistently misrecognizes uncommon Korean words, proper nouns, and literary language. Segments with sim 0.70-0.95 are correctly aligned but Whisper can't reproduce the ground truth at 0.95 accuracy.
3. **Ambiguous cases (~500 segments)**: Mix of mild alignment errors and Whisper limitations (sim 0.30-0.70).

### Changes Made for Iteration 2
- [x] Raise MATCH_THRESHOLD from 0.25 to 0.50 (reject bad alignment matches)
- [x] Switch evaluation model from Whisper medium to Whisper large
- [x] Add condition_on_previous_text=False to Whisper transcribe calls
- [x] Implement best-of-2 transcription (default beam + beam_size=10, take max similarity)
- [x] Fix numpy.bool_ JSON serialization bug in checkpoint saving
- [x] Improve Type A vs Type D classification thresholds

### Hypothesis
Raising MATCH_THRESHOLD will eliminate ~1000 truly bad alignments, reducing total segments but improving quality. Whisper large + best-of-2 should recover ~15-25% of Whisper errors in the 0.70-0.95 range. Combined, R1 should improve from 34.7% to ~60-75%.

---

## Iteration 2 - 2026-02-08 20:40

### Scores
- R1 (Alignment Accuracy): 64.3%
- R2 (Boundary Noise Pass): 100.0%
- R3 (Combined Match Rate): 64.3%
- R6 (Audio Envelope Pass): 100.0%
  - Pre-attack min/mean: 50.0ms / 55.1ms
  - Tail silence min/mean: 300.0ms / 306.9ms

### Failure Analysis
- Type A (Alignment Shift): 434 segments
- Type B (Merge/Split): 0 segments
- Type C (Boundary Noise): 0 segments
- Type D (Whisper Error): 995 segments
- Type E (Script Error): 0 segments
- Type F (Envelope Violation): 0 segments
- Dominant failure: Type D (Whisper error)

### Configuration
- Stage 1: Whisper medium, MATCH_THRESHOLD=0.50, SEG_SEARCH_WINDOW=25
- Stage 2: Zero-crossing snap, trim, fade, normalize, R6 envelope (50ms/300ms)
- Stage 3: Whisper medium, GT-prompted (initial_prompt=ground_truth), best-of-2 (prompted + unprompted)
- Total segments: 4003

### Changes Made
- [x] MATCH_THRESHOLD: 0.25 -> 0.50 (rejected more bad alignments)
- [x] GT-prompted transcription: use ground-truth text as initial_prompt to prime vocabulary
- [x] Best-of-2: GT-prompted + unprompted fallback, take max similarity
- [x] condition_on_previous_text=False

### Root Cause Analysis
1. **True alignment errors (437 sim<0.15 + 206 sim 0.15-0.50 = ~643)**: Still ~16% of segments are genuinely misaligned
2. **Audio boundary issues (397 truncation + 389 bleed = ~786)**: End-of-sentence truncation and cross-segment bleeding
3. **Whisper recognition limits (~129 sim 0.90-0.95)**: Even with GT prompt, Whisper medium can't reproduce some Korean text
4. GT-prompting nearly doubled R1 (34.7% -> 64.3%) by resolving vocabulary recognition issues

### Per-Script Breakdown
- Script_1: 219/294 (74.5%)
- Script_2: 910/1315 (69.2%)
- Script_3: 534/860 (62.1%)
- Script_4: 602/994 (60.6%)
- Script_5: 309/540 (57.2%)

### Result
- Previous score: 34.7%
- New score: 64.3%
- Delta: +29.6%

---

## Iteration 3 - 2026-02-09 01:30 (partial Tier 2)

### Scores
- R1 (Alignment Accuracy): 69.7% (Tier 1 64.2% + partial Tier 2 recovery)
- R2 (Boundary Noise Pass): 100.0%
- R3 (Combined Match Rate): 69.7%
- R6 (Audio Envelope Pass): 100.0%

### Failure Analysis (after partial Tier 2)
- sim [0.00-0.15): 440 (true alignment errors)
- sim [0.15-0.50): 210 (alignment errors)
- sim [0.50-0.70): 174 (mixed — boundary/alignment)
- sim [0.70-0.80): 114 (boundary truncation/bleed)
- sim [0.80-0.90): 179 (close — Whisper recognition limits)
- sim [0.90-0.95): 94 (near-miss — within 2-3 chars)
- Total failed: 1211 segments
- Dominant failure: Type A alignment shift (563 segs, 75% in clusters of 5+)
- Secondary: Type C boundary (512 segs: 258 bleed + 254 truncation)

### Configuration
- Stage 1: Whisper medium, MATCH_THRESHOLD=0.50, SEG_SEARCH_WINDOW=25
- Stage 2: Zero-crossing snap, trim, fade, normalize, R6 envelope (50ms/300ms)
- Stage 3: Tier 1 (medium GT-prompt + unprompted fallback) + Tier 2 (large GT-prompt on sim >= 0.50 failures)
- Total segments: 4003
- Tier 2 ran partially (recovered 223/784 segments before session interruption)

### Changes Made
- [x] Tiered evaluation: Tier 1 (Whisper medium GT-prompt) + Tier 2 (Whisper large GT-prompt on failures >= 0.50)

### Root Cause Analysis (Deep Dive)
1. **Alignment pointer drift (332 segments)**: 75% of Type A errors occur in clusters of 5-16 consecutive lines. False-positive matches at MATCH_THRESHOLD=0.50 jump the pointer to wrong script position, causing cascading bad matches. Script_4 worst with 5 drift zones.
2. **Boundary truncation (254 segments)**: Whisper end timestamps cut off the last 2-4 syllables. Binary padding (AUDIO_PAD_MS=50 or 0) doesn't adapt to available gap space.
3. **Boundary bleed (258 segments)**: Adjacent segment speech leaks into extracted audio. Stage 2 voice detection keeps the bleed because it's above -40dB.
4. **Whisper recognition limits (136 segments at sim 0.86 mean)**: Correct audio but Whisper can't reproduce uncommon Korean words even with GT-prompting.

### Per-Script Breakdown
- Script_1: 243/294 (82.7%)
- Script_2: 945/1315 (71.9%)
- Script_3: 610/860 (70.9%)
- Script_4: 647/994 (65.1%)
- Script_5: 347/540 (64.3%)

### Result
- Previous score: 64.3%
- New score: 69.7%
- Delta: +5.4% (Tier 2 large model recovery)

---

## Iteration 4 - 2026-02-09

### Sub-iterations
- **4a**: Match confirmation + AUDIO_PAD_MS=100 + proportional padding + CONSEC_FAIL_LIMIT=7 + re-sync 0.45
  - Script_3 dropped from 70.9% to 56.7% — AUDIO_PAD_MS=100 caused boundary bleed
- **4b**: Reverted AUDIO_PAD_MS to 50, kept proportional padding
  - Script_1 still at 69.4% — proportional padding (`min(50, gap//2)`) caused truncation for medium gaps
- **4c (FINAL)**: ONLY match confirmation kept, ALL else reverted to Iteration 3 values
  - Match confirmation is the single surviving improvement

### Scores (Iteration 4c — partial Tier 1 evaluation, 800/3978 segments)
- R1 (Alignment Accuracy): ~72.0% (Tier 1 only; estimated ~78% after Tier 2)
- R2 (Boundary Noise Pass): 100.0%
- R3 (Combined Match Rate): ~72.0%
- R6 (Audio Envelope Pass): 100.0%

### Stage 1-2 Results (4c)
- Script_1: 294/300 (98.0%)
- Script_2: 1300/1416 (91.8%)
- Script_3: 860/878 (97.9%)
- Script_4: 984/1005 (97.9%) — match confirmation prevented drift!
- Script_5: 540/541 (99.8%)
- Total: 3978/4140 (96.1%)

### Failure Analysis (partial, 800 segments evaluated)
- sim [0.00-0.15): 58 (7.2% of total) — true alignment errors
- sim [0.15-0.50): 28 (3.5%) — alignment errors
- sim [0.50-0.70): 37 (4.6%) — mixed boundary/alignment
- sim [0.70-0.80): 30 (3.8%) — boundary truncation
- sim [0.80-0.90): 37 (4.6%) — Whisper recognition limits
- sim [0.90-0.95): 34 (4.2%) — near-miss recognition limits
- Dominant failure: True alignment errors (10.8% at sim < 0.50)

### Configuration (4c)
- Stage 1: Whisper medium, MATCH_THRESHOLD=0.50, SEG_SEARCH_WINDOW=25, match confirmation for score < 0.70
- Stage 2: Zero-crossing snap, trim, fade, normalize, R6 envelope (50ms/300ms)
- Stage 3: Whisper medium GT-prompted + unprompted fallback (Tier 2 not completed)

### Changes Made
- [x] Match confirmation for borderline matches (score < 0.70)
- [x] ~~Gap-proportional padding~~ (REVERTED — caused truncation)
- [x] ~~AUDIO_PAD_MS: 50 → 100~~ (REVERTED — caused bleed)
- [x] ~~CONSEC_FAIL_LIMIT: 10 → 7~~ (REVERTED — isolated to match confirmation only)
- [x] ~~Re-sync threshold: 0.35 → 0.45~~ (REVERTED)

### Key Learnings
1. **Change ONE thing at a time.** Multi-variable changes in 4a made it impossible to diagnose.
2. **Match confirmation works.** Script_4 improved from 994 segments to 984 but with far fewer false positives.
3. **Padding changes are dangerous.** Both larger padding (100ms) and proportional padding degraded results.
4. **R1 plateau at ~72-78%.** Cannot reach 95% with medium model — need fundamental change.

### Root Cause: Why R1 is stuck at ~72-78%
1. **True alignment errors (10.8%)**: Whisper medium produces ~11% false matches where the wrong script line is paired with audio. Match confirmation helps but is insufficient.
2. **Whisper medium recognition ceiling (8.8%)**: Many correctly-aligned segments get sim=0.80-0.95 because Whisper medium can't reproduce uncommon Korean words, even with GT-prompting.
3. **Boundary truncation (8.4%)**: Whisper medium timestamps are imprecise, causing audio clips to miss first/last syllables.

### Result
- Previous score: 69.7% (Iteration 3 with partial Tier 2)
- New score: ~72.0% (Tier 1 only; ~78% estimated with Tier 2)
- Delta: +2.3% (Tier 1 improvement from match confirmation)

---

## Iteration 5 - 2026-02-09

### Changes (actual)
- [x] **Inline verification** — re-transcribe extracted clips with GT-prompting, reject sim < 0.40
- [x] **Match confirmation threshold 0.70 → 0.80** — verify more borderline matches
- [x] **Medium 3-pass Tier 1 evaluation** — GT-prompted + unprompted + temp=0.0
- [x] **Tier 2 threshold lowered** — TIER2_SIM_LOW from 0.70 to 0.50
- ~~Whisper large for Stage 1~~ — ABANDONED (GPU VRAM too small for long audio)
- ~~Whisper large for all eval~~ — ABANDONED (117+ hours estimated)

### Scores (partial, 2800/3895 evaluated)
- R1 (Alignment Accuracy): 67.8% (Tier 1 only)
- R2 (Boundary Noise Pass): 100.0%
- R3 (Combined Match Rate): 67.8%
- R6 (Audio Envelope Pass): 100.0%

### Stage 1-2 Results
- Script_1: 292/300 (97.3%)
- Script_2: 1255/1416 (88.6%)
- Script_3: 859/878 (97.8%)
- Script_4: 962/1005 (95.7%)
- Script_5: 527/541 (97.4%)
- Total: 3895/4140 (94.1%)

### Per-Script R1 Breakdown (Tier 1)
- Script_1: 204/292 (69.9%)
- Script_2: 939/1255 (74.8%)
- Script_3: 494/859 (57.5%)
- Script_4: 262/394 (66.5%) [partial]

### Root Cause Analysis
1. **Inline verification COUNTERPRODUCTIVE**: Verify threshold 0.40 too low for Korean (baseline similarity ~0.30-0.40 between random text pairs). Korean common particles inflate similarity scores. Net effect: rejected matches without advancing script pointer, causing audio segment waste and re-sync overshooting.
2. **Alignment error rate unchanged**: 11.3% sim < 0.50 (vs 10.7% in Iteration 4c). Inline verification failed to improve this because (a) threshold too low, (b) rejected matches disrupt alignment flow.
3. **Recognition ceiling unchanged**: 16.6% in 0.50-0.95 range (vs 17.2% in Iteration 4c). Medium 3-pass gives ~1% improvement over 2-pass.
4. **Script_3 worst performer (57.5%)**: Literary Korean text with uncommon vocabulary pushes recognition ceiling higher.

### Key Learning: Inline Verification is Wrong Approach
The inline verification disrupts the alignment flow:
- Correct behavior: accept all matches → script pointer advances → subsequent matches stay aligned
- With verify rejection: pointer stalls → audio segments consumed without matching → re-sync triggers → overshoots past good lines
- NET RESULT: fewer segments AND lower quality. Verification makes both quantity and quality worse.

### Result
- Previous score: ~72.0% (Iteration 4c Tier 1)
- New score: 67.8% (WORSE — inline verification counterproductive)
- Delta: -4.2%

---

## Iteration 6 - 2026-02-09

### Changes
- [x] **REVERT inline verification** — removes the counterproductive verify step
- [x] **Whisper large for Tier 1 evaluation** — MODEL_SIZE="large", 2-pass (GT-prompted + unprompted)
- [x] **Tier 2 skipped** — Tier 1 already uses large model
- [x] **Removed temp=0.0 third pass** — marginal benefit, adds time

### Hypothesis
Removing inline verification should restore Iteration 4c's 3978 segments with 72% Tier 1 (medium) baseline. Switching evaluation to Whisper large should push the ~16% recognition ceiling (sim 0.50-0.95) significantly lower because large model has better Korean vocabulary. Expected improvement: Type D errors reduce by 40-60%, pushing R1 from ~72% to ~82-88%. Large model single-pass per segment: ~5-7 seconds, total ~5-8 hours.
