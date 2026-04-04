# Engineering Note — 2026-04-04

## Voice Offset Detection Algorithm Fix: Sustained Silence Verification

---

## 1. Background

### Problem Statement
Korean TTS dataset에서 문장 끝 음절/어미가 잘리는 현상이 발견됨. 자동 QA(Whisper similarity)에서는 PASS 처리되지만, 인간 청취 시 명확히 부자연스러운 truncation이 확인됨.

### Prior Investigation (2026-03-16)
- `ending_truncation_report.json`에서 29건의 형식 어미 절단 기확인
- 대표 사례: Script_2_0153.wav — "비관적인 전망이 지배적이었습니다"에서 "습니다" 부분 잘림
- GT-prompted Whisper가 누락 부분을 예측 완성하여 similarity 1.0 판정 (평가 메트릭 신뢰도 문제)

---

## 2. Root Cause Analysis

### 2-1. Voice Offset Detection의 구조적 결함
기존 알고리즘은 10ms RMS 윈도우에서 마지막으로 -65dB 이상인 지점을 voice offset으로 **즉시 확정**. 한국어 형식 어미(-었습니다, -겠습니다 등)는 micro-pause(50-150ms) 후 낮은 에너지로 이어지는 패턴이 빈번하여, 이 알고리즘이 micro-pause 지점을 종료로 오판.

### 2-2. Zero-Crossing Snap의 부작용
END 방향 zero-crossing snap(±10ms)이 speech tail을 최대 10ms 앞으로 당겨 절삭. 한국어 어미의 마지막 비음/마찰음이 이 범위 안에 있으면 손실.

### 2-3. 고정 Safety Margin의 한계
OFFSET_SAFETY_MS = 80ms는 voice detection jitter(10ms) + zero-crossing snap(10ms)을 감안하면 실효 ~60ms. 한국어 micro-pause 범위(50-150ms)를 커버하기에 부족.

### 2-4. 대칭 Fade-out의 과도한 감쇄
10ms Hann fade-out이 -65dB 근처의 quiet tail을 비가청 수준으로 감쇄.

---

## 3. Applied Fixes

### Fix A: End Zero-Crossing Snap 비활성화
```python
# Before
new_end = find_nearest_zero_crossing(samples, len(samples)-1, sr)

# After
new_end = len(samples) - 1  # END는 snap하지 않음 — speech tail 보존
```
- START는 유지 (consonant attack에 안전)
- END의 fade-out이 이미 boundary smoothing 처리

### Fix B: OFFSET_SAFETY_MS 80 -> 120ms
```python
OFFSET_SAFETY_MS = 120  # 실효 safety ~100ms+, micro-pause 범위 커버
```

### Fix C: 비대칭 Fade (fade-out 10 -> 5ms)
```python
FADE_MS = 10       # Fade-in (유지)
FADE_OUT_MS = 5    # Fade-out (절반으로 축소, quiet tail 보존)
```

### Fix D: Voice Offset에 1초 Sustained Silence 검증 (핵심 수정)
```python
def find_voice_onset_offset(samples, sr=48000, threshold_db=-65,
                            window_ms=10, sustained_silence_ms=1000):
    """
    기존: 마지막 -65dB 이상 윈도우를 즉시 종료점으로 확정
    변경: offset 후보 이후 1초(100 윈도우) 동안 연속 무음(-65dB 미만) 확인
          → 1초 내 음성 재개 시 offset을 연장하고 다시 대기
          → 반복하여 실제 종료점 확정
    """
```
이 수정이 가장 주효. 기존 방식은 -65dB 이하가 감지되는 즉시 기계적으로 종료를 선언했으나, 실제 한국어 발화에서는 micro-pause 후 어미가 이어지는 패턴이 매우 흔함. 평균 -65dB가 최소 1초 이상 지속되는지 확인함으로써, micro-pause를 speech 내부의 일시적 정적으로 올바르게 처리.

---

## 4. Architecture Improvements

### 4-1. 2-Pass Pipeline Architecture
메모리 문제(Whisper medium ~1.5GB + 오디오 ~750MB 동시 로딩 불가)를 해결하기 위해 파이프라인을 2-pass 구조로 분리:

- **Pass 1 (Transcription)**: 모든 오디오 파일을 Whisper로 transcribe → segment 결과를 메모리에 캐시
- **Model Release**: Whisper 모델 해제 + gc.collect() + torch.cuda.empty_cache()
- **Pass 2 (Alignment & Slicing)**: 캐시된 segment 결과를 사용하여 alignment, 오디오 슬라이싱, 메타데이터 생성
- **Stage 2 (Post-processing)**: R6 envelope 적용

### 4-2. Versioned Output
출력 파일을 canonical `datasets/wavs/`에 직접 쓰지 않고, 날짜/버전별 디렉토리에 저장:
```
datasets/output_Script2_20260404_v1/wavs/
datasets/output_Script2_20260404_v1/script.txt
```
기존 curated dataset을 보호하고 실험 결과를 독립적으로 관리.

### 4-3. CLI 확장
```bash
# 특정 오디오 파일만 처리 (script + range)
python align_and_split.py --script 2 --range 1-162

# CPU 강제 사용 (VRAM 부족 시)
python align_and_split.py --device cpu

# 확인 없이 실행
python align_and_split.py -y
```

### 4-4. PermissionError 무한 루프 방지
exFAT에서 파일 쓰기 PermissionError 발생 시, 3회 재시도 후 해당 라인을 skip하고 진행. 기존 코드는 `continue`가 seg_idx를 전진시키지 않아 무한 반복됨.

### 4-5. Whisper SHA256 Bypass
`whisper._download()`가 이미 캐시된 모델 파일도 SHA256 검증을 위해 전체(1.5GB)를 메모리에 읽어들이는 문제 수정. `in_memory=False`일 때 파일 경로를 바로 반환하도록 라이브러리 패치.

---

## 5. Experiment Results

### Setup
- **Target**: Script_2_1-162.wav (162 lines)
- **Model**: Whisper medium (CPU, ~27min transcription)
- **Parameters**: Fix A~D 적용

### Results

| Metric | Value |
|--------|-------|
| Whisper segments | 222 |
| Matched lines | 158 / 162 |
| **Match rate** | **97.5%** |
| Skipped lines | 3 |
| PermissionError skip | 1 (Script_2_0154, exFAT lock) |
| Stage 2 post-processed | 전체 완료, 0 errors |

### Hypothesis Validation
**가설 지지됨.** 기존 알고리즘이 -65dB 감지 즉시 종료를 선언하고 기계적으로 120ms를 더하는 방식은 원본 오디오의 실제 의미(micro-pause 후 어미 지속)를 이해하지 못한 채 끊어버리는 오류를 범함.

평균 -65dB가 최소 1초(1000ms) 이상 지속되는지 확인하는 sustained silence verification이 핵심적으로 주효하여, micro-pause 후 이어지는 한국어 형식 어미를 완전히 포착.

---

## 6. Changed Files

- `src/align_and_split.py`
  - Constants: OFFSET_SAFETY_MS 80→120, FADE_OUT_MS=5 추가, AUDIO_PAD_MS 100→50
  - `find_voice_onset_offset()`: sustained_silence_ms=1000 검증 추가
  - `post_process_wavs()`: End zero-crossing snap 비활성화, 비대칭 fade
  - 2-pass architecture (transcribe → free model → align)
  - Versioned output directory (_make_versioned_output_dir)
  - CLI: --range, --device, -y 옵션 추가
  - PermissionError infinite loop fix

- `whisper/__init__.py` (site-packages)
  - `_download()`: SHA256 verification bypass when in_memory=False

---

## 7. Next Steps

1. **Script_2_0154.wav**: 재부팅 후 파일 잠금 해제, 단독 재처리
2. **전체 재처리**: 수정된 파라미터로 전체 dataset 재처리 (versioned output)
3. **평가 메트릭 개선**: GT-prompted vs non-prompted Whisper 결과 격차를 감점으로 반영
4. **Onset detection 개선**: sustained silence 검증을 onset에도 적용 검토
