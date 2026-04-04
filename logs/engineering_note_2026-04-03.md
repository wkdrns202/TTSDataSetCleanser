# Engineering Note — 2026-04-03

## Tail Truncation 결함 수정 및 Voice Offset 감지 알고리즘 개선

---

## 1. 발견된 문제

### 현상
- Script_2_0153.wav에서 **인간 청취 시 끝음절/어미 잘림** 확인
- 자동 QA(validation_results.csv)에서는 **similarity 1.0, verdict PASS** 처리됨
- ending_truncation_report.json에서 **29건의 형식 어미 절단** 기확인 (2026-03-16)

### 구체적 사례
```
Script_2_0153.wav
  GT:      "...비관적인 전망이 지배적이었습니다"
  실제 청취: 마지막 "습니다" 부분이 잘리거나 부자연스럽게 끊김
  자동 QA:  similarity 1.0 (PASS) — GT-prompted Whisper가 누락 부분을 예측 완성
```

---

## 2. 이전 알고리즘 (수정 전)

### Stage 2 Post-Processing (align_and_split.py)

```
Step 1: Zero-crossing snap — START/END 모두 ±10ms에서 가장 가까운 영점 교차로 이동
Step 2: Voice onset/offset 감지 — 10ms RMS 윈도우, -65dB 기준
        offset = "마지막으로 -65dB 이상인 윈도우"의 끝 sample
Step 3: Safety margin — onset -= 30ms, offset += 80ms
Step 4: Fade — 10ms raised-cosine (양쪽 동일)
Step 5: Peak normalize → Step 6: R6 envelope (400ms + 730ms) → Export
```

### 파라미터
```python
OFFSET_SAFETY_MS = 80        # offset 연장
FADE_MS = 10                 # fade-in/out 동일
# Zero-crossing: START와 END 모두 ±10ms snap
# Voice offset: 마지막 -65dB 지점 즉시 확정 (지속 무음 검증 없음)
```

---

## 3. 문제점 분석

### 3-1. Zero-crossing snap이 END에서 speech tail 절삭
- `find_nearest_zero_crossing(samples, len(samples)-1, sr)` → ±10ms 범위에서 영점 교차로 이동
- END 방향으로는 최대 10ms **앞으로** 당겨질 수 있음
- 한국어 어미의 마지막 비음/마찰음이 이 10ms 안에 있으면 절삭됨

### 3-2. Voice offset 감지에 지속 무음 검증 부재
- `voiced[-1]` (마지막 -65dB 이상 윈도우)를 즉시 종료점으로 확정
- micro-pause 후 이어지는 낮은 에너지 음절(-65dB 근처)을 놓침
- 한국어 형식 어미 "-었습니다"의 "습니다" 부분이 micro-pause 후 낮은 에너지로 발화되는 패턴에 취약

### 3-3. OFFSET_SAFETY_MS 80ms 부족
- voice detection 10ms 지터 + zero-crossing snap 10ms = 실효 safety ~60ms
- 한국어 micro-pause (50-150ms) 범위를 커버하기에 부족

### 3-4. 대칭 Fade-out 10ms → quiet tail 감쇄
- 10ms Hann fade-out이 -65dB 근처의 quiet tail을 비가청 수준으로 감쇄
- Fade-in은 consonant attack에 안전하지만 fade-out에는 과도

### 3-5. GT-prompted 평가가 절단을 마스킹
- Whisper는 initial_prompt로 GT 텍스트를 받으면, 실제로 들리지 않는 부분도 예측 완성
- 이로 인해 similarity 1.0 판정 → 절단이 자동 QA를 통과
- **평가 메트릭 자체의 신뢰도 문제**

---

## 4. 적용한 수정 사항

### Fix A: End zero-crossing snap 비활성화
```python
# 변경 전 (line 455)
new_end = find_nearest_zero_crossing(samples, len(samples) - 1, sr)

# 변경 후
new_end = len(samples) - 1  # END는 snap하지 않음 — speech tail 보존
```
- START는 유지 (consonant attack에 안전)
- END의 fade-out이 이미 boundary smoothing 처리

### Fix B: OFFSET_SAFETY_MS 80 → 120ms
```python
# 변경 전 (line 90)
OFFSET_SAFETY_MS = 80

# 변경 후
OFFSET_SAFETY_MS = 120
```
- 실효 safety 100ms+ → 한국어 micro-pause 범위 커버
- Stage 1 추출 범위 내에서만 작동 → cross-segment bleed 위험 없음

### Fix C: 비대칭 Fade (fade-out 10 → 5ms)
```python
# 변경 전 (line 82)
FADE_MS = 10  # 양쪽 동일

# 변경 후
FADE_MS = 10       # Fade-in (유지)
FADE_OUT_MS = 5    # Fade-out (절반으로 축소)
```
- post_process_wavs()에서 fade_in_samples / fade_out_samples 분리 적용

### Fix D: Voice offset에 1초 지속 무음 검증 추가
```python
# find_voice_onset_offset()에 sustained_silence_ms=1000 파라미터 추가

# 이전 로직:
#   offset = 마지막 -65dB 이상 윈도우 → 즉시 확정

# 변경 로직:
#   offset 후보 설정 → 뒤로 1초(100 윈도우) 스캔
#   → 1초간 연속 무음(-65dB 미만) 확인 → 종료 확정
#   → 1초 내 음성 재개 발견 → offset을 거기로 연장 → 다시 1초 대기
#   → 반복하여 실제 종료점 확정
```

---

## 5. 가설 및 기대효과

### 가설
1. Script_2_0153 등의 끝음절 잘림은 **micro-pause 후 낮은 에너지 어미**를 voice offset 감지가 놓친 것이 주 원인
2. Zero-crossing END snap과 짧은 OFFSET_SAFETY가 이를 악화시킴
3. GT-prompted Whisper가 빠진 어미를 예측 완성하여 자동 QA를 통과시킴

### 기대효과

| Fix | 기대효과 | 리스크 |
|-----|---------|--------|
| A (End ZC snap 비활성화) | speech tail 최대 10ms 복구 | 최소 — fade-out이 boundary 처리 |
| B (OFFSET 80→120ms) | 실효 safety 60→100ms+, micro-pause 커버 | 낮음 — Stage 1 범위 내 |
| C (Fade-out 10→5ms) | quiet tail 감쇄 구간 절반 축소 | 최소 |
| D (1초 지속 무음 검증) | micro-pause 후 이어지는 어미 완전 포착 | 중간 — 전체 처리 시간 소폭 증가 |

### 향후 과제
- **평가 신뢰도**: Non-prompted Whisper 전사 결과와 GT-prompted 결과의 격차를 신뢰도 감점으로 반영하는 메트릭 필요 (Phase 3 Selective Composer의 P3/P5 스코어로 구현 예정)
- **재현성 검증**: 수정된 파라미터로 새 raw 데이터에서 95% 유지되는지 확인 (Phase 2)

---

## 6. 수정 대상 파일

- `src/align_and_split.py`
  - Line 82-83: FADE_MS, FADE_OUT_MS 상수
  - Line 90: OFFSET_SAFETY_MS
  - Lines 332-389: find_voice_onset_offset() — sustained silence 검증 추가
  - Line 427-428: fade_in_samples, fade_out_samples 분리
  - Line 457: End zero-crossing snap 비활성화
  - Lines 480-481: 비대칭 fade 적용
