# SAM Data Engine vs. TTS Pipeline 분석 및 향후 방향
**작성일: 2026-03-29**
**참고 논문: Segment Anything (Kirillov et al., Meta AI Research, FAIR)**

---

## 1. SAM Data Engine 핵심 구조 (논문 Section 4)

SAM의 Data Engine은 3단계로 구성되며, 핵심은 **모델이 데이터를 생성하고, 그 데이터로 모델이 개선되는 선순환 구조**이다.

| Stage | SAM | 본 TTS 파이프라인 |
|-------|-----|-------------------|
| **1. Assisted-Manual** | 어노테이터가 SAM을 인터랙티브하게 사용; SAM 6회 재훈련; 어노테이션 시간 34s→14s | 파라미터 반복 최적화 (6회, 35%→95%), 그러나 Whisper 자체는 개선되지 않음 |
| **2. Semi-Automatic** | 자동 감지된 confident mask → 인간이 어려운 부분 보완; 5회 추가 재훈련 | Tier 1/Tier 2 (medium→large) 계층 검증은 유사하나 정적(static) — 재훈련 없음 |
| **3. Fully Automatic** | 모델이 1.1B 마스크를 자율 생성, stability check + confidence scoring으로 품질 보증 | 미달성 — 파이프라인이 고정된 Whisper 성능에 의존 |

### SAM의 주요 기법

- **Stability Check**: 확률맵 임계값을 0.5 ± δ로 변동시켜 출력이 일관되면 "stable"로 판정
- **Multi-Hypothesis (Ambiguity-Aware)**: 모호한 프롬프트에 대해 3개의 유효 마스크를 동시 예측
- **IoU Prediction**: 자체 출력의 품질(IoU)을 예측하는 경량 헤드 학습
- **NMS (Non-Maximal Suppression)**: 중복 마스크 필터링
- **점진적 자동화**: 인간 개입을 단계적으로 줄이면서 모델 품질을 유지/향상

---

## 2. 핵심 차이점: 선순환 구조의 부재

### SAM
```
annotate → train model → better model → better annotations → ...
```

### 본 TTS 파이프라인
```
optimize parameters around fixed Whisper → plateau at Whisper's ceiling
```

현재 달성한 95% 정확도는 **Whisper의 한국어 인식 한계에 의해 상한이 결정**된다. 잔여 ~5% 실패의 대다수가 Whisper의 어휘/디코딩 한계에 기인한다는 분석(보고서 Section 4.3)이 이를 뒷받침한다.

또한 현재 정확도는 **분석에 사용한 특정 데이터셋에서만 유효**할 수 있으며, 새로운 화자/도메인에 대한 일반화는 검증되지 않았다.

---

## 3. SAM 기법의 음성 정렬 적용 아이디어

### 3.1 Stability Check → 정렬 신뢰도 스코어링

SAM이 임계값 변동으로 마스크 안정성을 측정하듯, 음성 정렬에서:

- 동일 세그먼트에 대해 **복수의 디코딩 설정**(beam size, temperature 샘플)으로 Whisper 실행
- 전사 결과가 수렴하면 → 고신뢰 정렬
- 전사 결과가 분산되면 → 리뷰 대상 또는 2차 처리 플래그
- 별도 검증 모델 없이 **세그먼트별 신뢰도 점수** 획득 가능

### 3.2 Multi-Hypothesis Matching → 모호성 인식 정렬

SAM이 모호한 프롬프트에 3개 마스크를 예측하듯:

- 세그먼트가 복수의 대본 라인에 매칭될 수 있을 때(한국어 조사/어미 유사성 문제), **Top-K 정렬 후보**를 점수와 함께 생성
- 하나를 하드셀렉트하지 않고 가설 그래프(hypothesis graph)를 유지하여 전역적으로 해소 (정렬 경로에 대한 beam search)
- 한국어 조사/어미 false-positive 문제에 직접 대응

### 3.3 IoU Prediction → 정렬 품질 예측

SAM이 자체 출력의 IoU를 예측하는 경량 헤드를 학습하듯:

- "이 정렬이 R1 검증을 통과할 것인가?"를 예측하는 경량 분류기 학습
- 피처: 유사도 점수, 세그먼트 길이, Whisper logprob, 텍스트 길이 비율 등
- 활용: 고신뢰 → 자동 수락, 저신뢰 → 인간 리뷰 또는 Tier 2 처리

---

## 4. 향후 방향 로드맵 (A/B/C/D)

### (A) Data Engine Level — 선순환 구조 구축

**A-1: Whisper 파인튜닝 (본격적 선순환)**
- 검증된 ~4,000 speech-text 쌍을 활용하여 Whisper-medium을 한국어 도메인 어휘에 fine-tune
- 검증된 95% 출력 → 훈련 데이터 → 파인튜닝된 Whisper → 파이프라인 재실행 → 5% 실패 일부 복구 → 새 훈련 데이터 → 반복
- 가장 직접적인 SAM식 data engine
- 리스크: 단일 화자/도메인 과적합. 완화: held-out set 평가
- 선행 조건: 다양한 데이터 확보 시 효과 극대화

**A-2: Stability/Confidence 기법 (즉시 적용 가능)**
- 멀티 디코드 stability check 구현 — 모델 훈련 불필요
- 순수 알고리즘적 개선, 즉시 적용 가능
- 기대 효과: 5% 실패를 "Whisper 어휘 인식 한계" vs "정렬 모호성"으로 분리 진단

### (B) 모델 학습: F5TTS Phase 1/2 완성 및 MOS 기반 평가

- Phase 1에서 기본 음소-음향 매핑 학습 확인, 추가 학습 데이터 필요
- Phase 2: 감정/프로소디 포함 데이터로 추가 학습
- **MOS 평가 체계 구축**으로 자동 메트릭이 포착하지 못하는 지각적(perceptual) 품질 측정
- 파이프라인 출력 품질의 실질적 검증 수단

### (C) 추론/배포: koreanAIvoice.com

- 기존 도메인에 배포된 모델의 교체/추가/업데이트
- 실제 사용자 피드백 확보
- 아키텍처적 우선순위는 낮으나 피드백 가치 높음

### (D) TTS↔ASR 피드백 루프 — SAM식 선순환 구현

```
TTS가 음성 합성 → ASR이 역전사 → 입력 텍스트와 비교
→ 불일치 시: TTS 발음 오류 OR ASR 인식 오류
→ 이 신호로 양쪽 모두 개선
```

자기지도(self-supervised) 검증 루프로, SAM의 fully automatic stage가 confidence + stability check로 자체 출력을 필터링하는 것과 유사.

---

## 5. 우선순위 권고 (Impact vs. Effort)

| 순위 | 방향 | 근거 |
|------|------|------|
| 1 | **A-2** (stability check, multi-hypothesis) | 즉시 적용 가능, 훈련 불필요, 파이프라인 진단 능력 향상 |
| 2 | **B** (F5TTS Phase 2) | 실제 TTS 품질로 데이터 검증; MOS가 누락된 지각적 평가 제공 |
| 3 | **D** (TTS↔ASR 피드백 루프) | TTS 모델 확보 후, 음성 도메인의 SAM식 data engine 구현 |
| 4 | **A-1** (Whisper 파인튜닝) | 다양한 데이터 확보 시 효과적; 현재 단일 화자 4K 쌍은 과적합 위험 |
| 5 | **C** (배포) | B에서 사용 가능한 품질 달성 시 |

### 전략적 핵심

**B와 D가 결합되면 선순환 구조가 완성된다.** TTS 모델이 데이터 파이프라인의 소비자이자 품질 신호가 된다. 이것이 SAM의 핵심 패러다임과의 진정한 병렬(parallel)이다 — 모델 자체가 데이터 개선을 주도하는 구조.
