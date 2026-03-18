# TTS Dataset Cleanser

한국어 인공지능 음성합성(TTS) 모델의 학습용 데이터셋 자동 정제 파이프라인입니다. OpenAI Whisper ASR을 활용하여 장시간 원본 녹음 데이터를 개별 문장 단위 WAV로 분할하고, 자동 검증 및 품질 관리를 수행합니다.

## 프로젝트 개요

전문 성우의 고품질 녹음 데이터(48kHz/24-bit/Mono)를 기반으로, ASR 기반 자동 정렬(alignment) 기법을 통해 데이터 정제 및 라벨링 과정을 자동화합니다. 7회의 반복 R&D를 거쳐 최초 34.7%에서 최종 95.48%의 정렬 정확도를 달성하였습니다.

## 파이프라인 아키텍처

6-Stage 반복형(iterative) 파이프라인으로 구성됩니다:

```
STAGE 1        STAGE 2        STAGE 3        STAGE 4
Align &   ->   Clean &   ->   Validate  ->   Evaluate
Split          Post-process                      |
                                            STAGE 5
                                            Finalize
                                                 |
                                         All R >= 95%?
                                          YES /    \ NO
                                        DONE    STAGE 6
                                              Diagnose
                                              -> STAGE 1
```

| Stage | 이름 | 설명 |
|-------|------|------|
| Stage 1 | Align & Split | Whisper ASR로 음성 전사 후 스크립트와 정렬, 개별 WAV로 분할 |
| Stage 2 | Clean & Post-process | Zero-crossing snap, fade, 오디오 envelope 적용 |
| Stage 3 | Validate | 분할된 WAV을 재전사(re-transcribe)하여 원문과 비교 검증 |
| Stage 4 | Evaluate | 전체 메트릭 집계, R1~R6 요구사항 평가 |
| Stage 5 | Finalize | 95% 이상 달성 시 최종 데이터셋 확정 |
| Stage 6 | Diagnose & Improve | 95% 미만 시 실패 분석, 파라미터 조정 후 Stage 1로 복귀 |

## 원본 데이터

| 스크립트 | 문장 수 | 오디오 파일 수 | 내용 특성 |
|----------|---------|---------------|-----------|
| Script 1 | 300 | 3 | 문학 작품 낭독 |
| Script 2 | 1,644 | 6 | IT/프로그래밍 교육 콘텐츠 |
| Script 3 | 878 | 2 | 문학 텍스트 (고전/현대문학) |
| Script 4 | 1,005 | 5 | 일반 교양 콘텐츠 |
| Script 5 | 1,018* | 3 | 내러티브/서사 콘텐츠 |

\* Script 5는 541번 라인까지만 오디오가 존재하며, 542~1,018번은 미녹음 상태입니다.

- 녹음 포맷: 48kHz / 24-bit / Mono WAV (무손실 PCM)
- 파일 명명 규칙: `Script_{N}_{Start}-{End}.wav`

## 최종 품질 지표

| 요구사항 | 기준 | 달성 점수 | 판정 |
|----------|------|-----------|------|
| R1 (Alignment Accuracy) | >= 95% | 95.48% | PASS |
| R2 (Boundary Noise Clean) | >= 95% | 100.0% | PASS |
| R3 (Combined Pass Rate) | >= 95% | 95.4% | PASS |
| R4 (Metadata Integrity) | Complete | 99.9% | PASS |
| R5 (Reproducibility) | TRUE | TRUE | PASS |
| R6 (Audio Envelope) | >= 95% | 99.93% | PASS |

## 최종 데이터셋

| 항목 | 수치 |
|------|------|
| script.txt 엔트리 수 | 4,196 |
| WAV 파일 수 | 4,200 |
| 검역(Quarantine) 파일 | 74 (sim < 0.80) |
| 오디오 포맷 | 48kHz / 24-bit / Mono WAV |
| 총 오디오 시간 | ~8.18시간 |

### 스크립트별 통과율

| 스크립트 | 총 세그먼트 | 통과 | 통과율 |
|----------|------------|------|--------|
| Script 1 | 298 | 294 | 98.66% |
| Script 2 | 1,254 | 1,146 | 91.39% |
| Script 3 | 865 | 841 | 97.23% |
| Script 4 | 990 | 949 | 95.86% |
| Script 5 | 792 | 776 | 97.98% |

## 설치

```bash
pip install -r requirements.txt
```

시스템에 FFmpeg가 설치되어 있어야 합니다.

### 주요 의존성

| 패키지 | 용도 |
|--------|------|
| openai-whisper | ASR 전사 및 정렬 |
| numpy | 오디오 신호 처리 |
| soundfile | WAV 파일 I/O |
| python-Levenshtein | 텍스트 유사도 계산 (CER) |
| torch + CUDA | GPU 가속 |

## 사용법

1. 원본 오디오 파일을 `rawdata/audio/`에, 스크립트를 `rawdata/Scripts/`에 배치합니다.
2. Stage 1-2 (정렬 및 분할):
   ```bash
   python src/align_and_split.py
   ```
   - `--resume`: 체크포인트에서 이어서 실행 (기본값)
   - `--reset`: 처음부터 재실행
   - `--script N`: 특정 스크립트만 처리
3. Stage 3-4 (검증 및 평가):
   ```bash
   python src/evaluate_dataset.py
   ```
4. 결과물은 `datasets/wavs/`(WAV 파일)와 `datasets/script.txt`(메타데이터)에 출력됩니다.

## 핵심 알고리즘

- **Forward-only Search**: 현재 스크립트 위치에서 25줄 전방만 탐색 (후방 탐색은 한국어에서 중복 매칭 유발)
- **Segment Merging**: Whisper가 분할한 1~5개 연속 세그먼트를 결합하여 최적 매칭
- **Skip Penalty**: 먼 라인 매칭 시 줄당 0.01 패널티 부과
- **Re-sync**: 10회 연속 실패 시 75줄 범위에서 threshold 0.35로 재동기화
- **Two-tier Evaluation**: Whisper medium(Tier 1) + Whisper large(Tier 2, 실패 항목만)

## 핵심 교훈

- **Whisper temperature=0 금지**: 한국어는 연음/경음화/음운변동으로 소리와 글자의 괴리가 크므로 deterministic decoding은 치명적. 반드시 default temperature fallback 사용.
- **전방 탐색만 사용**: 후방 탐색은 한국어 조사/어미의 유사 패턴으로 중복 매칭 유발.
- **탐색 범위 제한**: 100~500줄의 넓은 범위는 false-match cascading 야기. 25줄이 최적.
- **GT-prompting 효과**: Ground-truth 텍스트를 initial_prompt로 사용하면 R1이 약 2배 향상 (34.7% -> 64.3%).
- **자동 메트릭의 한계**: R2/R6 자동 검사가 100% 통과해도 수동 청취로 audible bleed 발견 가능.

## 리포지토리 구조

```
{LocalDrive}:\Projects\AI_Research\TTSDataSetCleanser\
  src/align_and_split.py    # Stage 1-2: 정렬 및 분할
  src/evaluate_dataset.py   # Stage 3-4: 검증 및 평가
  src/pipeline_manager.py   # 워크플로 오케스트레이터
  rawdata/audio/            # 원본 오디오
  rawdata/Scripts/          # 원본 스크립트
  datasets/wavs/            # 출력: 분할된 WAV (4,200 files)
  datasets/script.txt       # 출력: 메타데이터
  logs/                     # 평가 보고서, iteration 로그
```

## 환경

| 항목 | 사양 |
|------|------|
| OS | Windows 10 Pro |
| Python | 3.11 |
| GPU | NVIDIA RTX 3060 Ti (8GB VRAM) |
| 파일시스템 | exFAT (로컬 드라이브) |
