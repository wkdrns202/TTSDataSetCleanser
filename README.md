# TTS Dataset Cleanser

긴 음성 데이터를 TTS(Text-to-Speech) 학습용 데이터셋으로 변환하는 도구입니다.

## 사용법

1. **오디오 파일 준비**: 
   - 7시간 20분 분량의 원본 오디오 파일(mp3, wav 등)을 `raw_audio/` 폴더에 넣으세요.

2. **설치**:
   ```bash
   pip install -r requirements.txt
   ```
   *(참고: 시스템에 FFmpeg가 설치되어 있어야 합니다.)*

3. **실행**:
   - 추후 제공될 스크립트를 통해 오디오 분할 및 라벨링을 수행합니다.
