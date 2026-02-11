import os
import whisper
import torch
import static_ffmpeg

print("1. static_ffmpeg 경로 설정...")
static_ffmpeg.add_paths()

print("2. Whisper 모델 로드 (tiny)...")
try:
    model = whisper.load_model("tiny", device="cpu")
    print("   모델 로드 성공")
except Exception as e:
    print(f"   모델 로드 실패: {e}")
    exit(1)

audio_path = "rawdata/audio/Script_1_1-122.wav"
print(f"3. 오디오 파일 확인: {audio_path}")
if not os.path.exists(audio_path):
    print("   파일이 없습니다!")
    exit(1)

print("4. Whisper 전사 시작 (30초만)...")
try:
    # fp16=False는 CPU에서 경고를 방지하기 위함
    result = model.transcribe(audio_path, language="ko", fp16=False)
    print("   전사 성공!")
    print(f"   결과 일부: {result['text'][:50]}...")
except Exception as e:
    print(f"   전사 실패: {e}")
