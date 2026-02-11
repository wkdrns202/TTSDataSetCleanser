import os
import whisper
import torch
from tqdm import tqdm
import glob

# 설정
WAV_DIR = os.path.join("datasets", "wavs")
METADATA_PATH = os.path.join("datasets", "metadata.txt")
MODEL_SIZE = "medium"  # base, small, medium, large 중 선택 (VRAM 용량에 따라 조절)
LANGUAGE = "ko"

def transcribe_audio():
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"디바이스 사용: {device}")
    
    # Whisper 모델 로드
    print(f"Whisper 모델({MODEL_SIZE}) 로드 중...")
    model = whisper.load_model(MODEL_SIZE, device=device)
    
    wav_files = sorted(glob.glob(os.path.join(WAV_DIR, "*.wav")))
    
    if not wav_files:
        print(f"경고: {WAV_DIR} 폴더에 처리할 wav 파일이 없습니다. split_audio.py를 먼저 실행했나요?")
        return

    print(f"총 {len(wav_files)}개 파일 라벨링 시작...")
    
    results = []
    
    # 파일별 전사 수행
    for wav_path in tqdm(wav_files):
        try:
            result = model.transcribe(wav_path, language=LANGUAGE)
            text = result["text"].strip()
            
            filename = os.path.basename(wav_path)
            
            # 메타데이터 포맷: 파일명|전사텍스트
            line = f"{filename}|{text}"
            results.append(line)
            
        except Exception as e:
            print(f"\n[오류] {wav_path} 처리 중 실패: {e}")

    # 결과 저장
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")
            
    print(f"\n완료! 메타데이터가 {METADATA_PATH}에 저장되었습니다.")
    print(f"총 {len(results)}개 문장 처리됨.")

if __name__ == "__main__":
    transcribe_audio()
