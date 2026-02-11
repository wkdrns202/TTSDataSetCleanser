import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm
import glob

# 설정
RAW_AUDIO_DIR = "raw_audio"
OUTPUT_DIR = os.path.join("datasets", "wavs")
MIN_SILENCE_LEN = 500  # 밀리초 (0.5초)
SILENCE_THRESH = -40   # dBFS (이 값보다 작으면 묵음으로 간주)
KEEP_SILENCE = 200     # 분할된 클립 앞뒤에 남길 묵음 길이 (밀리초)

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_audio():
    audio_files = glob.glob(os.path.join(RAW_AUDIO_DIR, "*"))
    
    if not audio_files:
        print(f"경고: {RAW_AUDIO_DIR} 폴더에 오디오 파일이 없습니다.")
        return

    file_counter = 1

    for file_path in audio_files:
        filename = os.path.basename(file_path)
        print(f"처리 중: {filename}...")
        
        try:
            # 오디오 로드 (포맷 자동 감지)
            audio = AudioSegment.from_file(file_path)
            print(f"  - 길이: {len(audio)/1000:.2f}초")

            # 묵음 기준 분할
            print("  - 묵음 기준 분할 시작 (시간이 걸릴 수 있습니다)...")
            chunks = split_on_silence(
                audio,
                min_silence_len=MIN_SILENCE_LEN,
                silence_thresh=SILENCE_THRESH,
                keep_silence=KEEP_SILENCE
            )

            print(f"  - {len(chunks)}개 클립 생성됨. 저장 시작...")

            for i, chunk in enumerate(tqdm(chunks)):
                # 너무 짧은 클립 제외 (예: 1초 미만)
                if len(chunk) < 1000:
                    continue
                
                # 내보내기
                output_filename = f"audio_{file_counter:05d}.wav"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                # 22050Hz, Mono, 16bit로 변환 (TTS 표준)
                chunk = chunk.set_frame_rate(22050).set_channels(1).set_sample_width(2)
                chunk.export(output_path, format="wav")
                
                file_counter += 1

        except Exception as e:
            print(f"오류 발생 ({filename}): {e}")

    print(f"\n완료! 총 {file_counter-1}개의 클립이 {OUTPUT_DIR}에 저장되었습니다.")

if __name__ == "__main__":
    process_audio()
