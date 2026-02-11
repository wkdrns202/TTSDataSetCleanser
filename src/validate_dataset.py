import os
import glob
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass
from pydub import AudioSegment
from tqdm import tqdm

# 프로젝트 루트 경로 계산 (이 파일의 상위 폴더의 상위 폴더)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
WAV_DIR = os.path.join(DATASET_DIR, "wavs")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.txt")

print(f"Dataset Dir: {DATASET_DIR}")
print(f"Metadata: {METADATA_PATH}")

def validate_dataset():
    if not os.path.exists(METADATA_PATH):
        print(f"메타데이터 파일이 없습니다. 경로: {os.path.abspath(METADATA_PATH)}")
        return

    print("데이터셋 검증 시작...")
    
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    total_files = len(lines)
    total_duration_ms = 0
    missing_files = []
    
    # Analyze by script
    script_stats = {} # {script_id: {found: 0, missing: 0}}
    
    print(f"메타데이터 라인 수: {total_files}")
    
    for line in tqdm(lines):
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue
            
        filename = parts[0]
        
        # Parse script id
        # Script_1_0001.wav
        try:
            s_id = filename.split("_")[1]
            if s_id not in script_stats:
                script_stats[s_id] = {"found": 0, "missing": 0}
        except:
            pass
            
        text = parts[1]
        
        wav_path = os.path.join(WAV_DIR, filename)
        
        if os.path.exists(wav_path):
            if s_id in script_stats: script_stats[s_id]["found"] += 1
            try:
                # audio = AudioSegment.from_wav(wav_path)
                # total_duration_ms += len(audio)
                pass # Skip duration for speed if checking counts
            except Exception:
                print(f"오류: {filename} 파일을 읽을 수 없습니다.")
        else:
            missing_files.append(filename)
            if s_id in script_stats: script_stats[s_id]["missing"] += 1

    print("\n[스크립트별 통계]")
    for s_id in sorted(script_stats.keys(), key=lambda x: int(x) if x.isdigit() else x):
        stats = script_stats[s_id]
        print(f"Script {s_id}: {stats['found']} files found")
            
    print("\n[검증 결과]")
    print(f"총 문장 수: {total_files}")
    print(f"총 오디오 길이: {total_duration_ms / 1000 / 60:.2f}분 ({total_duration_ms / 1000 / 3600:.2f}시간)")
    print(f"누락된 파일 수: {len(missing_files)}")
    
    if missing_files:
        print("누락된 파일 목록:")
        for f in missing_files[:10]:
            print(f" - {f}")
        if len(missing_files) > 10:
            print(f" ... 외 {len(missing_files)-10}개")

if __name__ == "__main__":
    validate_dataset()
