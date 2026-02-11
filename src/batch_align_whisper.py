import os
import sys
import re
import difflib
import warnings
import argparse
import torch
from pathlib import Path

# 경고 무시
warnings.filterwarnings("ignore")

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
    import whisper
    from pydub import AudioSegment
    from tqdm import tqdm
except ImportError:
    print("Installing requirements...")
    os.system("pip install openai-whisper pydub static-ffmpeg tqdm torch")
    import static_ffmpeg
    static_ffmpeg.add_paths()
    import whisper
    from pydub import AudioSegment
    from tqdm import tqdm

def normalize_text(text):
    # 한국어, 영어, 숫자만 남기고 제거 (공백도 제거하여 순수 텍스트 비교)
    text = re.sub(r'[^가-힣a-zA-Z0-9]', '', text)
    return text

def parse_filename(filename):
    # Script_1_1-122.wav -> script_id=1, start=1, end=122
    match = re.match(r'Script_(\d+)_(\d+)-(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None

def load_script(script_path):
    target_sentences = {}
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']
    
    lines = []
    used_enc = ""
    for enc in encodings:
        try:
            with open(script_path, 'r', encoding=enc) as f:
                lines = f.readlines()
            if len(lines) > 0:
                used_enc = enc
                print(f"  - Successfully loaded with encoding: {used_enc}")
                break
        except: continue
    
    if not lines:
        print("  - Failed to load script with any encoding.")
        return {}

    current_line_num = 0
    parsed_count = 0
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        current_line_num += 1
        target_sentences[current_line_num] = line
        parsed_count += 1
            
    print(f"  [DEBUG] Total parsed: {parsed_count} lines (Full Script)")
    return target_sentences

def process_file(audio_path, script_root, output_root, model, skipped_log_file):
    filename = os.path.basename(audio_path)
    file_stem = os.path.splitext(filename)[0]
    
    script_id, start_line, end_line = parse_filename(file_stem)
    
    script_path = os.path.join(script_root, f"Script_{script_id}_A0.txt")
    if not os.path.exists(script_path):
        print(f"Script file not found: {script_path}")
        return

    print(f"\nProcessing {filename}...")
    print(f"  - Script: {script_path}")
    print(f"  - Range (Filename): {start_line} ~ {end_line}")

    # 1. Load Script
    full_script = load_script(script_path)
    if not full_script:
        print("  - Warning: Script is empty or not found")
        return

    # 범위 설정 (파일명이 부정확하므로 여유를 둠 - 공백 제거로 인한 밀림 대응을 위해 200줄 여유)
    # 또한 start_line이 1인 경우는 처음부터 검색
    if start_line and end_line:
        search_start = max(1, start_line - 200)
        search_end = end_line + 200
        print(f"  - Search Range: {search_start} ~ {search_end} (Buffer +/- 200 applied)")
    else:
        search_start = 1
        search_end = max(full_script.keys()) if full_script else 1
        print(f"  - Search Range: Full Script ({len(full_script)} lines)")

    target_sentences = {k: v for k, v in full_script.items() if search_start <= k <= search_end}
    
    if not target_sentences:
        print("  - Warning: No matching lines found in target range")
        return
    print(f"  - Loaded {len(target_sentences)} sentences for matching")

    # 2. Transcribe
    print("  - Transcribing audio (Whisper)...")
    # verbose=False로 변경하여 CP949시 한국어 표현 불가 등이 야기하는 인코딩 에러 방지
    result = model.transcribe(audio_path, language="ko", verbose=False) 
    segments = result['segments']
    print(f"  - Found {len(segments)} segments")

    # 3. Load Audio
    audio = AudioSegment.from_wav(audio_path)
    
    # 4. Matching & Slicing
    wavs_dir = os.path.join(output_root, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    
    metadata_lines = []
    seg_idx = 0
    total_segments = len(segments)
    
    sorted_lines = sorted(target_sentences.keys())
    
    print("  - Matching and slicing...")
    
    matches_found = 0
    
    for line_num in tqdm(sorted_lines, desc="Progress"):
        target_text = target_sentences[line_num]
        norm_target = normalize_text(target_text)
        
        # 텍스트가 너무 짧으면 매칭이 어려울 수 있음
        if len(norm_target) < 2:
            # 너무 짧은 문장은 건너뛰거나 로그 남김
            with open(skipped_log_file, "a", encoding="utf-8") as f:
                f.write(f"{filename}|{line_num}|SKIP_SHORT|{target_text}\n")
            continue

        candidates = []
        search_window = 15 # 검색 윈도우 증가
        
        # 1~3 segments merging search
        for i in range(seg_idx, min(seg_idx + search_window, total_segments)):
            # Single
            seg1 = segments[i]
            norm_seg1 = normalize_text(seg1['text'])
            score1 = difflib.SequenceMatcher(None, norm_target, norm_seg1).ratio()
            candidates.append((score1, i, i+1, seg1['text']))
            
            # Merge 2
            if i + 1 < total_segments:
                seg2 = segments[i+1]
                merged2_text = seg1['text'] + " " + seg2['text']
                norm_merged2 = normalize_text(merged2_text)
                score2 = difflib.SequenceMatcher(None, norm_target, norm_merged2).ratio()
                candidates.append((score2, i, i+2, merged2_text))
                
            # Merge 3
            if i + 2 < total_segments:
                seg3 = segments[i+2]
                merged3_text = seg1['text'] + " " + segments[i+1]['text'] + " " + seg3['text']
                norm_merged3 = normalize_text(merged3_text)
                score3 = difflib.SequenceMatcher(None, norm_target, norm_merged3).ratio()
                candidates.append((score3, i, i+3, merged3_text))

        if not candidates: 
            with open(skipped_log_file, "a", encoding="utf-8") as f:
                f.write(f"{filename}|{line_num}|NO_CANDIDATES|{target_text}\n")
            continue
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, start_idx, end_idx, best_text = candidates[0]
        
        if best_score < 0.3: # Threshold 완화
            with open(skipped_log_file, "a", encoding="utf-8") as f:
                f.write(f"{filename}|{line_num}|LOW_SCORE({best_score:.2f})|{target_text}|BEST_MATCH:{best_text}\n")
            continue 
        
        matches_found += 1
        
        start_time = segments[start_idx]['start']
        end_time = segments[end_idx-1]['end']
        
        # 앞뒤 여유 50ms
        start_ms = max(0, int(start_time * 1000) - 50)
        end_ms = min(len(audio), int(end_time * 1000) + 50)
        
        chunk = audio[start_ms:end_ms]
        
        out_filename = f"Script_{script_id}_{line_num:04d}.wav"
        chunk.export(os.path.join(wavs_dir, out_filename), format="wav")
        metadata_lines.append(f"{out_filename}|{target_text}")
        
        # 매칭된 세그먼트 다음부터 검색 (중복 방지, 단 너무 많이 건너뛰지 않도록)
        # 하지만 동일 세그먼트가 다음 문장의 일부일 수도 있으므로(매우 짧은 문장 등)
        # 겹치지 않게 하려면 end_idx로 이동.
        # 한국어 데이터셋의 경우 문장이 명확히 구분되는 편이므로 end_idx가 안전.
        seg_idx = end_idx

    # Metadata Save (Append mode) - 즉시 저장
    meta_path = os.path.join(output_root, "metadata.txt")
    with open(meta_path, 'a', encoding='utf-8') as f:
        f.write("\n".join(metadata_lines) + "\n")
        
    print(f"  - Saved {len(metadata_lines)} items to {output_root}")

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 프로젝트 루트 (src의 상위)
    AUDIO_ROOT = os.path.join(BASE_DIR, "rawdata", "audio")
    SCRIPT_ROOT = os.path.join(BASE_DIR, "rawdata", "Scripts")
    OUTPUT_ROOT = os.path.join(BASE_DIR, "datasets")
    SKIPPED_LOG = os.path.join(BASE_DIR, "skipped_lines.log")
    
    print(f"Project Root: {BASE_DIR}")
    print(f"Audio Dir: {AUDIO_ROOT}")
    print(f"Script Dir: {SCRIPT_ROOT}")
    print(f"Output Dir: {OUTPUT_ROOT}")
    
    if not os.path.exists(AUDIO_ROOT):
        print(f"Error: Audio directory not found: {AUDIO_ROOT}")
        return

    wav_files = [f for f in os.listdir(AUDIO_ROOT) if f.endswith(".wav") and f.startswith("Script_")]
    
    # Sort files to process in order
    wav_files.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2].split('-')[0])))
    
    if not wav_files:
        print(f"No Script_*.wav files found in {AUDIO_ROOT}")
        return
        
    print("Loading Whisper Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = whisper.load_model("tiny", device=device) # tiny 모델 사용
    
    # Init directories
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Metadata & Log Init
    with open(os.path.join(OUTPUT_ROOT, "metadata.txt"), 'w', encoding='utf-8') as f:
        pass 
    with open(SKIPPED_LOG, 'w', encoding='utf-8') as f:
        f.write("Filename|LineNum|Reason|TargetText|Extra\n")

    # 전체 파일 처리
    for wav_file in wav_files:
        process_file(os.path.join(AUDIO_ROOT, wav_file), SCRIPT_ROOT, OUTPUT_ROOT, model, SKIPPED_LOG)

if __name__ == "__main__":
    main()
