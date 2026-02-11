import os
import glob

script_dir = r'rawdata/Scripts'
files = glob.glob(os.path.join(script_dir, 'Script_*_A0.txt'))

for file_path in files:
    try:
        # 인코딩 자동 감지 시도 (utf-8, cp949)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='cp949') as f:
                lines = f.readlines()

        # 빈 줄 제거
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        
        # 백업 (이미 있으면 건너뜀)
        if not os.path.exists(file_path + '.bak'):
            os.rename(file_path, file_path + '.bak')
        
        # 저장 (무조건 UTF-8로)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines) + '\n')
        
        print(f"Cleaned {os.path.basename(file_path)}: {len(lines)} -> {len(cleaned_lines)} lines")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
