import chardet
import os

script_path = "rawdata/Scripts/Script_1_A0.txt"

with open(script_path, 'rb') as f:
    raw_data = f.read(100) # 처음 100바이트만 읽기
    result = chardet.detect(raw_data)
    print(f"Detected encoding: {result}")
    
    print(f"Raw hex: {raw_data.hex()}")
    try:
        print(f"Decoded with detected: {raw_data.decode(result['encoding'])}")
    except:
        print("Decoded failed")
        
    print("-" * 20)
    # CP949 시도
    try:
        print(f"Decoded with cp949: {raw_data.decode('cp949')}")
    except:
        print("Decoded with cp949 failed")
        
    # UTF-16 시도
    try:
        print(f"Decoded with utf-16: {raw_data.decode('utf-16')}")
    except:
        print("Decoded with utf-16 failed")
