import difflib

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

v1_path = r'g:\Projects\AI_Research\TTSDataSetCleanser\rawdata\Scripts\Script_2_A0_v1.txt'
v2_path = r'g:\Projects\AI_Research\TTSDataSetCleanser\rawdata\Scripts\Script_2_A0_v2.txt'

try:
    v1_lines = read_file(v1_path)
    v2_lines = read_file(v2_path)
except FileNotFoundError:
    print("Files not found. Please check paths.")
    exit()

# Set of lines for fast lookup
v2_set = set(v2_lines)
v1_set = set(v1_lines)

missing_in_v2 = []
for i, line in enumerate(v1_lines):
    if line and line not in v2_set:
        missing_in_v2.append(f"Line {i+1}: {line}")

added_in_v2 = []
for i, line in enumerate(v2_lines):
    if line and line not in v1_set:
        added_in_v2.append(f"Line {i+1}: {line}")

print(f"Total lines in v1: {len(v1_lines)}")
print(f"Total lines in v2: {len(v2_lines)}")

print("\n=== Lines in v1 but NOT in v2 (Potentially missing dialogue) ===")
if not missing_in_v2:
    print("None")
else:
    for item in missing_in_v2[:50]: # Show first 50
        print(item)
    if len(missing_in_v2) > 50:
        print(f"... and {len(missing_in_v2) - 50} more.")

print("\n=== Lines in v2 but NOT in v1 (New or Changed) ===")
if not added_in_v2:
    print("None")
else:
    for item in added_in_v2[:50]:
        print(item)
    if len(added_in_v2) > 50:
        print(f"... and {len(added_in_v2) - 50} more.")
