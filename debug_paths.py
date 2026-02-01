import os

root = "extracted_data"
print(f"Walking {root}...")
for dirpath, dirnames, filenames in os.walk(root):
    print(f"Found dir: {dirpath}")
    if filenames:
        print(f"  First 3 files: {filenames[:3]}")
