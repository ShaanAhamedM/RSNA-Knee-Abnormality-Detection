import zipfile

zip_path = r"D:\RSNA\rsna-knee-abnormality-detection.zip"
print(f"Opening {zip_path}...")
with zipfile.ZipFile(zip_path, 'r') as z:
    all_names = z.namelist()
    print(f"Total entries in zip: {len(all_names)}")
    
    # Check top-level files / folders
    top_level = set(n.split('/')[0] for n in all_names if '/' in n)
    root_files = [n for n in all_names if '/' not in n]
    print(f"Root files: {root_files}")
    print(f"Top-level dirs: {top_level}")
    
    # Find all CSV files in the zip
    csv_files = [n for n in all_names if n.endswith('.csv')]
    print(f"All CSV files in zip: {csv_files}")
    
    # Check sample file paths
    sample_paths = all_names[:20]
    print(f"\nFirst 20 paths in zip:")
    for p in sample_paths:
        print(" ", p)
