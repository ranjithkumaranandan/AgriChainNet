# ============================================================
# PHASE 1 - STEP 2: DATA CLEANING
# Blockchain in Agriculture - Q1 Journal Paper
# Run this in: D:\September\
# ============================================================

import os
import cv2
import shutil
import hashlib
import numpy as np
from PIL import Image
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_PATH = "D:/September"

DATASETS = {
    "PlantVillage"  : os.path.join(BASE_PATH, "plantvillage"),
    "Multi_Crop"    : os.path.join(BASE_PATH, "Multi-Crop Disease Dataset"),
    "Leaf_Disease"  : os.path.join(BASE_PATH, "Plant Leaf Disease Recognition Dataset"),
    "Rice_Leaf"     : os.path.join(BASE_PATH, "Rice Leaf and Crop Disease Detection Dataset"),
}

# Target sizes per task
TARGET_SIZES = {
    "PlantVillage"  : (224, 224),   # Classification
    "Multi_Crop"    : (640, 640),   # Detection (already correct)
    "Leaf_Disease"  : (512, 512),   # Segmentation
    "Rice_Leaf"     : (224, 224),   # Classification
}

CLEANED_PATH = os.path.join(BASE_PATH, "cleaned_datasets")
LOG_PATH     = os.path.join(BASE_PATH, "cleaning_logs")
os.makedirs(CLEANED_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# ============================================================
# FUNCTION 1: Find corrupted images
# ============================================================
def find_corrupted_images(dataset_path):
    corrupted = []
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                img_path = os.path.join(root, f)
                try:
                    img = Image.open(img_path)
                    img.verify()  # Check if image is valid
                except Exception as e:
                    corrupted.append(img_path)
    return corrupted

# ============================================================
# FUNCTION 2: Find duplicate images using MD5 hash
# ============================================================
def find_duplicates(dataset_path):
    hash_map = {}
    duplicates = []

    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                img_path = os.path.join(root, f)
                try:
                    with open(img_path, 'rb') as file:
                        file_hash = hashlib.md5(file.read()).hexdigest()
                    if file_hash in hash_map:
                        duplicates.append(img_path)
                    else:
                        hash_map[file_hash] = img_path
                except:
                    pass

    return duplicates

# ============================================================
# FUNCTION 3: Resize and save cleaned images
# ============================================================
def resize_and_save(src_path, dst_path, target_size, skip_existing=True):
    if skip_existing and os.path.exists(dst_path):
        return True
    try:
        img = Image.open(src_path).convert('RGB')
        img = img.resize(target_size, Image.LANCZOS)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        img.save(dst_path, 'JPEG', quality=95)
        return True
    except:
        return False

# ============================================================
# FUNCTION 4: Copy YOLO annotation files
# ============================================================
def copy_annotation(src_path, dst_path):
    """Copy .txt annotation files for YOLO (Multi_Crop dataset)"""
    txt_path = os.path.splitext(src_path)[0] + '.txt'
    dst_txt  = os.path.splitext(dst_path)[0] + '.txt'
    if os.path.exists(txt_path):
        os.makedirs(os.path.dirname(dst_txt), exist_ok=True)
        shutil.copy2(txt_path, dst_txt)

# ============================================================
# FUNCTION 5: Clean one dataset
# ============================================================
def clean_dataset(dataset_name, dataset_path, target_size):
    print(f"\n{'='*55}")
    print(f"  Cleaning: {dataset_name}")
    print(f"{'='*55}")

    log_lines = [f"Cleaning Report: {dataset_name}\n"]

    # --- Step A: Find corrupted images ---
    print(f"  🔍 Checking for corrupted images...")
    corrupted = find_corrupted_images(dataset_path)
    print(f"     Found {len(corrupted)} corrupted images")
    log_lines.append(f"Corrupted images: {len(corrupted)}")
    for c in corrupted:
        log_lines.append(f"  CORRUPTED: {c}")

    # --- Step B: Find duplicates ---
    print(f"  🔍 Checking for duplicate images...")
    duplicates = find_duplicates(dataset_path)
    print(f"     Found {len(duplicates)} duplicate images")
    log_lines.append(f"Duplicate images: {len(duplicates)}")
    for d in duplicates:
        log_lines.append(f"  DUPLICATE: {d}")

    # --- Step C: Resize and copy clean images ---
    print(f"  📐 Resizing images to {target_size}...")
    dst_base   = os.path.join(CLEANED_PATH, dataset_name)
    skip_set   = set(corrupted + duplicates)

    total      = 0
    success    = 0
    failed     = 0

    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue

            src_path = os.path.join(root, f)

            # Skip corrupted and duplicates
            if src_path in skip_set:
                continue

            # Maintain folder structure
            rel_path = os.path.relpath(src_path, dataset_path)
            rel_path = os.path.splitext(rel_path)[0] + '.jpg'
            dst_path = os.path.join(dst_base, rel_path)

            total += 1
            if resize_and_save(src_path, dst_path, target_size):
                success += 1
                # Copy YOLO annotations if Multi_Crop
                if dataset_name == "Multi_Crop":
                    copy_annotation(src_path, dst_path)
            else:
                failed += 1
                log_lines.append(f"  FAILED: {src_path}")

            # Progress update every 5000 images
            if total % 5000 == 0:
                print(f"     Progress: {total} images processed...")

    print(f"  ✅ Done!")
    print(f"     Total processed : {total:,}")
    print(f"     Successfully cleaned : {success:,}")
    print(f"     Corrupted removed    : {len(corrupted):,}")
    print(f"     Duplicates removed   : {len(duplicates):,}")
    print(f"     Failed               : {failed:,}")

    log_lines.append(f"\nTotal Processed : {total}")
    log_lines.append(f"Success         : {success}")
    log_lines.append(f"Corrupted       : {len(corrupted)}")
    log_lines.append(f"Duplicates      : {len(duplicates)}")
    log_lines.append(f"Failed          : {failed}")

    # Save log
    log_file = os.path.join(LOG_PATH, f"{dataset_name}_cleaning_log.txt")
    with open(log_file, 'w') as lf:
        lf.write('\n'.join(log_lines))
    print(f"  📄 Log saved: {log_file}")

    return {
        'total'      : total,
        'success'    : success,
        'corrupted'  : len(corrupted),
        'duplicates' : len(duplicates),
        'failed'     : failed
    }

# ============================================================
# FUNCTION 6: Final summary
# ============================================================
def print_final_summary(all_results):
    print(f"\n{'='*55}")
    print(f"       CLEANING SUMMARY REPORT")
    print(f"       Blockchain in Agriculture - Q1 Paper")
    print(f"{'='*55}")

    total_clean = 0
    for name, r in all_results.items():
        print(f"\n📁 {name}")
        print(f"   Processed  : {r['total']:,}")
        print(f"   Clean      : {r['success']:,}")
        print(f"   Corrupted  : {r['corrupted']:,}")
        print(f"   Duplicates : {r['duplicates']:,}")
        print(f"   Failed     : {r['failed']:,}")
        total_clean += r['success']

    print(f"\n{'='*55}")
    print(f"  TOTAL CLEAN IMAGES READY: {total_clean:,}")
    print(f"  SAVED TO: {CLEANED_PATH}")
    print(f"{'='*55}")
    print(f"\n✅ Step 2 Complete! Ready for Step 3: Augmentation & Splitting")

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("="*55)
    print("  PHASE 1 - STEP 2: DATA CLEANING")
    print("  Blockchain in Agriculture - Q1 Paper")
    print("="*55)
    print(f"\n  Cleaned datasets will be saved to:")
    print(f"  {CLEANED_PATH}\n")

    all_results = {}

    for dataset_name, dataset_path in DATASETS.items():
        if not os.path.exists(dataset_path):
            print(f"\n⚠️  Skipping {dataset_name} — path not found!")
            continue

        target_size = TARGET_SIZES[dataset_name]
        result = clean_dataset(dataset_name, dataset_path, target_size)
        all_results[dataset_name] = result

    print_final_summary(all_results)