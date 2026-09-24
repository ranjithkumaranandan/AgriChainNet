# ============================================================
# PHASE 1 - STEP 3: AUGMENTATION & TRAIN/VAL/TEST SPLIT (FIXED)
# Blockchain in Agriculture - Q1 Journal Paper
# ============================================================

import os
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION — EXACT PATHS (FIXED)
# ============================================================
BASE_PATH    = "D:/September"
OUTPUT_PATH  = os.path.join(BASE_PATH, "final_datasets")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# EXACT class-level paths for each dataset
DATASET_PATHS = {
    "PlantVillage": os.path.join(BASE_PATH,
        "cleaned_datasets/PlantVillage/plantvillage dataset/color"),

    "Multi_Crop": os.path.join(BASE_PATH,
        "cleaned_datasets/Multi_Crop"),

    "Leaf_Disease": os.path.join(BASE_PATH,
        "cleaned_datasets/Leaf_Disease/Plant Leaf Disease Recognition Dataset"
        "/Data for Leaf Disease/Data for Leaf Disease/Background Removed"),

    "Rice_Leaf": os.path.join(BASE_PATH,
        "cleaned_datasets/Rice_Leaf/Rice Leaf and Crop Disease Detection Dataset"
        "/Rice Leaf and Crop Disease Detection Dataset"),
}

# Augmentation factor per dataset
AUG_FACTOR = {
    "PlantVillage": 2,
    "Multi_Crop"  : 2,
    "Leaf_Disease": 3,
    "Rice_Leaf"   : 3,
}

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
IMAGE_EXTS  = ['.jpg', '.jpeg', '.png', '.bmp']
random.seed(42)

# ============================================================
# AUGMENTATION
# ============================================================
def augment_image(img):
    augs = [
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.transpose(Image.FLIP_TOP_BOTTOM),
        img.rotate(90, expand=True).resize(img.size),
        img.rotate(180),
        ImageEnhance.Brightness(img).enhance(1.3),
        ImageEnhance.Brightness(img).enhance(0.7),
        ImageEnhance.Contrast(img).enhance(1.3),
        ImageEnhance.Color(img).enhance(1.3),
        img.filter(ImageFilter.GaussianBlur(radius=1)),
        ImageEnhance.Sharpness(img).enhance(2.0),
    ]
    return augs

# ============================================================
# PROCESS CLASSIFICATION DATASET
# ============================================================
def process_classification(dataset_name, src_base, aug_factor):
    print(f"\n{'='*55}")
    print(f"  Processing: {dataset_name}")
    print(f"  Path: {src_base}")
    print(f"{'='*55}")

    if not os.path.exists(src_base):
        print(f"  ❌ Path not found! Skipping.")
        return {"train": 0, "val": 0, "test": 0}

    dst_base   = os.path.join(OUTPUT_PATH, dataset_name)
    totals     = {"train": 0, "val": 0, "test": 0}
    class_dirs = [d for d in os.listdir(src_base)
                  if os.path.isdir(os.path.join(src_base, d))]

    if not class_dirs:
        print(f"  ❌ No class folders found!")
        return totals

    print(f"  Found {len(class_dirs)} classes")

    for class_name in sorted(class_dirs):
        class_src = os.path.join(src_base, class_name)
        images    = [f for f in os.listdir(class_src)
                     if os.path.splitext(f)[1].lower() in IMAGE_EXTS]

        if not images:
            continue

        random.shuffle(images)
        n         = len(images)
        t_end     = int(n * TRAIN_RATIO)
        v_end     = t_end + int(n * VAL_RATIO)

        splits = {
            "train": images[:t_end],
            "val"  : images[t_end:v_end],
            "test" : images[v_end:]
        }

        counts = {"train": 0, "val": 0, "test": 0}

        for split, img_list in splits.items():
            dst_dir = os.path.join(dst_base, split, class_name)
            os.makedirs(dst_dir, exist_ok=True)

            for i, fname in enumerate(img_list):
                src_path = os.path.join(class_src, fname)
                try:
                    img = Image.open(src_path).convert('RGB')

                    # Save original
                    img.save(os.path.join(dst_dir, f"orig_{i:05d}.jpg"),
                             'JPEG', quality=95)
                    counts[split] += 1
                    totals[split] += 1

                    # Augment training only
                    if split == "train":
                        aug_list = augment_image(img)
                        selected = random.sample(aug_list,
                                                 min(aug_factor, len(aug_list)))
                        for j, aug_img in enumerate(selected):
                            aug_img.save(
                                os.path.join(dst_dir, f"aug_{i:05d}_{j}.jpg"),
                                'JPEG', quality=95)
                            counts[split] += 1
                            totals[split] += 1
                except:
                    pass

        print(f"  ✅ {class_name[:35]:<35} "
              f"Tr:{counts['train']:>5} Va:{counts['val']:>4} Te:{counts['test']:>4}")

    return totals

# ============================================================
# PROCESS DETECTION DATASET (Multi_Crop - YOLO)
# ============================================================
def process_detection(dataset_name, src_base, aug_factor):
    print(f"\n{'='*55}")
    print(f"  Processing: {dataset_name} (Detection/YOLO)")
    print(f"  Path: {src_base}")
    print(f"{'='*55}")

    if not os.path.exists(src_base):
        print(f"  ❌ Path not found!")
        return {"train": 0, "val": 0, "test": 0}

    # Collect all images recursively
    all_images = []
    for root, dirs, files in os.walk(src_base):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                all_images.append(os.path.join(root, f))

    random.shuffle(all_images)
    n     = len(all_images)
    t_end = int(n * TRAIN_RATIO)
    v_end = t_end + int(n * VAL_RATIO)

    splits = {
        "train": all_images[:t_end],
        "val"  : all_images[t_end:v_end],
        "test" : all_images[v_end:]
    }

    dst_base = os.path.join(OUTPUT_PATH, dataset_name)
    totals   = {"train": 0, "val": 0, "test": 0}

    for split, img_list in splits.items():
        img_dir = os.path.join(dst_base, split, "images")
        lbl_dir = os.path.join(dst_base, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i, src_path in enumerate(img_list):
            try:
                img  = Image.open(src_path).convert('RGB')
                name = f"img_{i:06d}"
                img.save(os.path.join(img_dir, f"{name}.jpg"), 'JPEG', quality=95)
                totals[split] += 1

                txt = os.path.splitext(src_path)[0] + '.txt'
                if os.path.exists(txt):
                    shutil.copy2(txt, os.path.join(lbl_dir, f"{name}.txt"))

                # Augment train only
                if split == "train":
                    for j, aug_type in enumerate(['hflip', 'bright'][:aug_factor]):
                        if aug_type == 'hflip':
                            aug = img.transpose(Image.FLIP_LEFT_RIGHT)
                        else:
                            aug = ImageEnhance.Brightness(img).enhance(
                                random.uniform(0.7, 1.3))
                        aug.save(os.path.join(img_dir, f"aug_{i:06d}_{j}.jpg"),
                                 'JPEG', quality=95)
                        totals[split] += 1
            except:
                pass

    print(f"  ✅ Train:{totals['train']:>6} Val:{totals['val']:>5} Test:{totals['test']:>5}")
    return totals

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  PHASE 1 - STEP 3: AUGMENTATION & SPLITTING (FIXED)")
    print("  Blockchain in Agriculture - Q1 Paper")
    print("=" * 55)

    all_totals = {}

    # PlantVillage — Classification
    all_totals["PlantVillage"] = process_classification(
        "PlantVillage", DATASET_PATHS["PlantVillage"], AUG_FACTOR["PlantVillage"])

    # Multi_Crop — Detection (YOLO)
    all_totals["Multi_Crop"] = process_detection(
        "Multi_Crop", DATASET_PATHS["Multi_Crop"], AUG_FACTOR["Multi_Crop"])

    # Leaf_Disease — Segmentation/Classification
    all_totals["Leaf_Disease"] = process_classification(
        "Leaf_Disease", DATASET_PATHS["Leaf_Disease"], AUG_FACTOR["Leaf_Disease"])

    # Rice_Leaf — Classification
    all_totals["Rice_Leaf"] = process_classification(
        "Rice_Leaf", DATASET_PATHS["Rice_Leaf"], AUG_FACTOR["Rice_Leaf"])

    # Summary
    print(f"\n{'='*55}")
    print(f"       FINAL SUMMARY")
    print(f"{'='*55}")
    grand = 0
    for name, totals in all_totals.items():
        total = sum(totals.values())
        grand += total
        print(f"\n📁 {name}")
        print(f"   Train : {totals['train']:,}")
        print(f"   Val   : {totals['val']:,}")
        print(f"   Test  : {totals['test']:,}")
        print(f"   Total : {total:,}")

    print(f"\n{'='*55}")
    print(f"  GRAND TOTAL : {grand:,} images")
    print(f"  SAVED TO    : {OUTPUT_PATH}")
    print(f"{'='*55}")
    print(f"\n✅ Step 3 Complete! Ready for Phase 2: Model Building")
