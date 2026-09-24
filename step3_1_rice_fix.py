# ============================================================
# STEP 3 FIX v2 - Rice_Leaf Dataset (All subfolder variants)
# Blockchain in Agriculture - Q1 Journal Paper
# ============================================================

import os
import random
from PIL import Image, ImageEnhance, ImageFilter
import warnings
warnings.filterwarnings('ignore')

BASE_PATH   = "D:/September"
OUTPUT_PATH = os.path.join(BASE_PATH, "final_datasets")

RICE_BASE = os.path.join(BASE_PATH,
    "cleaned_datasets/Rice_Leaf"
    "/Rice Leaf and Crop Disease Detection Dataset"
    "/Rice Leaf and Crop Disease Detection Dataset")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
IMAGE_EXTS  = ['.jpg', '.jpeg', '.png', '.bmp']
AUG_FACTOR  = 3
random.seed(42)

# ALL possible subfolder name variants
SUB_VARIANTS = ['augmented', 'orginal', 'original', 'aug', 'org',
                'train', 'test', 'images', 'data']

def augment_image(img):
    return [
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

def collect_images_recursive(folder):
    """Collect all images from a folder and ALL its subfolders"""
    images = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                images.append(os.path.join(root, f))
    return images

def collect_class_images(rice_base):
    class_images = {}

    for class_name in os.listdir(rice_base):
        class_path = os.path.join(rice_base, class_name)
        if not os.path.isdir(class_path):
            continue

        # Collect ALL images recursively from entire class folder
        images = collect_images_recursive(class_path)

        if images:
            class_images[class_name] = images
            print(f"  Found: {class_name:<30} → {len(images):>5} images")

    return class_images

def process_rice_leaf():
    print("=" * 55)
    print("  FIXING v2: Rice_Leaf Dataset")
    print("=" * 55)

    if not os.path.exists(RICE_BASE):
        print("  ❌ Path not found!")
        return

    # Clear existing Rice_Leaf output first
    dst_base = os.path.join(OUTPUT_PATH, "Rice_Leaf")

    class_images = collect_class_images(RICE_BASE)

    if not class_images:
        print("  ❌ No images found!")
        return

    print(f"\n  Processing {len(class_images)} classes...\n")
    totals = {"train": 0, "val": 0, "test": 0}

    for class_name, images in class_images.items():
        random.shuffle(images)
        n     = len(images)
        t_end = int(n * TRAIN_RATIO)
        v_end = t_end + int(n * VAL_RATIO)

        splits = {
            "train": images[:t_end],
            "val"  : images[t_end:v_end],
            "test" : images[v_end:]
        }

        counts = {"train": 0, "val": 0, "test": 0}

        for split, img_list in splits.items():
            dst_dir = os.path.join(dst_base, split, class_name)
            os.makedirs(dst_dir, exist_ok=True)

            for i, src_path in enumerate(img_list):
                try:
                    img = Image.open(src_path).convert('RGB')

                    img.save(os.path.join(dst_dir, f"orig_{i:05d}.jpg"),
                             'JPEG', quality=95)
                    counts[split] += 1
                    totals[split] += 1

                    if split == "train":
                        aug_list = augment_image(img)
                        selected = random.sample(aug_list,
                                                 min(AUG_FACTOR, len(aug_list)))
                        for j, aug_img in enumerate(selected):
                            aug_img.save(
                                os.path.join(dst_dir, f"aug_{i:05d}_{j}.jpg"),
                                'JPEG', quality=95)
                            counts[split] += 1
                            totals[split] += 1
                except:
                    pass

        print(f"  ✅ {class_name:<30} "
              f"Tr:{counts['train']:>5} "
              f"Va:{counts['val']:>4} "
              f"Te:{counts['test']:>4}")

    total = sum(totals.values())
    print(f"\n{'='*55}")
    print(f"  Rice_Leaf COMPLETE!")
    print(f"  Train : {totals['train']:,}")
    print(f"  Val   : {totals['val']:,}")
    print(f"  Test  : {totals['test']:,}")
    print(f"  Total : {total:,}")
    print(f"{'='*55}")
    print(f"\n✅ Done! Saved to: {dst_base}")

if __name__ == "__main__":
    process_rice_leaf()
