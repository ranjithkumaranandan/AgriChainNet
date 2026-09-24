# ============================================================
# PHASE 2A - YOLOv8 DETECTION (FIXED)
# Using original Multi-Crop Disease Dataset directly
# Blockchain in Agriculture - Q1 Journal Paper
# ============================================================

import os
import yaml
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
BASE_PATH    = "D:/September"

# Use ORIGINAL dataset directly — it already has yaml + labels!
DATASET_PATH = os.path.join(BASE_PATH, "Multi-Crop Disease Dataset")
YAML_PATH    = os.path.join(DATASET_PATH, "data.yaml")
SAVE_PATH    = os.path.join(BASE_PATH, "models/detection")
os.makedirs(SAVE_PATH, exist_ok=True)

EPOCHS     = 50
BATCH_SIZE = 16
IMG_SIZE   = 640
PATIENCE   = 10
DEVICE     = 0 if torch.cuda.is_available() else 'cpu'

print("=" * 55)
print("  PHASE 2A - YOLOv8 DETECTION TRAINING (FIXED)")
print("  Blockchain in Agriculture - Q1 Paper")
print("=" * 55)
print(f"\n  Device : {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")

# ============================================================
# STEP 1: READ AND FIX YAML
# ============================================================
def fix_yaml():
    print(f"\n📝 Reading dataset YAML: {YAML_PATH}")

    with open(YAML_PATH, 'r') as f:
        data = yaml.safe_load(f)

    print(f"  Original content:")
    for k, v in data.items():
        print(f"    {k}: {v}")

    # Fix paths to absolute
    fixed = {
        'path' : DATASET_PATH.replace("\\", "/"),
        'train': 'train/images',
        'val'  : 'valid/images',
        'test' : 'test/images',
        'nc'   : data.get('nc', 1),
        'names': data.get('names', ['disease'])
    }

    # Save fixed yaml
    fixed_yaml = os.path.join(BASE_PATH, "multicrop_fixed.yaml")
    with open(fixed_yaml, 'w') as f:
        yaml.dump(fixed, f, default_flow_style=False)

    print(f"\n  ✅ Fixed YAML saved: {fixed_yaml}")
    print(f"  Classes: {fixed['nc']}")
    print(f"  Names  : {fixed['names']}")

    return fixed_yaml, fixed['nc'], fixed['names']

# ============================================================
# STEP 2: CHECK DATASET
# ============================================================
def check_dataset():
    print(f"\n📂 Checking dataset structure...")

    splits = {
        'train': os.path.join(DATASET_PATH, 'train'),
        'valid': os.path.join(DATASET_PATH, 'valid'),
        'test' : os.path.join(DATASET_PATH, 'test'),
    }

    all_ok = True
    for split, path in splits.items():
        img_path = os.path.join(path, 'images')
        lbl_path = os.path.join(path, 'labels')

        imgs = len([f for f in os.listdir(img_path)
                    if f.endswith(('.jpg','.jpeg','.png'))]) \
               if os.path.exists(img_path) else 0
        lbls = len([f for f in os.listdir(lbl_path)
                    if f.endswith('.txt')]) \
               if os.path.exists(lbl_path) else 0

        status = "✅" if imgs > 0 and lbls > 0 else "❌"
        print(f"  {status} {split:<6}: {imgs:>6} images | {lbls:>6} labels")

        if imgs == 0 or lbls == 0:
            all_ok = False

    return all_ok

# ============================================================
# STEP 3: TRAIN
# ============================================================
def train_yolo(yaml_path):
    print(f"\n🚀 Starting YOLOv8 Training...")
    print(f"   Model      : YOLOv8m (medium)")
    print(f"   Epochs     : {EPOCHS}")
    print(f"   Batch size : {BATCH_SIZE}")
    print(f"   Image size : {IMG_SIZE}")
    print("=" * 55)

    model = YOLO('yolov8m.pt')

    results = model.train(
        data         = yaml_path,
        epochs       = EPOCHS,
        batch        = BATCH_SIZE,
        imgsz        = IMG_SIZE,
        device       = DEVICE,
        patience     = PATIENCE,
        save         = True,
        project      = SAVE_PATH,
        name         = 'yolov8_multicrop',
        pretrained   = True,
        optimizer    = 'AdamW',
        lr0          = 0.001,
        lrf          = 0.01,
        weight_decay = 0.0005,
        warmup_epochs= 3,
        cos_lr       = True,
        augment      = True,
        hsv_h        = 0.015,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        degrees      = 10.0,
        translate    = 0.1,
        scale        = 0.5,
        flipud       = 0.5,
        fliplr       = 0.5,
        mosaic       = 1.0,
        mixup        = 0.1,
        workers      = 4,
        verbose      = True,
    )
    return model, results

# ============================================================
# STEP 4: EVALUATE
# ============================================================
def evaluate_model(model, yaml_path):
    print("\n📊 Evaluating on Test Set...")

    test_results = model.val(
        data    = yaml_path,
        split   = 'test',
        device  = DEVICE,
        verbose = True,
    )

    map50    = test_results.box.map50
    map5095  = test_results.box.map
    prec     = test_results.box.mp
    recall   = test_results.box.mr

    print(f"\n  📈 Detection Results:")
    print(f"  mAP@0.5      : {map50:.4f}")
    print(f"  mAP@0.5:0.95 : {map5095:.4f}")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Recall       : {recall:.4f}")

    with open(os.path.join(SAVE_PATH, 'test_results.txt'), 'w') as f:
        f.write("YOLOv8 Detection Results\n")
        f.write("Blockchain in Agriculture - Q1 Paper\n")
        f.write("="*40 + "\n")
        f.write(f"mAP@0.5      : {map50:.4f}\n")
        f.write(f"mAP@0.5:0.95 : {map5095:.4f}\n")
        f.write(f"Precision    : {prec:.4f}\n")
        f.write(f"Recall       : {recall:.4f}\n")

    return test_results

# ============================================================
# STEP 5: PLOT
# ============================================================
def plot_results():
    import pandas as pd
    results_csv = os.path.join(SAVE_PATH, 'yolov8_multicrop/results.csv')
    if not os.path.exists(results_csv):
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    metrics = [
        ('train/box_loss',   'Train Box Loss',    '#e74c3c'),
        ('train/cls_loss',   'Train Class Loss',  '#e67e22'),
        ('val/box_loss',     'Val Box Loss',      '#3498db'),
        ('val/cls_loss',     'Val Class Loss',    '#2ecc71'),
        ('metrics/mAP50',    'mAP@0.5',           '#9b59b6'),
        ('metrics/mAP50-95', 'mAP@0.5:0.95',      '#1abc9c'),
    ]

    for ax, (col, title, color) in zip(axes.flatten(), metrics):
        if col in df.columns:
            ax.plot(df[col], color=color, linewidth=2)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)

    plt.suptitle('YOLOv8 Training — Blockchain in Agriculture',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart_path = os.path.join(SAVE_PATH, 'yolo_training_curves.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Training curves saved!")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # Fix and load yaml
    yaml_path, num_classes, class_names = fix_yaml()

    # Check dataset
    if not check_dataset():
        print("\n❌ Dataset check failed!")
        exit(1)

    # Train
    model, results = train_yolo(yaml_path)

    # Evaluate
    test_results = evaluate_model(model, yaml_path)

    # Plot
    plot_results()

    print(f"\n{'='*55}")
    print(f"  DETECTION COMPLETE!")
    print(f"  mAP@0.5      : {test_results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {test_results.box.map:.4f}")
    print(f"  Model saved  : {SAVE_PATH}/yolov8_multicrop/weights/best.pt")
    print(f"{'='*55}")
    print(f"\n✅ Phase 2A Done! Ready for Phase 2B: U-Net Segmentation")
