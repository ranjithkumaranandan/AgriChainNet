# ============================================================
# PHASE 1 - STEP 1: DATASET EXPLORATION & VISUALIZATION
# Blockchain in Agriculture - Q1 Journal Paper
# Run this in: D:\September\
# ============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION - Update paths if needed
# ============================================================
BASE_PATH = "D:/September"

DATASETS = {
    "PlantVillage": os.path.join(BASE_PATH, "plantvillage"),
    "Multi_Crop": os.path.join(BASE_PATH, "Multi-Crop Disease Dataset"),
    "Leaf_Disease": os.path.join(BASE_PATH, "Plant Leaf Disease Recognition Dataset"),
    "Rice_Leaf": os.path.join(BASE_PATH, "Rice Leaf and Crop Disease Detection Dataset"),
}

OUTPUT_PATH = os.path.join(BASE_PATH, "exploration_results")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================
# FUNCTION 1: Count images in each dataset
# ============================================================
def count_images(dataset_path):
    class_counts = {}
    total = 0
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    for root, dirs, files in os.walk(dataset_path):
        images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
        if images:
            class_name = os.path.basename(root)
            class_counts[class_name] = len(images)
            total += len(images)

    return class_counts, total

# ============================================================
# FUNCTION 2: Check image sizes
# ============================================================
def check_image_sizes(dataset_path, sample_limit=100):
    sizes = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    count = 0

    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in image_extensions:
                try:
                    img_path = os.path.join(root, f)
                    img = Image.open(img_path)
                    sizes.append(img.size)  # (width, height)
                    count += 1
                    if count >= sample_limit:
                        break
                except:
                    pass
        if count >= sample_limit:
            break

    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        return {
            "min_size": (min(widths), min(heights)),
            "max_size": (max(widths), max(heights)),
            "avg_size": (int(np.mean(widths)), int(np.mean(heights))),
            "most_common": max(set(sizes), key=sizes.count)
        }
    return {}

# ============================================================
# FUNCTION 3: Plot class distribution
# ============================================================
def plot_class_distribution(class_counts, dataset_name, output_path):
    if not class_counts:
        return

    # Sort by count
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0][:20] for item in sorted_items]  # Truncate long names
    counts = [item[1] for item in sorted_items]

    # Color gradient
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(classes)))

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(classes, counts, color=colors, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=8)

    ax.set_xlabel('Number of Images', fontsize=12)
    ax.set_title(f'{dataset_name}\nClass Distribution ({sum(counts):,} total images)',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()

    save_path = os.path.join(output_path, f'{dataset_name}_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Distribution chart saved: {save_path}")

# ============================================================
# FUNCTION 4: Show sample images from a dataset
# ============================================================
def show_sample_images(dataset_path, dataset_name, output_path, samples_per_class=2):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    class_samples = defaultdict(list)

    for root, dirs, files in os.walk(dataset_path):
        images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
        if images:
            class_name = os.path.basename(root)
            for img_file in images[:samples_per_class]:
                class_samples[class_name].append(os.path.join(root, img_file))

    # Limit to first 12 classes for display
    selected_classes = list(class_samples.keys())[:12]
    n_classes = len(selected_classes)

    if n_classes == 0:
        return

    cols = 6
    rows = (n_classes * samples_per_class + cols - 1) // cols
    rows = max(rows, 2)

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3))
    axes = axes.flatten()

    idx = 0
    for class_name in selected_classes:
        for img_path in class_samples[class_name][:samples_per_class]:
            if idx >= len(axes):
                break
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((150, 150))
                axes[idx].imshow(img)
                axes[idx].set_title(class_name[:15], fontsize=7, pad=2)
                axes[idx].axis('off')
            except:
                axes[idx].axis('off')
            idx += 1

    # Hide unused axes
    for i in range(idx, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'{dataset_name} - Sample Images', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_path, f'{dataset_name}_samples.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Sample images saved: {save_path}")

# ============================================================
# FUNCTION 5: Summary report
# ============================================================
def print_summary_report(all_results):
    print("\n" + "="*60)
    print("       DATASET EXPLORATION SUMMARY REPORT")
    print("       Blockchain in Agriculture - Q1 Paper")
    print("="*60)

    total_images = 0
    for name, result in all_results.items():
        print(f"\n📁 {name}")
        print(f"   Total Images  : {result['total']:,}")
        print(f"   Total Classes : {result['classes']}")
        print(f"   Avg Img Size  : {result['sizes'].get('avg_size', 'N/A')}")
        print(f"   Size Range    : {result['sizes'].get('min_size', 'N/A')} → {result['sizes'].get('max_size', 'N/A')}")
        total_images += result['total']

    print(f"\n{'='*60}")
    print(f"  TOTAL IMAGES ACROSS ALL DATASETS: {total_images:,}")
    print(f"{'='*60}\n")

# ============================================================
# FUNCTION 6: Combined overview pie chart
# ============================================================
def plot_combined_overview(all_results, output_path):
    names = list(all_results.keys())
    totals = [all_results[n]['total'] for n in names]

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(totals, labels=names, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 10})
    ax1.set_title('Image Distribution\nAcross Datasets', fontsize=13, fontweight='bold')

    # Bar chart - classes
    classes_count = [all_results[n]['classes'] for n in names]
    bars = ax2.bar(names, classes_count, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_title('Number of Classes\nPer Dataset', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Number of Classes')
    for bar, val in zip(bars, classes_count):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(val), ha='center', fontsize=11, fontweight='bold')
    ax2.set_xticklabels(names, rotation=15, ha='right')

    plt.suptitle('Combined Dataset Overview\nBlockchain in Agriculture Paper',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_path, 'combined_overview.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Combined overview saved: {save_path}")

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("  PHASE 1 - STEP 1: EXPLORING YOUR DATASETS")
    print("="*60)

    all_results = {}

    for dataset_name, dataset_path in DATASETS.items():
        print(f"\n🔍 Exploring: {dataset_name}")
        print(f"   Path: {dataset_path}")

        if not os.path.exists(dataset_path):
            print(f"   ⚠️  Path not found! Check DATASETS config above.")
            continue

        # Count images
        print(f"   Counting images...")
        class_counts, total = count_images(dataset_path)

        # Check sizes
        print(f"   Checking image sizes (sampling 100 images)...")
        sizes = check_image_sizes(dataset_path)

        # Plot distribution
        print(f"   Plotting class distribution...")
        plot_class_distribution(class_counts, dataset_name, OUTPUT_PATH)

        # Show samples
        print(f"   Saving sample images...")
        show_sample_images(dataset_path, dataset_name, OUTPUT_PATH)

        all_results[dataset_name] = {
            'total': total,
            'classes': len(class_counts),
            'class_counts': class_counts,
            'sizes': sizes
        }

    # Print summary
    if all_results:
        print_summary_report(all_results)
        plot_combined_overview(all_results, OUTPUT_PATH)

    print(f"\n✅ ALL DONE! Check results in: {OUTPUT_PATH}")
    print("="*60)