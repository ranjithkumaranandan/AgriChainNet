# ============================================================
# PRIORITY 2 - BASELINE MODELS COMPARISON
# Models: VGG16, MobileNetV2, EfficientNetB0, InceptionV3
# Dataset: PlantVillage + Rice_Leaf
# Blockchain in Agriculture - Q1 Journal Paper
# ============================================================

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_PATH  = "D:/September/final_datasets"
SAVE_PATH  = "D:/September/models/baselines"
LOG_PATH   = "D:/September/models/baselines/logs"
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH,  exist_ok=True)

BATCH_SIZE  = 32
NUM_EPOCHS  = 20       # Fewer epochs for baselines
LR          = 0.001
NUM_WORKERS = 4
IMG_SIZE    = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("  PRIORITY 2 - BASELINE MODELS COMPARISON")
print("  Blockchain in Agriculture - Q1 Paper")
print("=" * 60)
print(f"\n  Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")

# ============================================================
# TRANSFORMS
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================================
# LOAD DATASETS
# ============================================================
def load_data():
    print("\n📂 Loading datasets...")
    train_ds, val_ds, test_ds = [], [], []

    for name in ["PlantVillage", "Rice_Leaf"]:
        path = os.path.join(BASE_PATH, name)
        if not os.path.exists(os.path.join(path, "train")):
            print(f"  ⚠️  {name} not found, skipping.")
            continue
        t = datasets.ImageFolder(os.path.join(path, "train"),
                                 transform=train_transform)
        v = datasets.ImageFolder(os.path.join(path, "val"),
                                 transform=val_transform)
        s = datasets.ImageFolder(os.path.join(path, "test"),
                                 transform=val_transform)
        train_ds.append(t)
        val_ds.append(v)
        test_ds.append(s)
        print(f"  ✅ {name}: Train={len(t):,} Val={len(v):,} "
              f"Test={len(s):,} Classes={len(t.classes)}")

    num_classes = len(train_ds[0].classes)
    class_names = train_ds[0].classes

    train_loader = DataLoader(ConcatDataset(train_ds),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(ConcatDataset(val_ds),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(ConcatDataset(test_ds),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\n  Total classes: {num_classes}")
    return train_loader, val_loader, test_loader, num_classes, class_names

# ============================================================
# BUILD BASELINE MODELS
# ============================================================
def build_model(model_name, num_classes):
    if model_name == "VGG16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(1280, num_classes)

    elif model_name == "EfficientNetB0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(1280, num_classes)

    elif model_name == "InceptionV3":
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        model.fc    = nn.Linear(2048, num_classes)
        model.AuxLogits.fc = nn.Linear(768, num_classes)

    elif model_name == "ResNet50":
        # Your existing model — for comparison
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for name, param in model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad) / 1e6
    print(f"  {model_name:<15} Params:{params:.1f}M  "
          f"Trainable:{trainable:.1f}M")
    return model.to(DEVICE)

# ============================================================
# TRAIN ONE EPOCH
# ============================================================
def train_epoch(model, loader, optimizer, criterion,
                scaler, is_inception=False):
    model.train()
    total_loss = correct = total = 0

    for inputs, labels in tqdm(loader, desc="  Train", leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with autocast():
            if is_inception:
                outputs, aux = model(inputs)
                loss = criterion(outputs, labels) + \
                       0.4 * criterion(aux, labels)
            else:
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        _, pred     = outputs.max(1)
        correct    += pred.eq(labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, 100. * correct / total

# ============================================================
# VALIDATE ONE EPOCH
# ============================================================
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  Val  ", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            with autocast():
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, pred     = outputs.max(1)
            correct    += pred.eq(labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, 100. * correct / total

# ============================================================
# TEST MODEL
# ============================================================
def test_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  Test ", leave=False):
            inputs = inputs.to(DEVICE)
            with autocast():
                outputs = model(inputs)
            _, pred = outputs.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    f1  = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1, all_preds, all_labels

# ============================================================
# TRAIN ONE BASELINE MODEL
# ============================================================
def train_baseline(model_name, model, train_loader,
                   val_loader, test_loader):
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    is_inception = (model_name == "InceptionV3")
    criterion    = nn.CrossEntropyLoss()
    optimizer    = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4)
    scheduler    = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS,
                                     eta_min=1e-6)
    scaler       = GradScaler()

    best_acc      = 0.0
    best_wts      = copy.deepcopy(model.state_dict())
    patience_ctr  = 0
    patience      = 5
    history       = {'train_acc': [], 'val_acc': [],
                     'train_loss': [], 'val_loss': []}
    start         = time.time()

    for epoch in range(NUM_EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer,
                                      criterion, scaler, is_inception)
        va_loss, va_acc = val_epoch(model, val_loader, criterion)
        scheduler.step()

        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)

        print(f"  Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
              f"Train:{tr_acc:.2f}% Val:{va_acc:.2f}%", end="")

        if va_acc > best_acc:
            best_acc = va_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),
                       os.path.join(SAVE_PATH,
                                    f'{model_name}_best.pth'))
            print(f" ✅ Best!")
            patience_ctr = 0
        else:
            patience_ctr += 1
            print(f" (patience {patience_ctr}/{patience})")

        if patience_ctr >= patience:
            print(f"  ⚠️  Early stopping at epoch {epoch+1}")
            break

    elapsed = (time.time() - start) / 60
    model.load_state_dict(best_wts)

    # Test
    test_acc, test_f1, preds, labels = test_model(model, test_loader)

    print(f"\n  ✅ {model_name} DONE!")
    print(f"     Test Accuracy : {test_acc:.2f}%")
    print(f"     Test F1 Score : {test_f1:.4f}")
    print(f"     Training Time : {elapsed:.1f} mins")

    return {
        'model'   : model_name,
        'acc'     : test_acc,
        'f1'      : test_f1,
        'time'    : elapsed,
        'history' : history,
        'preds'   : preds,
        'labels'  : labels,
    }

# ============================================================
# PLOT COMPARISON
# ============================================================
def plot_comparison(all_results):
    names  = [r['model'] for r in all_results]
    accs   = [r['acc']   for r in all_results]
    f1s    = [r['f1']    for r in all_results]
    times  = [r['time']  for r in all_results]

    colors = ['#e74c3c', '#3498db', '#2ecc71',
              '#f39c12', '#9b59b6']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Accuracy
    bars = axes[0].bar(names, accs, color=colors, edgecolor='white')
    axes[0].set_title('Test Accuracy (%)', fontweight='bold', fontsize=13)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim([min(accs) - 5, 100])
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f'{val:.2f}%', ha='center',
                     fontweight='bold', fontsize=9)
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].grid(True, alpha=0.3, axis='y')

    # F1 Score
    bars = axes[1].bar(names, f1s, color=colors, edgecolor='white')
    axes[1].set_title('F1 Score (Weighted)', fontweight='bold', fontsize=13)
    axes[1].set_ylabel('F1 Score')
    axes[1].set_ylim([min(f1s) - 0.05, 1.0])
    for bar, val in zip(bars, f1s):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.003,
                     f'{val:.4f}', ha='center',
                     fontweight='bold', fontsize=9)
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Training Time
    bars = axes[2].bar(names, times, color=colors, edgecolor='white')
    axes[2].set_title('Training Time (mins)', fontweight='bold', fontsize=13)
    axes[2].set_ylabel('Time (minutes)')
    for bar, val in zip(bars, times):
        axes[2].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f'{val:.1f}m', ha='center',
                     fontweight='bold', fontsize=9)
    axes[2].tick_params(axis='x', rotation=15)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Baseline Model Comparison — Blockchain in Agriculture',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_PATH, 'baseline_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✅ Comparison chart saved!")

# ============================================================
# SAVE SUMMARY TABLE
# ============================================================
def save_summary(all_results):
    path = os.path.join(LOG_PATH, 'baseline_results.txt')
    with open(path, 'w') as f:
        f.write("BASELINE MODEL COMPARISON\n")
        f.write("Blockchain in Agriculture - Q1 Paper\n")
        f.write("="*55 + "\n")
        f.write(f"{'Model':<18} {'Accuracy':>10} {'F1 Score':>10} "
                f"{'Time(min)':>10}\n")
        f.write("-"*55 + "\n")
        for r in all_results:
            f.write(f"{r['model']:<18} {r['acc']:>9.2f}% "
                    f"{r['f1']:>10.4f} {r['time']:>10.1f}\n")
        f.write("="*55 + "\n")

    print(f"\n  ✅ Summary saved: {path}")

    # Print to console
    print(f"\n{'='*55}")
    print(f"  BASELINE COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Model':<18} {'Accuracy':>10} {'F1':>8} {'Time':>8}")
    print(f"  {'-'*50}")
    for r in sorted(all_results, key=lambda x: x['acc'], reverse=True):
        marker = " ← BEST" if r['acc'] == max(x['acc'] for x in all_results) \
                 else ""
        print(f"  {r['model']:<18} {r['acc']:>9.2f}% "
              f"{r['f1']:>8.4f} {r['time']:>6.1f}m{marker}")
    print(f"{'='*55}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # Load data
    train_loader, val_loader, test_loader, \
        num_classes, class_names = load_data()

    # Define baseline models to compare
    # ResNet50 already done (98.75%) — AgriAttentionNet is our proposed
    BASELINES = [
        "VGG16",
        "MobileNetV2",
        "EfficientNetB0",
    ]

    all_results = []

    print(f"\n🏗️  Building {len(BASELINES)} models...")
    for name in BASELINES:
        model  = build_model(name, num_classes)
        result = train_baseline(name, model, train_loader,
                                val_loader, test_loader)
        all_results.append(result)

        # Free GPU memory between models
        del model
        torch.cuda.empty_cache()

    # Plot and save
    plot_comparison(all_results)
    save_summary(all_results)

    print(f"\n✅ Priority 2 Complete!")
    print(f"   Next → Priority 3: CBAM Attention ResNet50")
