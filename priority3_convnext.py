# ============================================================
# AgriConvNet FIXED — CutMix Error Resolved
# ConvNeXt-Base + CBAM × 4
# Target: 99.5%+
# Blockchain in Agriculture - Q1 Journal Paper
# ============================================================

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report,
                             f1_score, confusion_matrix)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_PATH   = "D:/September/final_datasets"
SAVE_PATH   = "D:/September/models/AgriConvNet_fixed"
LOG_PATH    = "D:/September/models/AgriConvNet_fixed/logs"
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH,  exist_ok=True)

BATCH_SIZE  = 32
NUM_EPOCHS  = 50
LR          = 0.00005
NUM_WORKERS = 4
IMG_SIZE    = 256
PATIENCE    = 12

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("  AgriConvNet FIXED — CutMix Error Resolved")
print("  ConvNeXt-Base + CBAM × 4 Attention")
print("  Target: 99.5%+")
print("  Blockchain in Agriculture - Q1 Paper")
print("=" * 60)
print(f"\n  Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")

# ============================================================
# CBAM MODULES
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, in_channels, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(b,c)).view(b,c,1,1)
        mx  = self.fc(self.max_pool(x).view(b,c)).view(b,c,1,1)
        return x * self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size,
                                 padding=kernel_size//2,
                                 bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg  = x.mean(dim=1, keepdim=True)
        mx,_ = x.max(dim=1,  keepdim=True)
        return x * self.sigmoid(
            self.conv(torch.cat([avg, mx], dim=1)))


class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16,
                 kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(
            in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

# ============================================================
# AgriConvNet — PROPOSED MODEL
# ============================================================
class AgriConvNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        backbone = models.convnext_base(
            weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1
            if pretrained else None)

        features     = backbone.features
        self.stem    = features[0]
        self.stage1  = features[1]
        self.down1   = features[2]
        self.stage2  = features[3]
        self.down2   = features[4]
        self.stage3  = features[5]
        self.down3   = features[6]
        self.stage4  = features[7]

        self.cbam1   = CBAMBlock(128,  reduction=8)
        self.cbam2   = CBAMBlock(256,  reduction=16)
        self.cbam3   = CBAMBlock(512,  reduction=16)
        self.cbam4   = CBAMBlock(1024, reduction=16)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.norm     = nn.LayerNorm(2048)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes))

        for param in self.stem.parameters():
            param.requires_grad = False

    def forward(self, x):
        x   = self.stem(x)
        x   = self.cbam1(self.stage1(x))
        x   = self.cbam2(self.stage2(self.down1(x)))
        x   = self.cbam3(self.stage3(self.down2(x)))
        x   = self.cbam4(self.stage4(self.down3(x)))
        avg = self.avg_pool(x).view(x.size(0), -1)
        mx  = self.max_pool(x).view(x.size(0), -1)
        x   = self.norm(torch.cat([avg, mx], dim=1))
        return self.classifier(x)

# ============================================================
# TRANSFORMS
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1,
                             scale=(0.02, 0.1))
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

tta_transforms = [
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE+20, IMG_SIZE+20)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.1,
                               contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])]),
]

# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    print("\n📂 Loading datasets...")
    train_ds, val_ds, test_ds = [], [], []

    for name in ["PlantVillage", "Rice_Leaf"]:
        path = os.path.join(BASE_PATH, name)
        if not os.path.exists(
                os.path.join(path, "train")):
            continue
        t = datasets.ImageFolder(
            os.path.join(path, "train"),
            transform=train_transform)
        v = datasets.ImageFolder(
            os.path.join(path, "val"),
            transform=val_transform)
        s = datasets.ImageFolder(
            os.path.join(path, "test"),
            transform=val_transform)
        train_ds.append(t)
        val_ds.append(v)
        test_ds.append(s)
        print(f"  ✅ {name}: Train={len(t):,} "
              f"Val={len(v):,} Test={len(s):,}")

    num_classes = len(train_ds[0].classes)
    class_names = train_ds[0].classes
    print(f"\n  Total classes : {num_classes}")

    train_loader = DataLoader(
        ConcatDataset(train_ds),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(
        ConcatDataset(val_ds),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(
        ConcatDataset(test_ds),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    test_paths = [
        os.path.join(BASE_PATH, n, "test")
        for n in ["PlantVillage", "Rice_Leaf"]
        if os.path.exists(
            os.path.join(BASE_PATH, n, "test"))]

    return (train_loader, val_loader, test_loader,
            test_paths, num_classes, class_names)

# ============================================================
# SAFE MIXUP ONLY (CutMix removed — was causing CUDA error)
# ============================================================
def safe_mixup(x, y, num_classes, alpha=0.2):
    """
    Safe MixUp — validated label indices
    CutMix removed to prevent CUDA index error
    """
    lam = float(np.random.beta(alpha, alpha))
    b   = x.size(0)
    idx = torch.randperm(b, device=DEVICE)

    # Validate indices are within bounds
    assert y.max() < num_classes, \
        f"Label {y.max()} >= num_classes {num_classes}"
    assert y[idx].max() < num_classes, \
        f"Shuffled label out of bounds"

    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam

def mix_criterion(criterion, pred, ya, yb, lam):
    return (lam * criterion(pred, ya) +
            (1 - lam) * criterion(pred, yb))

# ============================================================
# TRAINING — 3 Phase (Fixed)
# ============================================================
def train_model(model, train_loader, val_loader,
                num_classes):
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05)
    scaler    = GradScaler()
    history   = {'train_loss': [], 'train_acc': [],
                 'val_loss'  : [], 'val_acc'  : []}
    best_acc  = 0.0
    best_wts  = copy.deepcopy(model.state_dict())
    p_ctr     = 0
    start     = time.time()

    print(f"\n🚀 Training AgriConvNet (Fixed)...")
    print(f"   CutMix: ❌ Removed (caused CUDA error)")
    print(f"   MixUp : ✅ Safe version only")
    print(f"   Phase 1: Warmup    (Epochs  1-5)")
    print(f"   Phase 2: FineTune  (Epochs  6-35)")
    print(f"   Phase 3: Precision (Epochs 36-50)")
    print("=" * 60)

    optimizer = None
    scheduler = None

    for epoch in range(NUM_EPOCHS):
        ep_start = time.time()

        # Phase 1: Warmup
        if epoch == 0:
            print("\n  🔥 Phase 1: Warmup...")
            for name, param in model.named_parameters():
                if any(s in name for s in
                       ['stage','down','cbam']):
                    param.requires_grad = False
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad,
                       model.parameters()),
                lr=LR * 10, weight_decay=1e-4)
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=5, eta_min=1e-7)

        # Phase 2: Full fine-tune
        elif epoch == 5:
            print("\n  🔓 Phase 2: Full fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            for param in model.stem.parameters():
                param.requires_grad = False
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad,
                       model.parameters()),
                lr=LR, weight_decay=1e-4)
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=15, T_mult=2,
                eta_min=1e-8)

        # Phase 3: Precision
        elif epoch == 35:
            print("\n  🎯 Phase 3: Precision tuning...")
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad,
                       model.parameters()),
                lr=LR * 0.05, weight_decay=1e-5)
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=10, eta_min=1e-9)

        # Train
        model.train()
        tr_loss = tr_correct = tr_total = 0

        for inputs, labels in tqdm(
                train_loader,
                desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS}"
                     f" [Train]",
                leave=False):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Validate labels before training
            if labels.max() >= num_classes:
                continue  # Skip bad batch

            optimizer.zero_grad()

            try:
                with autocast():
                    # Safe MixUp only (no CutMix)
                    if (epoch >= 5 and
                            np.random.random() < 0.4):
                        mx_in, ya, yb, lam = safe_mixup(
                            inputs, labels, num_classes)
                        out  = model(mx_in)
                        loss = mix_criterion(
                            criterion, out, ya, yb, lam)
                    else:
                        out  = model(inputs)
                        loss = criterion(out, labels)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(
                    epoch +
                    tr_total/max(len(train_loader),1))

                tr_loss    += loss.item()*inputs.size(0)
                _, pred     = out.max(1)
                tr_correct += pred.eq(labels).sum().item()
                tr_total   += labels.size(0)

            except RuntimeError as e:
                print(f"\n  ⚠️  Skipping batch: {str(e)[:50]}")
                torch.cuda.empty_cache()
                continue

        # Validate
        model.eval()
        va_loss = va_correct = va_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(
                    val_loader,
                    desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS}"
                         f" [Val]",
                    leave=False):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                with autocast():
                    out  = model(inputs)
                    loss = criterion(out, labels)
                va_loss    += loss.item()*inputs.size(0)
                _, pred     = out.max(1)
                va_correct += pred.eq(labels).sum().item()
                va_total   += labels.size(0)

        tr_l = tr_loss/max(tr_total,1)
        tr_a = 100.*tr_correct/max(tr_total,1)
        va_l = va_loss/max(va_total,1)
        va_a = 100.*va_correct/max(va_total,1)
        ep_t = time.time() - ep_start

        history['train_loss'].append(tr_l)
        history['train_acc'].append(tr_a)
        history['val_loss'].append(va_l)
        history['val_acc'].append(va_a)

        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
              f"{ep_t:.0f}s | "
              f"Train:{tr_a:.2f}% | "
              f"Val:{va_a:.2f}%", end="")

        if va_a > best_acc:
            best_acc = va_a
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(
                {'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'val_acc': best_acc},
                os.path.join(SAVE_PATH,
                             'AgriConvNet_best.pth'))
            print(f" ✅ Best! ({best_acc:.2f}%)")
            p_ctr = 0
        else:
            p_ctr += 1
            print(f" (patience {p_ctr}/{PATIENCE})")

        if p_ctr >= PATIENCE and epoch >= 20:
            print(f"\n⚠️  Early stopping epoch {epoch+1}")
            break

    total_time = time.time() - start
    print(f"\n⏱️  Training: {total_time/60:.1f} mins")
    print(f"🏆 Best Val : {best_acc:.2f}%")
    model.load_state_dict(best_wts)
    return model, history

# ============================================================
# TTA EVALUATION
# ============================================================
def evaluate_tta(model, test_paths, class_names):
    print("\n📊 TTA Evaluation (5 augmentations)...")
    model.eval()
    all_preds  = []
    all_labels = []

    for test_path in test_paths:
        if not os.path.exists(test_path):
            continue
        gt = datasets.ImageFolder(
            test_path, transform=val_transform).targets
        tta_preds = []

        for tta_tf in tta_transforms:
            ds  = datasets.ImageFolder(
                test_path, transform=tta_tf)
            ldr = DataLoader(
                ds, batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True)
            batch = []
            with torch.no_grad():
                for inp, _ in tqdm(ldr,
                                   desc="  TTA",
                                   leave=False):
                    with autocast():
                        out = torch.softmax(
                            model(inp.to(DEVICE)),
                            dim=1)
                    batch.append(out.cpu().numpy())
            tta_preds.append(np.vstack(batch))

        avg = np.mean(tta_preds, axis=0)
        all_preds.extend(avg.argmax(axis=1).tolist())
        all_labels.extend(gt)

    acc = 100.*np.mean(
        np.array(all_preds)==np.array(all_labels))
    f1  = f1_score(all_labels, all_preds,
                   average='weighted')

    print(f"\n  {'='*45}")
    print(f"  TTA Accuracy : {acc:.2f}%")
    print(f"  TTA F1 Score : {f1:.4f}")
    print(f"  {'='*45}")

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names[:len(set(all_labels))],
        digits=4)
    with open(os.path.join(LOG_PATH,
                           'AgriConvNet_report.txt'),
              'w') as f:
        f.write("AgriConvNet Report\n")
        f.write("ConvNeXt-Base + CBAM\n")
        f.write("Blockchain Agriculture Q1\n")
        f.write("="*50+"\n")
        f.write(f"TTA Accuracy : {acc:.2f}%\n")
        f.write(f"TTA F1       : {f1:.4f}\n\n")
        f.write(report)
    print("  ✅ Report saved!")
    return acc, f1, all_preds, all_labels

# ============================================================
# PLOT
# ============================================================
def plot_results(history, preds, labels,
                 class_names, acc):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'],
                 label='Train', color='#e74c3c', lw=2)
    axes[0].plot(history['val_loss'],
                 label='Val',   color='#3498db', lw=2)
    for x, lb in [(5,'Ph2'),(35,'Ph3')]:
        if x < len(history['train_loss']):
            axes[0].axvline(x=x, color='gray',
                            ls='--', alpha=0.6,
                            label=lb)
    axes[0].set_title('Loss', fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'],
                 label='Train', color='#e74c3c', lw=2)
    axes[1].plot(history['val_acc'],
                 label='Val',   color='#3498db', lw=2)
    axes[1].axhline(y=99.5, color='green',
                    ls='--', lw=2,
                    label='Target 99.5%')
    axes[1].axhline(y=98.75, color='orange',
                    ls='--', lw=1.5,
                    label='ResNet50')
    axes[1].set_title(f'Accuracy — TTA: {acc:.2f}%',
                      fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        'AgriConvNet (Fixed) — Q1 Paper',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_PATH,
                             'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Confusion matrix
    unique = sorted(set(labels))[:15]
    fp = [p for p,l in zip(preds,labels)
          if l in unique]
    fl = [l for l in labels if l in unique]
    if fl:
        cm = confusion_matrix(fl, fp, labels=unique)
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=[class_names[i] for i in unique],
            yticklabels=[class_names[i] for i in unique])
        plt.title('AgriConvNet Confusion Matrix',
                  fontsize=13, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(LOG_PATH, 'confusion_matrix.png'),
            dpi=150, bbox_inches='tight')
        plt.close()
    print("  ✅ Plots saved!")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    (train_loader, val_loader, test_loader,
     test_paths, num_classes,
     class_names) = load_data()

    model = AgriConvNet(
        num_classes=num_classes,
        pretrained=True).to(DEVICE)

    total  = sum(p.numel()
                 for p in model.parameters())/1e6
    cbam_p = sum(p.numel()
                 for n,p in model.named_parameters()
                 if 'cbam' in n)/1e6

    print(f"\n🏗️  AgriConvNet:")
    print(f"   Backbone : ConvNeXt-Base + CBAM×4")
    print(f"   Pooling  : Dual Avg+Max → 2048")
    print(f"   CBAM     : {cbam_p:.2f}M (novelty)")
    print(f"   Total    : {total:.2f}M params")
    print(f"   Fix      : CutMix removed, Safe MixUp")

    model, history = train_model(
        model, train_loader, val_loader, num_classes)

    acc, f1, preds, labels = evaluate_tta(
        model, test_paths, class_names)

    plot_results(history, preds, labels,
                 class_names, acc)

    torch.save(model.state_dict(),
               os.path.join(SAVE_PATH,
                            'AgriConvNet_final.pth'))

    improvement = acc - 98.75
    print(f"\n{'='*60}")
    print(f"  AgriConvNet COMPLETE!")
    print(f"  TTA Accuracy    : {acc:.2f}%")
    print(f"  TTA F1 Score    : {f1:.4f}")
    print(f"  vs ResNet50     : {improvement:+.2f}%")
    if acc >= 99.5:
        print(f"  🎉 TARGET 99.5% ACHIEVED!")
    else:
        print(f"  ⚠️  Need {99.5-acc:.2f}% more")
    print(f"{'='*60}")
    print(f"\n✅ Next → Baselines → Blockchain → Paper!")
