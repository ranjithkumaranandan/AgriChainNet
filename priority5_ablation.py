# ============================================================
# PRIORITY 5 - ABLATION STUDY (FAST VERSION)
# Only 2 most important configs
# Blockchain in Agriculture - Q1 Journal Paper
# ============================================================

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

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
# CONFIGURATION — FAST
# ============================================================
BASE_PATH   = "D:/September/final_datasets"
SAVE_PATH   = "D:/September/models/ablation_fast"
LOG_PATH    = "D:/September/models/ablation_fast/logs"
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH,  exist_ok=True)

BATCH_SIZE  = 64
NUM_EPOCHS  = 8      # Reduced from 15
LR          = 0.0001
NUM_WORKERS = 4
IMG_SIZE    = 224
PATIENCE    = 3      # Reduced from 5

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "cpu")
print("=" * 60)
print("  ABLATION STUDY (FAST) — 2 Key Configs")
print("  Config 1: No Attention vs Config 2: Full CBAM")
print("  Blockchain in Agriculture - Q1 Paper")
print("=" * 60)
print(f"\n  Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")

# ============================================================
# ATTENTION MODULES
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
        avg = self.fc(
            self.avg_pool(x).view(b,c)).view(b,c,1,1)
        mx  = self.fc(
            self.max_pool(x).view(b,c)).view(b,c,1,1)
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
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class IdentityBlock(nn.Module):
    def forward(self, x):
        return x

# ============================================================
# MODEL
# ============================================================
class AgriConvNetAblation(nn.Module):
    def __init__(self, num_classes,
                 use_cbam=True, pretrained=True):
        super().__init__()

        backbone = models.convnext_small(
            weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1
            if pretrained else None)

        f = backbone.features
        self.stem   = f[0]
        self.stage1 = f[1]
        self.down1  = f[2]
        self.stage2 = f[3]
        self.down2  = f[4]
        self.stage3 = f[5]
        self.down3  = f[6]
        self.stage4 = f[7]

        # CBAM or Identity
        if use_cbam:
            self.attn1 = CBAMBlock(96,  reduction=8)
            self.attn2 = CBAMBlock(192, reduction=16)
            self.attn3 = CBAMBlock(384, reduction=16)
            self.attn4 = CBAMBlock(768, reduction=16)
        else:
            self.attn1 = IdentityBlock()
            self.attn2 = IdentityBlock()
            self.attn3 = IdentityBlock()
            self.attn4 = IdentityBlock()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.norm     = nn.LayerNorm(1536)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1536, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(384, num_classes))

        for p in self.stem.parameters():
            p.requires_grad = False

    def forward(self, x):
        x   = self.stem(x)
        x   = self.attn1(self.stage1(x))
        x   = self.attn2(self.stage2(self.down1(x)))
        x   = self.attn3(self.stage3(self.down2(x)))
        x   = self.attn4(self.stage4(self.down3(x)))
        avg = self.avg_pool(x).view(x.size(0), -1)
        mx  = self.max_pool(x).view(x.size(0), -1)
        x   = self.norm(torch.cat([avg, mx], dim=1))
        return self.classifier(x)

# ============================================================
# TRANSFORMS
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE+16, IMG_SIZE+16)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2,
                           saturation=0.2),
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
        print(f"  ✅ {name}: {len(t):,} train")

    num_classes = len(train_ds[0].classes)
    print(f"  Classes: {num_classes}")

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

    return (train_loader, val_loader,
            test_loader, num_classes)

# ============================================================
# MIXUP
# ============================================================
def safe_mixup(x, y, num_classes, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=DEVICE)
    if y.max() >= num_classes:
        return x, y, y, 1.0
    return lam*x+(1-lam)*x[idx], y, y[idx], lam

def mix_criterion(criterion, pred, ya, yb, lam):
    return lam*criterion(pred,ya)+(1-lam)*criterion(pred,yb)

# ============================================================
# TRAIN
# ============================================================
def train_config(model, train_loader, val_loader,
                 num_classes, config_name):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad,
               model.parameters()),
        lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)
    scaler    = GradScaler()

    best_acc  = 0.0
    best_wts  = copy.deepcopy(model.state_dict())
    p_ctr     = 0
    history   = {'val_acc': []}
    start     = time.time()

    for epoch in range(NUM_EPOCHS):
        if epoch == 2:
            for p in model.parameters():
                p.requires_grad = True
            for p in model.stem.parameters():
                p.requires_grad = False
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad,
                       model.parameters()),
                lr=LR, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=NUM_EPOCHS-2,
                eta_min=1e-8)

        # Train
        model.train()
        tr_correct = tr_total = 0

        for inputs, labels in tqdm(
                train_loader,
                desc=f"  [{config_name}] "
                     f"Ep {epoch+1}/{NUM_EPOCHS}",
                leave=False):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            if labels.max() >= num_classes:
                continue
            optimizer.zero_grad()

            try:
                with autocast():
                    if (epoch >= 2 and
                            np.random.random() < 0.3):
                        mi,ya,yb,lam = safe_mixup(
                            inputs,labels,num_classes)
                        out  = model(mi)
                        loss = mix_criterion(
                            criterion,out,ya,yb,lam)
                    else:
                        out  = model(inputs)
                        loss = criterion(out, labels)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                _, pred     = out.max(1)
                tr_correct += pred.eq(labels).sum().item()
                tr_total   += labels.size(0)

            except RuntimeError:
                torch.cuda.empty_cache()
                continue

        scheduler.step()

        # Validate
        model.eval()
        va_correct = va_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                with autocast():
                    out = model(inputs)
                _, pred     = out.max(1)
                va_correct += pred.eq(labels).sum().item()
                va_total   += labels.size(0)

        tr_a = 100.*tr_correct/max(tr_total,1)
        va_a = 100.*va_correct/max(va_total,1)
        history['val_acc'].append(va_a)

        print(f"  [{config_name}] "
              f"Ep {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Tr:{tr_a:.2f}% Va:{va_a:.2f}%",
              end="")

        if va_a > best_acc:
            best_acc = va_a
            best_wts = copy.deepcopy(model.state_dict())
            print(f" ✅ Best!")
            p_ctr = 0
        else:
            p_ctr += 1
            print(f" (p{p_ctr}/{PATIENCE})")
            if p_ctr >= PATIENCE and epoch >= 4:
                print(f"  ⚡ Early stop!")
                break

    elapsed = (time.time()-start)/60
    model.load_state_dict(best_wts)
    return model, best_acc, elapsed, history

# ============================================================
# TEST
# ============================================================
def test_model(model, test_loader):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(
                test_loader,
                desc="  Testing",
                leave=False):
            inputs = inputs.to(DEVICE)
            with autocast():
                out = model(inputs)
            _, pred = out.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = 100.*np.mean(
        np.array(all_preds)==np.array(all_labels))
    f1  = f1_score(all_labels, all_preds,
                   average='weighted')
    return acc, f1

# ============================================================
# PLOT
# ============================================================
def plot_ablation(results, histories):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    configs = [r['config'] for r in results]
    accs    = [r['acc']    for r in results]
    colors  = ['#e74c3c', '#2ecc71']

    bars = axes[0].bar(configs, accs,
                       color=colors,
                       edgecolor='white',
                       linewidth=2,
                       width=0.4)
    axes[0].set_ylim([min(accs)-5, 100])
    axes[0].set_title('Accuracy: With vs Without CBAM',
                      fontweight='bold', fontsize=13)
    axes[0].set_ylabel('Test Accuracy (%)')
    for bar, acc in zip(bars, accs):
        axes[0].text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.2,
            f'{acc:.2f}%', ha='center',
            fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    diff = accs[1] - accs[0]
    axes[0].set_title(
        f'Accuracy Comparison\n'
        f'CBAM improves by {diff:+.2f}%',
        fontweight='bold', fontsize=12)

    # Val accuracy curves
    for i, (r, h) in enumerate(
            zip(results, histories)):
        axes[1].plot(
            h['val_acc'],
            label=r['config'],
            color=colors[i],
            linewidth=2,
            marker='o',
            markersize=4)

    axes[1].set_title('Validation Accuracy Curve',
                      fontweight='bold', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        'AgriConvNet Ablation Study\n'
        'Effect of CBAM Attention Module',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(
        os.path.join(LOG_PATH, 'ablation_study.png'),
        dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✅ Ablation chart saved!")

# ============================================================
# SAVE RESULTS
# ============================================================
def save_results(results):
    path = os.path.join(LOG_PATH,
                        'ablation_results.txt')
    diff = results[1]['acc'] - results[0]['acc']

    with open(path, 'w') as f:
        f.write("ABLATION STUDY RESULTS\n")
        f.write("AgriConvNet — Effect of CBAM\n")
        f.write("Blockchain in Agriculture - Q1\n")
        f.write("="*50+"\n\n")
        f.write(f"{'Config':<25} {'Acc':>10} "
                f"{'F1':>8} {'Time':>8}\n")
        f.write("-"*50+"\n")
        for r in results:
            marker = " ← PROPOSED" \
                if 'CBAM' in r['config'] else ""
            f.write(f"{r['config']:<25} "
                    f"{r['acc']:>9.2f}% "
                    f"{r['f1']:>8.4f} "
                    f"{r['time']:>6.1f}m"
                    f"{marker}\n")
        f.write("="*50+"\n")
        f.write(f"CBAM Improvement: {diff:+.2f}%\n")

    print(f"\n{'='*55}")
    print(f"  ABLATION STUDY RESULTS")
    print(f"{'='*55}")
    print(f"  {'Config':<25} {'Acc':>8} {'F1':>8}")
    print(f"  {'-'*45}")
    for r in results:
        marker = " ✅ OURS" \
            if 'CBAM' in r['config'] else ""
        print(f"  {r['config']:<25} "
              f"{r['acc']:>7.2f}% "
              f"{r['f1']:>8.4f}{marker}")
    print(f"  {'-'*45}")
    print(f"  CBAM Improvement: {diff:+.2f}%")
    print(f"{'='*55}")
    print(f"  ✅ Saved: {path}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    (train_loader, val_loader,
     test_loader, num_classes) = load_data()

    # Only 2 configs — fast!
    CONFIGS = [
        {
            'name'    : 'ConvNeXt (No CBAM)',
            'use_cbam': False,
        },
        {
            'name'    : 'AgriConvNet (Full CBAM)',
            'use_cbam': True,
        },
    ]

    results   = []
    histories = []

    for cfg in CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Config: {cfg['name']}")
        print(f"{'='*60}")

        model = AgriConvNetAblation(
            num_classes = num_classes,
            use_cbam    = cfg['use_cbam'],
            pretrained  = True).to(DEVICE)

        total = sum(p.numel()
                    for p in model.parameters())/1e6
        print(f"  Params: {total:.2f}M")

        model, best_val, elapsed, history = \
            train_config(model, train_loader,
                         val_loader, num_classes,
                         cfg['name'])

        acc, f1 = test_model(model, test_loader)

        print(f"\n  ✅ {cfg['name']}:")
        print(f"     Test Acc : {acc:.2f}%")
        print(f"     F1 Score : {f1:.4f}")
        print(f"     Time     : {elapsed:.1f} mins")

        results.append({
            'config': cfg['name'],
            'acc'   : acc,
            'f1'    : f1,
            'time'  : elapsed
        })
        histories.append(history)

        torch.save(
            model.state_dict(),
            os.path.join(
                SAVE_PATH,
                f"{cfg['name'].replace(' ','_')}.pth"))

        del model
        torch.cuda.empty_cache()

    # Plot and save
    plot_ablation(results, histories)
    save_results(results)

    diff = results[1]['acc'] - results[0]['acc']
    print(f"\n✅ ABLATION COMPLETE!")
    print(f"   CBAM adds {diff:+.2f}% improvement!")
    print(f"\n🚀 Next → Blockchain → Paper Writing!")
