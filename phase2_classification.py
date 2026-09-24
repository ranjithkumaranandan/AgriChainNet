# ============================================================
# PHASE 2 - MODEL C: ResNet50 CLASSIFICATION
# Datasets: PlantVillage + Rice_Leaf
# Blockchain in Agriculture - Q1 Journal Paper
# ============================================================

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
BASE_PATH    = "D:/September/final_datasets"
SAVE_PATH    = "D:/September/models/classification"
LOG_PATH     = "D:/September/models/classification/logs"
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH,  exist_ok=True)

# Training hyperparameters
BATCH_SIZE   = 32
NUM_EPOCHS   = 30
LR           = 0.001
WEIGHT_DECAY = 1e-4
IMG_SIZE     = 224
NUM_WORKERS  = 4

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# DATA TRANSFORMS
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# LOAD DATASETS
# ============================================================
def load_datasets():
    print("\n📂 Loading datasets...")

    datasets_config = {
        "PlantVillage": os.path.join(BASE_PATH, "PlantVillage"),
        "Rice_Leaf"   : os.path.join(BASE_PATH, "Rice_Leaf"),
    }

    train_datasets = []
    val_datasets   = []
    test_datasets  = []
    all_classes    = set()

    for name, path in datasets_config.items():
        train_path = os.path.join(path, "train")
        val_path   = os.path.join(path, "val")
        test_path  = os.path.join(path, "test")

        if not os.path.exists(train_path):
            print(f"  ⚠️  {name} train path not found, skipping.")
            continue

        train_ds = datasets.ImageFolder(train_path, transform=train_transform)
        val_ds   = datasets.ImageFolder(val_path,   transform=val_transform)
        test_ds  = datasets.ImageFolder(test_path,  transform=val_transform)

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        test_datasets.append(test_ds)
        all_classes.update(train_ds.classes)

        print(f"  ✅ {name}: "
              f"Train={len(train_ds):,} "
              f"Val={len(val_ds):,} "
              f"Test={len(test_ds):,} "
              f"Classes={len(train_ds.classes)}")

    # Combine datasets
    combined_train = ConcatDataset(train_datasets)
    combined_val   = ConcatDataset(val_datasets)
    combined_test  = ConcatDataset(test_datasets)

    # Get total classes from first dataset (PlantVillage dominates)
    num_classes = len(train_datasets[0].classes)
    class_names = train_datasets[0].classes

    print(f"\n  📊 Combined: "
          f"Train={len(combined_train):,} "
          f"Val={len(combined_val):,} "
          f"Test={len(combined_test):,}")
    print(f"  📊 Total classes (PlantVillage): {num_classes}")

    return combined_train, combined_val, combined_test, num_classes, class_names


# ============================================================
# BUILD MODEL
# ============================================================
def build_model(num_classes):
    print(f"\n🏗️  Building ResNet50 model...")
    print(f"    Classes: {num_classes}")

    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze early layers (feature extraction)
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"    Total params    : {total_params:,}")
    print(f"    Trainable params: {trainable_params:,}")

    return model.to(DEVICE)


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(model, train_loader, val_loader, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler    = GradScaler()  # Mixed precision

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss'  : [], 'val_acc'  : []
    }

    best_val_acc  = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience      = 7
    patience_counter = 0

    print(f"\n🚀 Starting Training for {num_epochs} epochs...")
    print(f"   Batch size : {BATCH_SIZE}")
    print(f"   LR         : {LR}")
    print(f"   Device     : {DEVICE}")
    print("=" * 55)

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # ── Training Phase ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total   = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
                    leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item() * inputs.size(0)
            _, predicted   = outputs.max(1)
            train_total   += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc' : f"{100.*train_correct/train_total:.1f}%"
            })

        # ── Validation Phase ──
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader,
                                       desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
                                       leave=False):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                with autocast():
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)

                val_loss    += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total   += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Compute epoch metrics
        epoch_train_loss = train_loss / train_total
        epoch_train_acc  = 100. * train_correct / train_total
        epoch_val_loss   = val_loss / val_total
        epoch_val_acc    = 100. * val_correct / val_total
        epoch_time       = time.time() - epoch_start

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{num_epochs}] "
              f"Time:{epoch_time:.0f}s | "
              f"Train Loss:{epoch_train_loss:.4f} Acc:{epoch_train_acc:.2f}% | "
              f"Val Loss:{epoch_val_loss:.4f} Acc:{epoch_val_acc:.2f}%", end="")

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc   = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch'     : epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc'   : best_val_acc,
            }, os.path.join(SAVE_PATH, 'resnet50_best.pth'))
            print(f" ✅ Best model saved! ({best_val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f" (patience: {patience_counter}/{patience})")

        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break

    total_time = time.time() - start_time
    print(f"\n⏱️  Total training time: {total_time/60:.1f} minutes")
    print(f"🏆 Best Val Accuracy: {best_val_acc:.2f}%")

    model.load_state_dict(best_model_wts)
    return model, history


# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(model, test_loader, class_names):
    print("\n📊 Evaluating on Test Set...")
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(DEVICE)
            with autocast():
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Metrics
    acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n  Test Accuracy: {acc:.2f}%")

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names[:len(set(all_labels))],
        digits=4)
    print("\n  Classification Report:")
    print(report)

    # Save report
    with open(os.path.join(LOG_PATH, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {acc:.2f}%\n\n")
        f.write(report)

    return all_preds, all_labels, acc


# ============================================================
# PLOT RESULTS
# ============================================================
def plot_results(history, all_preds, all_labels, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss', color='#e74c3c')
    axes[0].plot(history['val_loss'],   label='Val Loss',   color='#3498db')
    axes[0].set_title('Loss Curve', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(history['train_acc'], label='Train Acc', color='#e74c3c')
    axes[1].plot(history['val_acc'],   label='Val Acc',   color='#3498db')
    axes[1].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('ResNet50 Training — Blockchain in Agriculture',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_PATH, 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Training curves saved!")

    # Confusion matrix (top 15 classes only for readability)
    unique_labels = sorted(set(all_labels))[:15]
    filtered_preds  = [p for p, l in zip(all_preds, all_labels)
                       if l in unique_labels]
    filtered_labels = [l for l in all_labels if l in unique_labels]

    if filtered_labels:
        cm = confusion_matrix(filtered_labels, filtered_preds,
                              labels=unique_labels)
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[class_names[i] for i in unique_labels],
                    yticklabels=[class_names[i] for i in unique_labels])
        plt.title('Confusion Matrix (Top 15 Classes)',
                  fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_PATH, 'confusion_matrix.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Confusion matrix saved!")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  PHASE 2 - ResNet50 CLASSIFICATION")
    print("  Blockchain in Agriculture - Q1 Paper")
    print("=" * 55)

    # Load data
    train_ds, val_ds, test_ds, num_classes, class_names = load_datasets()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)

    # Build model
    model = build_model(num_classes)

    # Train
    model, history = train_model(model, train_loader, val_loader, NUM_EPOCHS)

    # Evaluate
    all_preds, all_labels, test_acc = evaluate_model(
        model, test_loader, class_names)

    # Plot
    plot_results(history, all_preds, all_labels, class_names)

    # Save final model
    torch.save(model.state_dict(),
               os.path.join(SAVE_PATH, 'resnet50_final.pth'))

    print(f"\n{'='*55}")
    print(f"  CLASSIFICATION COMPLETE!")
    print(f"  Test Accuracy : {test_acc:.2f}%")
    print(f"  Model saved   : {SAVE_PATH}")
    print(f"  Logs saved    : {LOG_PATH}")
    print(f"{'='*55}")
    print(f"\n✅ Phase 2C Done! Ready for Phase 2A: YOLOv8 Detection")
