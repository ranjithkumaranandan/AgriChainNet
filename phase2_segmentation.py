# ============================================================
# PHASE 2B - U-Net SEGMENTATION
# Dataset: Plant Leaf Disease Recognition Dataset
# Blockchain in Agriculture - Q1 Journal Paper
# ============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import jaccard_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_PATH    = "D:/September"
DATASET_PATH = os.path.join(BASE_PATH,
    "final_datasets/Leaf_Disease")
SAVE_PATH    = os.path.join(BASE_PATH, "models/segmentation")
LOG_PATH     = os.path.join(BASE_PATH, "models/segmentation/logs")
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH,  exist_ok=True)

IMG_SIZE     = 512
BATCH_SIZE   = 8
NUM_EPOCHS   = 30
LR           = 0.0001
NUM_WORKERS  = 4
PATIENCE     = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 55)
print("  PHASE 2B - U-Net SEGMENTATION TRAINING")
print("  Blockchain in Agriculture - Q1 Paper")
print("=" * 55)
print(f"\n  Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")

# ============================================================
# STEP 1: EXPLORE DATASET STRUCTURE
# ============================================================
def explore_dataset(base_path):
    print(f"\n📂 Exploring dataset structure...")
    classes = []

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(base_path, split)
        if not os.path.exists(split_path):
            continue
        class_dirs = [d for d in os.listdir(split_path)
                      if os.path.isdir(os.path.join(split_path, d))]
        if split == 'train':
            classes = class_dirs
        total = sum(len(os.listdir(os.path.join(split_path, c)))
                    for c in class_dirs
                    if os.path.isdir(os.path.join(split_path, c)))
        print(f"  {split:<6}: {len(class_dirs)} classes, {total:,} images")

    print(f"  Classes: {classes}")
    return classes

# ============================================================
# STEP 2: DATASET CLASS
# Since Leaf_Disease has no pixel masks, we generate them
# using color-based segmentation (background removed images)
# ============================================================
class LeafDiseaseDataset(Dataset):
    def __init__(self, root_path, split='train', transform=None,
                 img_size=512):
        self.root      = os.path.join(root_path, split)
        self.transform = transform
        self.img_size  = img_size
        self.samples   = []
        self.class_map = {}
        self.img_exts  = ['.jpg', '.jpeg', '.png', '.bmp']

        # Collect all images and assign class labels
        class_dirs = sorted([d for d in os.listdir(self.root)
                             if os.path.isdir(os.path.join(self.root, d))])

        for idx, class_name in enumerate(class_dirs):
            self.class_map[class_name] = idx
            class_path = os.path.join(self.root, class_name)
            for fname in os.listdir(class_path):
                if os.path.splitext(fname)[1].lower() in self.img_exts:
                    self.samples.append((
                        os.path.join(class_path, fname),
                        idx,
                        class_name
                    ))

        self.num_classes = len(class_dirs)
        print(f"  {split:<6}: {len(self.samples):,} images, "
              f"{self.num_classes} classes")

    def generate_mask(self, img_np):
        """
        Generate segmentation mask from background-removed image.
        Since background is removed (black/white bg), we threshold
        to create binary leaf vs background mask.
        """
        # Convert to HSV for better leaf segmentation
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        # Green/yellow/brown leaf range
        lower1 = np.array([20,  20,  20])
        upper1 = np.array([180, 255, 255])
        mask1  = cv2.inRange(hsv, lower1, upper1)

        # Also threshold non-black pixels (background removed images)
        gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, mask2 = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # Combine masks
        mask = cv2.bitwise_and(mask1, mask2)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        # Normalize to 0/1
        mask = (mask > 0).astype(np.float32)
        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx, class_name = self.samples[idx]

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Generate mask
        mask = self.generate_mask(img)
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=img,
                                       mask=mask.astype(np.uint8))
            img  = augmented['image']
            mask = augmented['mask'].float()
        else:
            img  = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float()

        return img, mask.unsqueeze(0), class_idx

# ============================================================
# STEP 3: TRANSFORMS
# ============================================================
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                       rotate_limit=30, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2,
                  saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ============================================================
# STEP 4: LOSS FUNCTIONS
# ============================================================
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss  = self.bce(pred, target)

        pred_sig  = torch.sigmoid(pred)
        smooth    = 1e-6
        inter     = (pred_sig * target).sum(dim=(2, 3))
        dice_loss = 1 - (2. * inter + smooth) / (
            pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)

        return bce_loss + dice_loss.mean()

# ============================================================
# STEP 5: METRICS
# ============================================================
def compute_metrics(pred_logits, target, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    pred_np   = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()

    smooth = 1e-6
    inter  = (pred_np * target_np).sum()
    dice   = (2. * inter + smooth) / (pred_np.sum() + target_np.sum() + smooth)
    iou    = inter / (pred_np.sum() + target_np.sum() - inter + smooth)

    correct = (pred_np == target_np).sum()
    acc     = correct / len(pred_np)

    return {'dice': dice, 'iou': iou, 'acc': acc}

# ============================================================
# STEP 6: TRAINING LOOP
# ============================================================
def train_model(model, train_loader, val_loader):
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS,
                                  eta_min=1e-7)
    scaler    = GradScaler()

    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou' : [], 'val_iou' : [],
    }

    best_dice      = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_ctr   = 0

    print(f"\n🚀 Starting U-Net Training for {NUM_EPOCHS} epochs...")
    print(f"   Batch size : {BATCH_SIZE}")
    print(f"   LR         : {LR}")
    print(f"   Device     : {DEVICE}")
    print("=" * 55)

    start = time.time()

    for epoch in range(NUM_EPOCHS):
        ep_start = time.time()

        # ── Train ──
        model.train()
        tr_loss = tr_dice = tr_iou = 0.0
        n_tr    = 0

        for imgs, masks, _ in tqdm(train_loader,
                                   desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]",
                                   leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                preds = model(imgs)
                loss  = criterion(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            m = compute_metrics(preds.detach(), masks)
            bs = imgs.size(0)
            tr_loss += loss.item() * bs
            tr_dice += m['dice']   * bs
            tr_iou  += m['iou']    * bs
            n_tr    += bs

        # ── Validate ──
        model.eval()
        va_loss = va_dice = va_iou = 0.0
        n_va    = 0

        with torch.no_grad():
            for imgs, masks, _ in tqdm(val_loader,
                                       desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]",
                                       leave=False):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                with autocast():
                    preds = model(imgs)
                    loss  = criterion(preds, masks)

                m = compute_metrics(preds, masks)
                bs = imgs.size(0)
                va_loss += loss.item() * bs
                va_dice += m['dice']   * bs
                va_iou  += m['iou']    * bs
                n_va    += bs

        # Averages
        tr_loss /= n_tr; tr_dice /= n_tr; tr_iou /= n_tr
        va_loss /= n_va; va_dice /= n_va; va_iou /= n_va
        ep_time  = time.time() - ep_start

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_dice'].append(tr_dice)
        history['val_dice'].append(va_dice)
        history['train_iou'].append(tr_iou)
        history['val_iou'].append(va_iou)

        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] {ep_time:.0f}s | "
              f"Loss: {tr_loss:.4f}/{va_loss:.4f} | "
              f"Dice: {tr_dice:.4f}/{va_dice:.4f} | "
              f"IoU: {tr_iou:.4f}/{va_iou:.4f}", end="")

        if va_dice > best_dice:
            best_dice      = va_dice
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_dice': best_dice},
                       os.path.join(SAVE_PATH, 'unet_best.pth'))
            print(f" ✅ Best! Dice={best_dice:.4f}")
            patience_ctr = 0
        else:
            patience_ctr += 1
            print(f" (patience {patience_ctr}/{PATIENCE})")

        if patience_ctr >= PATIENCE:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break

    total_time = time.time() - start
    print(f"\n⏱️  Training time: {total_time/60:.1f} mins")
    print(f"🏆 Best Val Dice: {best_dice:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history

# ============================================================
# STEP 7: EVALUATE ON TEST SET
# ============================================================
def evaluate(model, test_loader):
    print("\n📊 Evaluating on Test Set...")
    model.eval()

    all_dice = all_iou = all_acc = 0.0
    n = 0

    with torch.no_grad():
        for imgs, masks, _ in tqdm(test_loader, desc="Testing"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            with autocast():
                preds = model(imgs)
            m = compute_metrics(preds, masks)
            bs = imgs.size(0)
            all_dice += m['dice'] * bs
            all_iou  += m['iou']  * bs
            all_acc  += m['acc']  * bs
            n        += bs

    dice = all_dice / n
    iou  = all_iou  / n
    acc  = all_acc  / n

    print(f"\n  Test Results:")
    print(f"  Dice Score  : {dice:.4f}")
    print(f"  IoU (Jaccard): {iou:.4f}")
    print(f"  Pixel Acc   : {acc:.4f}")

    with open(os.path.join(LOG_PATH, 'segmentation_results.txt'), 'w') as f:
        f.write("U-Net Segmentation Results\n")
        f.write("Blockchain in Agriculture - Q1 Paper\n")
        f.write("="*40 + "\n")
        f.write(f"Dice Score   : {dice:.4f}\n")
        f.write(f"IoU Score    : {iou:.4f}\n")
        f.write(f"Pixel Acc    : {acc:.4f}\n")

    return dice, iou, acc

# ============================================================
# STEP 8: VISUALIZE PREDICTIONS
# ============================================================
def visualize_predictions(model, test_loader, n_samples=6):
    print("\n🖼️  Saving sample predictions...")
    model.eval()

    imgs_list  = []
    masks_list = []
    preds_list = []

    with torch.no_grad():
        for imgs, masks, _ in test_loader:
            imgs_gpu = imgs.to(DEVICE)
            with autocast():
                preds = torch.sigmoid(model(imgs_gpu))
            imgs_list.extend(imgs.cpu())
            masks_list.extend(masks.cpu())
            preds_list.extend(preds.cpu())
            if len(imgs_list) >= n_samples:
                break

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 4))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for i in range(min(n_samples, len(imgs_list))):
        # Denormalize image
        img = imgs_list[i].numpy().transpose(1, 2, 0)
        img = (img * std + mean).clip(0, 1)

        mask = masks_list[i].squeeze().numpy()
        pred = (preds_list[i].squeeze().numpy() > 0.5).astype(float)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image', fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth', fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction', fontsize=10)
        axes[i, 2].axis('off')

    plt.suptitle('U-Net Segmentation — Blockchain in Agriculture',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_PATH, 'segmentation_predictions.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Predictions saved!")

# ============================================================
# STEP 9: PLOT TRAINING CURVES
# ============================================================
def plot_curves(history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(history['train_loss'], label='Train', color='#e74c3c')
    axes[0].plot(history['val_loss'],   label='Val',   color='#3498db')
    axes[0].set_title('Loss',      fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_dice'], label='Train', color='#e74c3c')
    axes[1].plot(history['val_dice'],   label='Val',   color='#3498db')
    axes[1].set_title('Dice Score', fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['train_iou'], label='Train', color='#e74c3c')
    axes[2].plot(history['val_iou'],   label='Val',   color='#3498db')
    axes[2].set_title('IoU Score',  fontweight='bold')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('U-Net Training — Blockchain in Agriculture',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_PATH, 'unet_training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Training curves saved!")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # Explore dataset
    classes = explore_dataset(DATASET_PATH)
    num_classes = len(classes)

    # Create datasets
    print(f"\n📦 Creating datasets...")
    train_ds = LeafDiseaseDataset(DATASET_PATH, 'train',
                                  train_transform, IMG_SIZE)
    val_ds   = LeafDiseaseDataset(DATASET_PATH, 'val',
                                  val_transform,   IMG_SIZE)
    test_ds  = LeafDiseaseDataset(DATASET_PATH, 'test',
                                  val_transform,   IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)

    # Build U-Net model with ResNet34 encoder
    print(f"\n🏗️  Building U-Net with ResNet34 encoder...")
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total params: {total_params:,}")

    # Train
    model, history = train_model(model, train_loader, val_loader)

    # Evaluate
    dice, iou, acc = evaluate(model, test_loader)

    # Visualize
    visualize_predictions(model, test_loader)

    # Plot curves
    plot_curves(history)

    # Save final model
    torch.save(model.state_dict(),
               os.path.join(SAVE_PATH, 'unet_final.pth'))

    print(f"\n{'='*55}")
    print(f"  SEGMENTATION COMPLETE!")
    print(f"  Dice Score  : {dice:.4f}")
    print(f"  IoU Score   : {iou:.4f}")
    print(f"  Pixel Acc   : {acc:.4f}")
    print(f"  Model saved : {SAVE_PATH}")
    print(f"  Logs saved  : {LOG_PATH}")
    print(f"{'='*55}")
    print(f"\n✅ Phase 2B Done! All 3 models complete!")
    print(f"   Next → Phase 3: Blockchain Integration")
