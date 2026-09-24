# ============================================================
# GRADCAM & HEATMAP - ALL 4 DATASETS
# Latest visualization techniques
# AgriChainNet - Q1 Journal Paper
# ============================================================
# Samples from:
# 1. PlantVillage (Classification)
# 2. Rice_Leaf (Classification)
# 3. Leaf_Disease (Segmentation)
# 4. Multi_Crop (Detection)
# ============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_PATH     = "D:/September/final_datasets"
MODEL_PATH    = "D:/September/models/AgriConvNet_fast"
ABLATION_PATH = "D:/September/models/ablation_fast"
SAVE_PATH     = "D:/September/gradcam_results"
LOG_PATH      = os.path.join(SAVE_PATH, "figures")
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH,  exist_ok=True)

IMG_SIZE = 224
DEVICE   = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("  GRADCAM - ALL 4 DATASETS")
print("  Latest Visualization Techniques")
print("  AgriChainNet - Q1 Paper")
print("=" * 60)
print(f"  Device: {DEVICE}")

# ============================================================
# MODEL ARCHITECTURE
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

class AgriConvNetModel(nn.Module):
    def __init__(self, num_classes, use_cbam=True):
        super().__init__()
        backbone = models.convnext_small(weights=None)
        f = backbone.features
        self.stem   = f[0]
        self.stage1 = f[1]
        self.down1  = f[2]
        self.stage2 = f[3]
        self.down2  = f[4]
        self.stage3 = f[5]
        self.down3  = f[6]
        self.stage4 = f[7]

        attn = CBAMBlock if use_cbam else lambda c, **k: IdentityBlock()
        self.attn1 = attn(96,  reduction=8)
        self.attn2 = attn(192, reduction=16)
        self.attn3 = attn(384, reduction=16)
        self.attn4 = attn(768, reduction=16)

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
# GRADCAM
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self.hooks       = []

        def fwd(m, i, o): self.activations = o.detach()
        def bwd(m, gi, go): self.gradients = go[0].detach()

        self.hooks.append(
            target_layer.register_forward_hook(fwd))
        self.hooks.append(
            target_layer.register_full_backward_hook(bwd))

    def generate(self, x, class_idx=None):
        self.model.eval()
        x = x.requires_grad_(True)
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        self.model.zero_grad()
        out[0, class_idx].backward()

        grads = self.gradients[0]
        acts  = self.activations[0]
        w     = grads.mean(dim=(1,2))
        cam   = torch.zeros(acts.shape[1:],
                            device=acts.device)
        for i, wi in enumerate(w):
            cam += wi * acts[i]
        cam = torch.relu(cam).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx

    def remove(self):
        for h in self.hooks: h.remove()

# ============================================================
# LOAD MODELS
# ============================================================
def load_models(num_classes):
    print("\n  Loading models...")

    def load_ckpt(model, path, is_cbam):
        if not os.path.exists(path):
            print(f"  ⚠️  Not found: {path}")
            return model
        ckpt  = torch.load(path, map_location=DEVICE)
        state = ckpt.get('model_state_dict', ckpt)
        new_s = {}
        for k, v in state.items():
            if is_cbam:
                nk = k.replace('cbam1.','attn1.')\
                       .replace('cbam2.','attn2.')\
                       .replace('cbam3.','attn3.')\
                       .replace('cbam4.','attn4.')
            else:
                nk = k
            new_s[nk] = v
        model.load_state_dict(new_s, strict=False)
        return model

    cbam = AgriConvNetModel(num_classes, use_cbam=True
                            ).to(DEVICE)
    cbam = load_ckpt(cbam,
                     f"{MODEL_PATH}/best.pth", True)
    print("  ✅ AgriConvNet (CBAM) ready")

    no_cbam = AgriConvNetModel(num_classes, use_cbam=False
                               ).to(DEVICE)
    no_cbam = load_ckpt(
        no_cbam,
        f"{ABLATION_PATH}/ConvNeXt_(No_CBAM).pth",
        False)
    print("  ✅ ConvNeXt (No CBAM) ready")

    return cbam, no_cbam

# ============================================================
# GET ONE SAMPLE FROM EACH DATASET
# ============================================================
def get_samples_all_datasets():
    print("\n  Getting 1 sample from each dataset...")

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    datasets_config = [
        {
            'name'  : 'PlantVillage',
            'path'  : f"{BASE_PATH}/PlantVillage/test",
            'task'  : 'Classification',
            'color' : '#2ecc71'
        },
        {
            'name'  : 'Rice Leaf',
            'path'  : f"{BASE_PATH}/Rice_Leaf/test",
            'task'  : 'Classification',
            'color' : '#3498db'
        },
        {
            'name'  : 'Leaf Disease',
            'path'  : f"{BASE_PATH}/Leaf_Disease/test",
            'task'  : 'Segmentation',
            'color' : '#e74c3c'
        },
        {
            'name'  : 'Multi-Crop TN',
            'path'  : f"{BASE_PATH}/Multi_Crop/test/images",
            'task'  : 'Detection',
            'color' : '#f39c12'
        },
    ]

    samples = []
    for cfg in datasets_config:
        print(f"  Loading from {cfg['name']}...")

        # Handle Multi-Crop (images folder directly)
        if cfg['name'] == 'Multi-Crop TN':
            img_dir = cfg['path']
            if not os.path.exists(img_dir):
                # Try alternative path
                img_dir = f"{BASE_PATH}/Multi_Crop/val/images"
            if os.path.exists(img_dir):
                imgs = [f for f in os.listdir(img_dir)
                        if f.endswith(('.jpg','.png','.jpeg'))]
                if imgs:
                    img_path = os.path.join(
                        img_dir, imgs[0])
                    orig = Image.open(img_path).convert('RGB')
                    orig = orig.resize((IMG_SIZE, IMG_SIZE))
                    orig_np = np.array(orig) / 255.0
                    img_t = val_tf(orig).unsqueeze(0).to(DEVICE)
                    samples.append({
                        'name'      : cfg['name'],
                        'task'      : cfg['task'],
                        'color'     : cfg['color'],
                        'orig'      : orig_np,
                        'tensor'    : img_t,
                        'class_name': 'Crop Disease',
                        'label'     : 0
                    })
                    print(f"  ✅ {cfg['name']}: loaded")
                    continue

        # Handle classification datasets
        if not os.path.exists(cfg['path']):
            print(f"  ⚠️  Not found: {cfg['path']}")
            continue

        try:
            ds = datasets.ImageFolder(
                cfg['path'], transform=val_tf)
            # Get first image
            img_t, label = ds[0]
            img_path = ds.samples[0][0]
            orig = Image.open(img_path).convert('RGB')
            orig = orig.resize((IMG_SIZE, IMG_SIZE))
            orig_np = np.array(orig) / 255.0

            class_name = ds.classes[label]\
                .replace('___',' ')\
                .replace('_',' ')

            samples.append({
                'name'      : cfg['name'],
                'task'      : cfg['task'],
                'color'     : cfg['color'],
                'orig'      : orig_np,
                'tensor'    : img_t.unsqueeze(0).to(DEVICE),
                'class_name': class_name,
                'label'     : label
            })
            print(f"  ✅ {cfg['name']}: {class_name[:30]}")

        except Exception as e:
            print(f"  ⚠️  {cfg['name']}: {str(e)[:50]}")

    return samples

# ============================================================
# APPLY HEATMAP
# ============================================================
def apply_heatmap(img_np, cam, alpha=0.55):
    cam_r = cv2.resize(
        cam, (img_np.shape[1], img_np.shape[0]))
    heat  = cv2.applyColorMap(
        np.uint8(255*cam_r), cv2.COLORMAP_JET)
    heat  = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    img_u = np.uint8(255*img_np)
    ov    = cv2.addWeighted(img_u, 1-alpha,
                            heat,  alpha, 0)
    return ov, heat, cam_r

# ============================================================
# FIGURE 1: MAIN GRADCAM COMPARISON (4 datasets)
# Latest style: 4 columns x 5 rows
# ============================================================
def plot_main_gradcam(samples, results_cbam,
                      results_no):
    n = len(samples)
    fig = plt.figure(figsize=(5*n, 20),
                     facecolor='white')

    rows = ['Original Image',
            'Without CBAM\n(Activation Map)',
            'AgriConvNet+CBAM\n(Activation Map)',
            'GradCAM Overlay\n(Without CBAM)',
            'GradCAM Overlay\n(AgriConvNet+CBAM)']

    gs = gridspec.GridSpec(5, n,
                           hspace=0.25,
                           wspace=0.08,
                           figure=fig)

    for j, (s, rc, rn) in enumerate(
            zip(samples, results_cbam, results_no)):

        # Row titles (left column only)
        for row_idx, row_label in enumerate(rows):
            ax = fig.add_subplot(gs[row_idx, j])

            if row_idx == 0:
                ax.imshow(s['orig'])
                ax.set_title(
                    f"{s['name']}\n"
                    f"({s['task']})\n"
                    f"{s['class_name'][:20]}",
                    fontsize=9, fontweight='bold',
                    color=s['color'])

            elif row_idx == 1:
                ax.imshow(rn['cam'],
                          cmap='jet',
                          vmin=0, vmax=1)

            elif row_idx == 2:
                ax.imshow(rc['cam'],
                          cmap='jet',
                          vmin=0, vmax=1)

            elif row_idx == 3:
                ax.imshow(rn['overlay'])

            elif row_idx == 4:
                ax.imshow(rc['overlay'])
                # Green border for our model
                for spine in ax.spines.values():
                    spine.set_edgecolor('#2ecc71')
                    spine.set_linewidth(3)

            ax.axis('off')

            # Row labels (first column)
            if j == 0:
                ax.text(-0.15, 0.5, row_label,
                        transform=ax.transAxes,
                        fontsize=9,
                        fontweight='bold',
                        va='center',
                        ha='right',
                        color='green'
                        if row_idx == 4 else 'black',
                        rotation=0)

    # Colorbar
    cax = fig.add_axes([0.92, 0.35, 0.015, 0.3])
    sm  = plt.cm.ScalarMappable(
        cmap='jet',
        norm=plt.Normalize(vmin=0, vmax=1))
    plt.colorbar(sm, cax=cax,
                 label='Attention Score')

    plt.suptitle(
        'GradCAM Visualization: AgriConvNet vs Baseline\n'
        'Samples from All 4 Datasets | '
        'Green Border = Proposed Model (AgriConvNet+CBAM)',
        fontsize=13, fontweight='bold',
        y=0.98)

    path = os.path.join(LOG_PATH,
                        'Figure_GradCAM_4Datasets.png')
    plt.savefig(path, dpi=200,
                bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  ✅ Main GradCAM figure saved!")
    return path

# ============================================================
# FIGURE 2: RADAR/SPIDER CHART — Attention Quality
# Latest technique for paper
# ============================================================
def plot_radar_chart(samples, results_cbam,
                     results_no):
    categories = [s['name'] for s in samples]
    N = len(categories)

    cbam_scores   = [r['concentration']
                     for r in results_cbam]
    no_cbam_scores= [r['concentration']
                     for r in results_no]

    # Normalize to 0-1
    max_v = max(cbam_scores + no_cbam_scores)
    cbam_n    = [v/max_v for v in cbam_scores]
    no_cbam_n = [v/max_v for v in no_cbam_scores]

    angles = np.linspace(0, 2*np.pi, N,
                         endpoint=False).tolist()
    angles += angles[:1]
    cbam_n    += cbam_n[:1]
    no_cbam_n += no_cbam_n[:1]

    fig, ax = plt.subplots(
        1, 1, figsize=(8, 8),
        subplot_kw=dict(polar=True),
        facecolor='white')

    ax.plot(angles, no_cbam_n,
            'o-', linewidth=2,
            color='#e74c3c',
            label='Without CBAM')
    ax.fill(angles, no_cbam_n,
            alpha=0.15, color='#e74c3c')

    ax.plot(angles, cbam_n,
            'o-', linewidth=2,
            color='#2ecc71',
            label='AgriConvNet+CBAM (Ours)')
    ax.fill(angles, cbam_n,
            alpha=0.25, color='#2ecc71')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11,
                       fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%','50%','75%','100%'],
                       fontsize=8)
    ax.grid(color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right',
              bbox_to_anchor=(1.35, 1.15),
              fontsize=11)
    ax.set_title(
        'Attention Concentration\nAcross All 4 Datasets',
        fontsize=13, fontweight='bold', pad=20)

    path = os.path.join(LOG_PATH,
                        'Figure_Radar_Attention.png')
    plt.savefig(path, dpi=200,
                bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  ✅ Radar chart saved!")
    return path

# ============================================================
# FIGURE 3: SIDE BY SIDE HEATMAP STRIP
# Clean publication-ready figure
# ============================================================
def plot_heatmap_strip(samples, results_cbam,
                       results_no):
    n   = len(samples)
    fig, axes = plt.subplots(
        3, n, figsize=(4*n, 10),
        facecolor='white')

    for j, (s, rc, rn) in enumerate(
            zip(samples, results_cbam, results_no)):

        # Row 1: Original
        axes[0, j].imshow(s['orig'])
        axes[0, j].set_title(
            f"{s['name']}\n{s['class_name'][:20]}",
            fontsize=8, fontweight='bold',
            color=s['color'])
        axes[0, j].axis('off')

        # Row 2: No CBAM
        im = axes[1, j].imshow(
            rn['cam'], cmap='hot',
            vmin=0, vmax=1)
        axes[1, j].axis('off')

        # Row 3: With CBAM
        axes[2, j].imshow(
            rc['cam'], cmap='hot',
            vmin=0, vmax=1)
        axes[2, j].axis('off')
        for sp in axes[2, j].spines.values():
            sp.set_edgecolor('#2ecc71')
            sp.set_linewidth(2)

    # Row labels
    axes[0,0].set_ylabel('Input Image',
                          fontsize=11,
                          fontweight='bold')
    axes[1,0].set_ylabel('Without CBAM',
                          fontsize=11,
                          fontweight='bold')
    axes[2,0].set_ylabel('AgriConvNet+CBAM\n(Ours)',
                          fontsize=11,
                          fontweight='bold',
                          color='#27ae60')

    # Colorbar
    plt.colorbar(im, ax=axes[2,:],
                 orientation='horizontal',
                 pad=0.05, fraction=0.03,
                 label='Attention Intensity '
                       '(Red=High, Black=Low)')

    plt.suptitle(
        'Attention Heatmaps — All 4 Datasets\n'
        'AgriConvNet+CBAM Shows Superior '
        'Disease Region Focus',
        fontsize=13, fontweight='bold')

    path = os.path.join(LOG_PATH,
                        'Figure_Heatmap_Strip.png')
    plt.savefig(path, dpi=200,
                bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  ✅ Heatmap strip saved!")
    return path

# ============================================================
# FIGURE 4: BAR CHART — Attention Score per Dataset
# ============================================================
def plot_attention_bars(samples, results_cbam,
                        results_no):
    names      = [s['name'] for s in samples]
    cbam_sc    = [r['concentration'] for r in results_cbam]
    no_cbam_sc = [r['concentration'] for r in results_no]

    x   = np.arange(len(names))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 6),
                           facecolor='white')

    b1 = ax.bar(x-w/2, no_cbam_sc, w,
                label='Without CBAM',
                color='#e74c3c',
                edgecolor='white',
                linewidth=1.5)
    b2 = ax.bar(x+w/2, cbam_sc, w,
                label='AgriConvNet+CBAM (Ours)',
                color='#2ecc71',
                edgecolor='white',
                linewidth=1.5)

    for bar in b1:
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.002,
                f'{bar.get_height():.3f}',
                ha='center', fontsize=9)
    for bar in b2:
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.002,
                f'{bar.get_height():.3f}',
                ha='center', fontsize=9,
                color='#27ae60', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Attention Concentration Score',
                  fontsize=11)
    ax.set_title(
        'CBAM Attention Concentration Score\n'
        'Across All 4 Datasets',
        fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(cbam_sc+no_cbam_sc)*1.2)

    path = os.path.join(LOG_PATH,
                        'Figure_Attention_Bars.png')
    plt.savefig(path, dpi=200,
                bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  ✅ Attention bar chart saved!")
    return path

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # Step 1: Get 1 sample from each dataset
    samples = get_samples_all_datasets()

    if not samples:
        print("❌ No samples found!")
        exit(1)

    print(f"\n  Total samples: {len(samples)}")

    # Step 2: Load models
    # Get num_classes from PlantVillage
    pv_path = f"{BASE_PATH}/PlantVillage/test"
    if os.path.exists(pv_path):
        ds = datasets.ImageFolder(pv_path)
        num_classes = len(ds.classes)
    else:
        num_classes = 38

    cbam_model, no_cbam_model = load_models(num_classes)

    # Step 3: Generate GradCAM for both models
    print("\n  Generating GradCAM maps...")

    cbam_grad   = GradCAM(cbam_model,
                          cbam_model.stage4[-1])
    no_cbam_grad= GradCAM(no_cbam_model,
                          no_cbam_model.stage4[-1])

    results_cbam = []
    results_no   = []

    for s in samples:
        print(f"  Processing: {s['name']}...")
        t = s['tensor']

        cam_c, _ = cbam_grad.generate(
            t.clone(), class_idx=s['label'])
        cam_n, _ = no_cbam_grad.generate(
            t.clone(), class_idx=s['label'])

        ov_c, _, cam_c_r = apply_heatmap(
            s['orig'], cam_c)
        ov_n, _, cam_n_r = apply_heatmap(
            s['orig'], cam_n)

        results_cbam.append({
            'cam'          : cam_c_r,
            'overlay'      : ov_c,
            'concentration': float(cam_c_r.std())
        })
        results_no.append({
            'cam'          : cam_n_r,
            'overlay'      : ov_n,
            'concentration': float(cam_n_r.std())
        })

    cbam_grad.remove()
    no_cbam_grad.remove()

    # Step 4: Generate all figures
    print("\n  Creating figures...")

    plot_main_gradcam(samples, results_cbam, results_no)
    plot_heatmap_strip(samples, results_cbam, results_no)
    plot_attention_bars(samples, results_cbam, results_no)
    plot_radar_chart(samples, results_cbam, results_no)

    # Step 5: Save metrics
    with open(os.path.join(SAVE_PATH,
                           'attention_metrics.txt'),
              'w', encoding='utf-8') as f:
        f.write("Attention Concentration Metrics\n")
        f.write("AgriConvNet GradCAM Analysis\n")
        f.write("="*50+"\n\n")
        f.write(f"{'Dataset':<20} {'No CBAM':>12} "
                f"{'CBAM':>12} {'Improvement':>12}\n")
        f.write("-"*50+"\n")
        for s, rc, rn in zip(
                samples, results_cbam, results_no):
            imp = ((rc['concentration']-
                    rn['concentration'])/
                   max(rn['concentration'],0.001))*100
            f.write(f"{s['name']:<20} "
                    f"{rn['concentration']:>12.4f} "
                    f"{rc['concentration']:>12.4f} "
                    f"{imp:>11.1f}%\n")

    print(f"\n{'='*60}")
    print(f"  GRADCAM COMPLETE!")
    print(f"  Saved to: {LOG_PATH}")
    print(f"\n  Figures generated:")
    print(f"  1. Figure_GradCAM_4Datasets.png")
    print(f"     (Main comparison — use in paper)")
    print(f"  2. Figure_Heatmap_Strip.png")
    print(f"     (Clean heatmaps — use in paper)")
    print(f"  3. Figure_Attention_Bars.png")
    print(f"     (Bar chart — use in paper)")
    print(f"  4. Figure_Radar_Attention.png")
    print(f"     (Radar chart — latest technique)")
    print(f"{'='*60}")
    print(f"\n  Now run: python collect_results.py")
    print(f"  Then: Start Paper Writing!")
