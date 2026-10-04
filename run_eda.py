import os
import glob
from collections import defaultdict
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "skin_disease_classification", "Split_smol")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*60)
print("EXPLORATORY DATA ANALYSIS (EDA) - SKIN DISEASE DATASET")
print(f"Dataset path: {DATASET_DIR}")
print("="*60)

splits = [d for d in ['train', 'val', 'test'] if os.path.isdir(os.path.join(DATASET_DIR, d))]
print(f"Detected splits: {splits}")

data_records = []
corrupted_files = []
image_sizes = []

for split in splits:
    split_dir = os.path.join(DATASET_DIR, split)
    classes = [c for c in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, c))]
    
    for cls in sorted(classes):
        cls_dir = os.path.join(split_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for f in files:
            img_path = os.path.join(cls_dir, f)
            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    w, h = img.size
                    mode = img.mode
                    image_sizes.append((w, h))
                data_records.append({
                    'split': split,
                    'class': cls,
                    'filename': f,
                    'path': img_path,
                    'width': w,
                    'height': h,
                    'mode': mode
                })
            except Exception as e:
                corrupted_files.append((img_path, str(e)))

df = pd.DataFrame(data_records)

print(f"\n[1] Overall Statistics:")
print(f"- Total valid images: {len(df)}")
print(f"- Corrupted images: {len(corrupted_files)}")
if corrupted_files:
    for c, err in corrupted_files:
        print(f"  Corrupt: {c} - {err}")
else:
    print("  [OK] All images verified healthy and uncorrupted!")

# Class Counts
pivot_counts = pd.pivot_table(
    df, 
    index='class', 
    columns='split', 
    values='filename', 
    aggfunc='count', 
    fill_value=0
)
pivot_counts['total'] = pivot_counts.sum(axis=1)
if 'train' in pivot_counts.columns and 'val' in pivot_counts.columns:
    pivot_counts['train_pct'] = (pivot_counts['train'] / pivot_counts['total'] * 100).round(1)
    pivot_counts['val_pct'] = (pivot_counts['val'] / pivot_counts['total'] * 100).round(1)

print("\n[2] Class Distribution Summary:")
print(pivot_counts.to_string())

# Image Size statistics
widths = [r['width'] for r in data_records]
heights = [r['height'] for r in data_records]
modes = set(r['mode'] for r in data_records)
print(f"\n[3] Image Properties:")
print(f"- Unique color modes: {modes}")
print(f"- Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}, median={np.median(widths)}")
print(f"- Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}, median={np.median(heights)}")

# 1. Plot Class Distribution Bar Chart
fig, ax = plt.subplots(figsize=(12, 6))
classes = sorted(df['class'].unique())
x = np.arange(len(classes))
width = 0.26

train_counts = [len(df[(df['class'] == c) & (df['split'] == 'train')]) for c in classes]
val_counts = [len(df[(df['class'] == c) & (df['split'] == 'val')]) for c in classes]
test_counts = [len(df[(df['class'] == c) & (df['split'] == 'test')]) for c in classes]

rects1 = ax.bar(x - width, train_counts, width, label='Train', color='#2b5c8f', edgecolor='black', alpha=0.85)
rects2 = ax.bar(x, val_counts, width, label='Validation', color='#e28743', edgecolor='black', alpha=0.85)
rects3 = ax.bar(x + width, test_counts, width, label='Test', color='#2ca02c', edgecolor='black', alpha=0.85)

ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
ax.set_title('Dataset Class Distribution (Train, Validation, Test)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=35, ha='right', fontsize=10, fontweight='medium')
ax.legend(frameon=True, fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h}', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
dist_chart_path = os.path.join(RESULTS_DIR, "eda_class_distribution.png")
plt.savefig(dist_chart_path, dpi=300)
plt.close()
print(f"\n[4] Class distribution plot saved to: {dist_chart_path}")

# 2. Plot Sample Images (3 images per class)
num_classes = len(classes)
samples_per_class = 3
fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(10, 2.5 * num_classes))

for i, cls in enumerate(classes):
    cls_df = df[df['class'] == cls]
    # Pick diverse sample images from train
    sample_rows = cls_df.sample(min(samples_per_class, len(cls_df)), random_state=42)
    for j, (_, row) in enumerate(sample_rows.iterrows()):
        ax = axes[i, j] if num_classes > 1 else axes[j]
        img = Image.open(row['path']).convert('RGB')
        ax.imshow(img)
        ax.axis('off')
        if j == 0:
            ax.set_title(f"{cls}\n({row['width']}x{row['height']})", fontsize=10, fontweight='bold', loc='left')
        else:
            ax.set_title(f"({row['width']}x{row['height']})", fontsize=9, color='#555555')

plt.suptitle("Sample Images from Each Skin Disease Class", fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
samples_plot_path = os.path.join(RESULTS_DIR, "eda_sample_images.png")
plt.savefig(samples_plot_path, dpi=200)
plt.close()
print(f"[5] Sample images plot saved to: {samples_plot_path}")

# Save eda summary to markdown
summary_md = f"""# Exploratory Data Analysis (EDA) Summary

## 1. Overview
- **Total Classes**: {len(classes)}
- **Total Images**: {len(df)}
- **Train Images**: {len(df[df['split'] == 'train'])} ({len(df[df['split'] == 'train'])/len(df)*100:.1f}%)
- **Validation Images**: {len(df[df['split'] == 'val'])} ({len(df[df['split'] == 'val'])/len(df)*100:.1f}%)
- **Test Images**: {len(df[df['split'] == 'test'])} ({len(df[df['split'] == 'test'])/len(df)*100:.1f}%)
- **Corrupted Images**: {len(corrupted_files)}
- **Color Mode**: {', '.join(modes)}
- **Average Image Resolution**: {np.mean(widths):.0f} x {np.mean(heights):.0f} px

## 2. Class Distribution Table
| No | Class Name | Train | Val | Test | Total |
|:---|:---|:---:|:---:|:---:|:---:|
"""
for idx, (cls_name, row) in enumerate(pivot_counts.iterrows(), 1):
    summary_md += f"| {idx} | {cls_name} | {int(row.get('train', 0))} | {int(row.get('val', 0))} | {int(row.get('test', 0))} | {int(row['total'])} |\n"

summary_md += f"| | **Total** | **{len(df[df['split'] == 'train'])}** | **{len(df[df['split'] == 'val'])}** | **{len(df[df['split'] == 'test'])}** | **{len(df)}** |\n"

summary_path = os.path.join(BASE_DIR, "results", "eda_summary.md")
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary_md)
print(f"[6] EDA Summary written to: {summary_path}")
print("="*60)
