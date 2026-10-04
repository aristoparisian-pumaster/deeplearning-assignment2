import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import (
    vgg16, VGG16_Weights,
    resnet50, ResNet50_Weights,
    mobilenet_v2, MobileNet_V2_Weights
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Configuration
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "skin_disease_classification", "Split_smol")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODELS_DIR = os.path.join(RESULTS_DIR, "saved_models")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

print("="*65)
print("DEEP LEARNING MODEL TRAINING PIPELINE")
print(f"Device: {DEVICE}")
print(f"Batch Size: {BATCH_SIZE} | Epochs: {NUM_EPOCHS} | LR: {LEARNING_RATE}")
print("="*65)

# 1. Transforms (Preprocessing & Augmentation)
# ImageNet normalization
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# 2. Datasets & Loaders
train_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), transform=eval_transform)
test_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=eval_transform)

class_names = train_dataset.classes
num_classes = len(class_names)

print(f"Detected {num_classes} classes: {class_names}")
print(f"Dataset split sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 3. Model Builders
def get_vgg16_model(num_classes):
    model = vgg16(weights=VGG16_Weights.DEFAULT)
    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False
    # Replace classifier
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

def get_resnet50_model(num_classes):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # Freeze feature extractor
    for param in model.parameters():
        param.requires_grad = False
    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

def get_mobilenetv2_model(num_classes):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model

# 4. Training Function
def train_model(model_name, model_fn):
    print(f"\n>>> Starting Training: {model_name} <<<")
    model = model_fn(num_classes).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': []
    }

    best_val_acc = 0.0
    best_weights_path = os.path.join(MODELS_DIR, f"{model_name.lower()}_best.pth")
    start_train_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_duration = time.time() - epoch_start

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['epoch_times'].append(epoch_duration)

        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS:02d} [{epoch_duration:.1f}s] - "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

        # Checkpoint best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_weights_path)

    total_training_time = time.time() - start_train_time
    print(f"Training completed in {total_training_time:.1f}s. Best Val Acc: {best_val_acc:.4f}")

    # Load best weights for testing
    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=DEVICE))

    # Testing Phase
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    conf_mat = confusion_matrix(all_labels, all_preds)
    clf_report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)

    print(f"\n--- {model_name} Test Results ---")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {precision:.4f} (Macro)")
    print(f"Recall:    {recall:.4f} (Macro)")
    print(f"F1-Score:  {f1:.4f} (Macro)")

    return {
        'model_name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_training_time': total_training_time,
        'history': history,
        'test_acc': test_acc,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'confusion_matrix': conf_mat.tolist(),
        'classification_report': clf_report,
        'predictions': all_preds.tolist(),
        'ground_truth': all_labels.tolist()
    }

# 5. Execute Training for All 3 Models
models_dict = {
    'VGG16': get_vgg16_model,
    'ResNet50': get_resnet50_model,
    'MobileNetV2': get_mobilenetv2_model
}

results_all = {}
for name, model_fn in models_dict.items():
    res = train_model(name, model_fn)
    results_all[name] = res

# Save full results JSON
results_json_path = os.path.join(RESULTS_DIR, "training_and_evaluation_results.json")
with open(results_json_path, 'w', encoding='utf-8') as f:
    json.dump(results_all, f, indent=4)
print(f"\n[Artifact] Saved detailed evaluation results to: {results_json_path}")

# 6. Generate Training Curves Plots
for name, res in results_all.items():
    hist = res['history']
    epochs = range(1, NUM_EPOCHS + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Accuracy Plot
    ax1.plot(epochs, hist['train_acc'], 'o-', label='Train Accuracy', color='#2b5c8f', lw=2)
    ax1.plot(epochs, hist['val_acc'], 's--', label='Val Accuracy', color='#e28743', lw=2)
    ax1.set_title(f"{name} - Training & Validation Accuracy", fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Loss Plot
    ax2.plot(epochs, hist['train_loss'], 'o-', label='Train Loss', color='#2b5c8f', lw=2)
    ax2.plot(epochs, hist['val_loss'], 's--', label='Val Loss', color='#e28743', lw=2)
    ax2.set_title(f"{name} - Training & Validation Loss", fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss (Cross Entropy)', fontsize=11)
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_file = os.path.join(FIGURES_DIR, f"{name.lower()}_training_curves.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"[Artifact] Saved {name} curve to: {plot_file}")

# Combined 3-Model Validation Accuracy & Loss Comparison Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
colors = {'VGG16': '#2b5c8f', 'ResNet50': '#2ca02c', 'MobileNetV2': '#d9534f'}
styles = {'VGG16': 'o-', 'ResNet50': 's--', 'MobileNetV2': '^:'}

for name, res in results_all.items():
    epochs = range(1, NUM_EPOCHS + 1)
    ax1.plot(epochs, res['history']['val_acc'], styles[name], label=f"{name}", color=colors[name], lw=2.2)
    ax2.plot(epochs, res['history']['val_loss'], styles[name], label=f"{name}", color=colors[name], lw=2.2)

ax1.set_title("Validation Accuracy Comparison Across Models", fontsize=12, fontweight='bold')
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Validation Accuracy", fontsize=11)
ax1.set_ylim(0, 1.0)
ax1.legend(frameon=True, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.set_title("Validation Loss Comparison Across Models", fontsize=12, fontweight='bold')
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Validation Loss", fontsize=11)
ax2.legend(frameon=True, fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
comb_curve_path = os.path.join(FIGURES_DIR, "combined_validation_curves.png")
plt.savefig(comb_curve_path, dpi=300)
plt.close()
print(f"[Artifact] Saved combined curves to: {comb_curve_path}")

# 7. Confusion Matrix Visualizations
fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
for idx, (name, res) in enumerate(results_all.items()):
    ax = axes[idx]
    cm = np.array(res['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=[c[:10] for c in class_names],
                yticklabels=[c[:10] for c in class_names])
    ax.set_title(f"{name} Confusion Matrix\n(Test Acc: {res['test_acc']*100:.1f}%)", fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Class', fontsize=10)
    ax.set_ylabel('True Class', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

plt.suptitle("Confusion Matrix Comparison on Unseen Test Set (89 Images)", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
cm_all_path = os.path.join(FIGURES_DIR, "all_confusion_matrices.png")
plt.savefig(cm_all_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[Artifact] Saved all confusion matrices to: {cm_all_path}")

# 8. Comparison Table Generation
comparison_data = []
for name, res in results_all.items():
    comparison_data.append({
        'Model': name,
        'Accuracy (%)': f"{res['test_acc'] * 100:.2f}%",
        'Precision (Macro)': f"{res['test_precision']:.4f}",
        'Recall (Macro)': f"{res['test_recall']:.4f}",
        'F1-Score (Macro)': f"{res['test_f1']:.4f}",
        'Total Parameters': f"{res['total_params']:,}",
        'Trainable Parameters': f"{res['trainable_params']:,}",
        'Training Time (s)': f"{res['total_training_time']:.1f}s"
    })

comp_df = pd.DataFrame(comparison_data)
comp_csv_path = os.path.join(RESULTS_DIR, "model_comparison_table.csv")
comp_df.to_csv(comp_csv_path, index=False)

comp_md = f"""# Model Comparison Table

| Model | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Total Params | Trainable Params | Training Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
for _, row in comp_df.iterrows():
    comp_md += f"| **{row['Model']}** | {row['Accuracy (%)']} | {row['Precision (Macro)']} | {row['Recall (Macro)']} | {row['F1-Score (Macro)']} | {row['Total Parameters']} | {row['Trainable Parameters']} | {row['Training Time (s)']} |\n"

comp_md_path = os.path.join(RESULTS_DIR, "model_comparison_table.md")
with open(comp_md_path, 'w', encoding='utf-8') as f:
    f.write(comp_md)

print("\n" + "="*65)
print("FINAL COMPARISON TABLE:")
print(comp_df.to_string(index=False))
print("="*65)
