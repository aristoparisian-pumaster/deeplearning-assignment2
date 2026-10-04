# Skin Disease Classification Using Deep Learning

**Course**: Fundamental of Deep Learning (Week 4 Assignment 1)  
**Program**: Master of Computer Science / IT (S2), President University  
**Author**: Aristo  

---

## 📌 Project Overview
This repository contains the end-to-end implementation and comparative benchmark of three distinct Convolutional Neural Network (CNN) architectures for 9-class skin disease classification using transfer learning:
- **VGG16**: Classic deep sequential CNN
- **ResNet50**: Deep residual network with skip connections
- **MobileNetV2**: Lightweight mobile CNN with inverted residual bottlenecks

The dataset used is the [Skin Disease Classification Dataset](https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset), comprising 876 dermoscopic images across 9 classes.

---

## 📊 Dataset Classes
1. Actinic keratosis (Precancerous)
2. Atopic Dermatitis (Inflammatory)
3. Benign keratosis (Benign Tumor)
4. Dermatofibroma (Benign Histiocytoma)
5. Melanocytic nevus (Benign Melanocytic)
6. Melanoma (Malignant Melanoma)
7. Squamous cell carcinoma (Malignant Carcinoma)
8. Tinea Ringworm Candidiasis (Fungal Infection)
9. Vascular lesion (Benign Vascular)

**Dataset Split (Stratified)**:
- **Training Set**: 696 images (79.5%)
- **Validation Set**: 91 images (10.4%)
- **Testing Set**: 89 images (10.2%)

---

## 📈 Benchmark Comparison Table

| Model | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Total Parameters | Trainable Parameters | Training Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **VGG16** | **75.28%** | 0.7772 | 0.7519 | 0.7450 | 21,139,785 | 6,425,097 | 1259.4s |
| **ResNet50** | **75.28%** | 0.7756 | 0.7543 | **0.7521** | 24,034,889 | 526,857 | 601.0s |
| **MobileNetV2** | **66.29%** | 0.7057 | 0.6605 | 0.6390 | 2,554,121 | 330,249 | **218.4s** |

---

## 📂 Project Structure
```text
Assignment/
├── Assignment_Deep_Learning_Aristo.ipynb  # Interactive Jupyter Notebook
├── run_eda.py                             # Exploratory Data Analysis script
├── train_models.py                        # Model training and testing pipeline
├── requirements.txt                       # Python dependencies
├── README.md                              # Documentation
└── results/
    ├── figures/
    │   ├── eda_class_distribution.png
    │   ├── eda_sample_images.png
    │   ├── combined_validation_curves.png
    │   ├── all_confusion_matrices.png
    │   ├── vgg16_training_curves.png
    │   ├── resnet50_training_curves.png
    │   └── mobilenetv2_training_curves.png
    ├── saved_models/                      # Checkpoints (*_best.pth)
    ├── model_comparison_table.csv
    ├── model_comparison_table.md
    └── training_and_evaluation_results.json
```

---

## 🚀 How to Run

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run Exploratory Data Analysis (EDA)**:
```bash
python run_eda.py
```

3. **Train and Evaluate All 3 Models**:
```bash
python train_models.py
```
