# Skin Cancer Classification Project

A deep learning project for skin cancer classification using hybrid CNN-ViT architecture on ISIC2024 and PAD-UFES-20 datasets.

##  Datasets

### ISIC2024 Dataset
- **Size**: 29,868 images (393 malignant, 29,475 benign)
- **Format**: Variable size images, resized to 224×224 pixels
- **Features**: 161+ metadata features including:
  - 34 numerical columns (NUM_COLS)
  - 43 generated columns (NEW_NUM_COLS)
  - 6 categorical columns (CAT_COLS) with one-hot encoding
  - 1 special column (count_per_patient)
  - 77 normalized columns (NORM_COLS)
  - 12 image feature columns from ResNet18 extraction

### PAD-UFES-20 Dataset
- **Size**: 2,298 images from 1,373 patients
- **Lesion Types**: 6 categories (BCC, MEL, SCC, ACK, NEV, SEK)
- **Features**: 26 clinical metadata features including:
  - Quantitative: age, diameter_1, diameter_2
  - Binary symptoms: itch, grew, hurt, changed, bleed, elevation, biopsed
  - Target mapping: Benign (ACK, NEV, SEK) = 0, Malignant (BCC, MEL, SCC, BOD) = 1

##  Model Architecture

### Hybrid CNN-ViT Model
```
Input (224×224×3)
    ↓
┌─────────────────────┐    ┌─────────────────────┐
│   EfficientNet-B0   │    │   Vision Transformer │
│   (CNN Branch)      │    │   (ViT Branch)      │
│                     │    │                     │
│   Features: 1280    │    │   Features: 768     │
└─────────────────────┘    └─────────────────────┘
    ↓                           ↓
┌─────────────────────┐    ┌─────────────────────┐
│   Global Avg Pool   │    │   Global Avg Pool   │
└─────────────────────┘    └─────────────────────┘
    ↓                           ↓
    └─────────── Concatenate ───────────┘
                    ↓
            ┌─────────────────────┐
            │   FC Layer (512)    │
            │   ReLU + Dropout    │
            └─────────────────────┘
                    ↓
            ┌─────────────────────┐
            │   FC Layer (256)    │
            │   ReLU + Dropout    │
            └─────────────────────┘
                    ↓
            ┌─────────────────────┐
            │   Output (1)        │
            │   Sigmoid           │
            └─────────────────────┘
```

### Available Models
1. **EfficientNet-B0**: Pure CNN approach
2. **Vision Transformer (ViT)**: Pure transformer approach  
3. **Hybrid CNN-ViT**: Combined architecture (recommended)

##  Data Processing Pipeline

### Image Processing
1. **Preprocessing**: Resize to 224×224, RGB conversion, pixel normalization
2. **Feature Extraction**: ResNet18 pre-trained model for deep features
3. **Augmentation**: Flip, rotation, brightness, blur, erasing (training only)

### Metadata Processing
1. **Missing Values**: Imputation strategies
2. **Categorical Encoding**: One-hot encoding for categorical variables
3. **Feature Engineering**: New feature creation and normalization
4. **SMOTE**: Applied on feature space (ISIC2024 only, PAD-UFES-20 is balanced)

### Data Splitting
- **ISIC2024**: 80% train, 10% validation, 10% test
- **Cross-validation**: 3-fold StratifiedGroupKFold
- **Patient-level splitting**: Prevents data leakage

##  Usage

### Requirements
```bash
pip install torch torchvision timm
pip install scikit-learn lightgbm xgboost
pip install pandas polars numpy matplotlib seaborn
pip install imbalanced-learn tqdm joblib pillow
```

### Training ISIC2024 Model
```bash
python SCC_isic.py
```

### Training PAD-UFES-20 Model
```bash
python SCC_pad.py
```

### Configuration
Key parameters in the scripts:
- `MODEL_TYPE`: 'efficientnet', 'vit', or 'hybrid'
- `USE_SMOTE`: Enable/disable SMOTE sampling
- `N_SPLITS`: Number of cross-validation folds
- `IMG_MODEL_BATCH_SIZE`: Batch size for training

##  Performance Results

### ISIC2024 Results (with SMOTE)
| Model | AUC | Raw pAUC (0-20%) | Norm pAUC (0-20%) |
|-------|-----|-------------------|-------------------|
| Hybrid | 0.9408 | 0.1427 | 0.7136 |
| ViT | 0.9405 | 0.1403 | 0.7013 |
| CNN | 0.9311 | 0.1418 | 0.7090 |

### PAD-UFES-20 Results
| Model | Class | Precision | Recall | F1-score |
|-------|-------|-----------|--------|----------|
| Hybrid | Benign | 0.9444 | 0.5484 | 0.6939 |
| Hybrid | Malignant | 0.6387 | 0.9612 | 0.7674 |
| ViT | Overall | 0.7338 | 0.9113 | 0.8129 |

##  Key Features

- **Multi-modal Learning**: Combines image and metadata features
- **Advanced Augmentation**: SMOTE on feature space rather than raw images
- **Cross-validation**: Robust evaluation with patient-level splitting
- **Multiple Architectures**: EfficientNet, ViT, and Hybrid options
- **Clinical Relevance**: Real-world applicable with PAD-UFES-20 dataset


##  Research Contributions

1. **Hybrid Architecture**: Novel combination of CNN and ViT for skin lesion classification
2. **Feature-level SMOTE**: Applying data augmentation on extracted features rather than raw images
3. **Multi-dataset Validation**: Comprehensive evaluation on both research and clinical datasets
4. **Metadata Integration**: Effective fusion of image and clinical metadata

##  Evaluation Metrics

- **AUC-ROC**: Area Under Receiver Operating Characteristic curve
- **pAUC**: Partial AUC for specific false positive rate ranges
- **Precision, Recall, F1-score**: Standard classification metrics
- **Confusion Matrix**: Detailed classification analysis

---

*This project implements state-of-the-art deep learning techniques for automated skin cancer detection, combining the strengths of convolutional neural networks and vision transformers for improved diagnostic accuracy.*
