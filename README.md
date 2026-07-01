# 🍎 Fresh or Rotten Fruits Scanner

A deep learning project that explores transfer learning for fruit freshness classification using DenseNet architectures.

## Overview

The goal of this project was to understand what backend model architecture would best support a future fruit-scanning app. Instead of jumping directly into app development, we focused on building a strong machine learning foundation.

We trained and evaluated DenseNet121 and DenseNet201 models on a 13,000+ image fruit dataset to classify fresh vs. rotten produce. The project compares two strategies: training from scratch and using ImageNet pre-trained weights with fine-tuning.

The pre-trained DenseNet201 model converged much faster and achieved about **99.7% test accuracy** with very low loss, while also showing smoother validation curves. This validated that transfer learning is highly effective for small-to-medium computer vision datasets.

## Motivation

Consumers often struggle to judge fruit freshness beyond obvious visual cues. Grocery prices are increasing, and buying damaged or quickly spoiled produce can lead to food waste. Existing fruit quality detection systems are usually designed for farms or facilities and are not easily accessible to everyday consumers.

This project explores whether a deep learning model can serve as the backend foundation for a future real-time fruit scanning application.

## Dataset

We used the **Fresh and Rotten Fruits Dataset**.

- Training images: **10,901**
- Testing images: **2,698**
- Total images: **13,599**
- Number of classes: **6**

Classes include:

- Fresh Apples
- Rotten Apples
- Fresh Bananas
- Rotten Bananas
- Fresh Oranges
- Rotten Oranges

The dataset includes different backgrounds, salt-and-pepper noise, and varied image conditions to help test model robustness.

## Project Pipeline

```text
Dataset
   ↓
Image Preprocessing
   ↓
CNN Model Training
   ↓
Scratch Training vs. ImageNet Transfer Learning
   ↓
Model Evaluation
   ↓
Accuracy, Loss, Convergence, and Confusion Matrix Analysis
```

## Technologies Used

- Python
- TensorFlow / Keras
- Google Colab
- NVIDIA GPU
- NumPy
- Matplotlib
- DenseNet121
- DenseNet201
- Transfer Learning

## Data Preprocessing

All images were resized to:

```text
224 × 224 × 3
```

The preprocessing pipeline included:

- Image resizing
- Folder-based label encoding
- Train/test dataset loading
- Batch processing
- Normalization
- Evaluation under noisy and varied image conditions

## Model Architectures

### DenseNet201 Trained from Scratch

The scratch model used random initialization and trained the full DenseNet201 network end-to-end on the fruit dataset.

### DenseNet201 with ImageNet Pre-training

The pre-trained model used DenseNet201 initialized with ImageNet weights. The original classification head was removed and replaced with a new classifier for 6 fruit freshness classes.

The model structure included:

- DenseNet201 feature extractor
- Frozen base layers
- GlobalAveragePooling2D
- Dense layer with Softmax activation

## Why DenseNet201?

We selected DenseNet201 because its dense connectivity improves gradient flow and feature reuse, which stabilizes training in deeper architectures.

For fruit freshness classification, the model needs to detect subtle texture and color variations, such as bruising, mold, discoloration, and rotten spots. A deeper architecture like DenseNet201 helps capture these fine-grained visual patterns.

DenseNet201 also has strong ImageNet pre-trained weights, making it a good fit for transfer learning on a moderate-sized dataset. Empirically, DenseNet201 outperformed DenseNet121 and ResNet50 in both accuracy and convergence speed, validating the architectural choice.

## Training Strategy

The model was trained using:

- Optimizer: **Adam**
- Loss function: **Categorical Crossentropy**
- Epochs: **30**
- Batch size: **32**
- Learning rate scheduling
- Early stopping
- Model checkpointing based on validation loss

Learning rate decay was used to take larger steps early in training and smaller steps later for fine-tuning. Model checkpointing saved the best model when validation loss improved.

## Results

| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| VGG16 | 22.3% | 1.7753 |
| ResNet50 | 83.1% | 1.1303 |
| DenseNet121 | 99.1% | 0.0212 |
| DenseNet201 Scratch | 99.70% | 0.0101 |
| DenseNet201 ImageNet Pre-trained | 99.74% | 0.0144 |

## Key Findings

- DenseNet201 achieved **99.7%+ test accuracy**.
- The ImageNet pre-trained DenseNet201 converged in about **8 epochs**.
- The scratch-trained DenseNet201 required around **20 epochs** to converge.
- Transfer learning improved convergence speed and training stability.
- Pre-trained DenseNet201 showed smoother validation curves and strong generalization.
- DenseNet201 outperformed DenseNet121 and ResNet50 on this dataset.

## Conclusion

This project demonstrated that transfer learning is highly effective for small-to-medium image classification datasets. The pre-trained DenseNet201 model provided strong accuracy, faster convergence, and stable validation performance, making it a strong backend candidate for a future real-time fruit freshness scanning application.

The key takeaway is that pre-training significantly improves model stability, convergence speed, and generalization, which are critical for deploying a computer vision model in real consumer environments.

## Future Work

- Expand the dataset to more fruit and vegetable categories
- Improve performance under real-world lighting conditions
- Add stronger data augmentation
- Deploy the model using TensorFlow Lite
- Build a mobile or web-based fruit scanning interface
- Combine classification with object detection models such as YOLO
- Optimize inference speed for real-time use

## My Contributions

- Proposed the original fruit-scanning app idea and machine learning direction
- Helped define the project scope and model evaluation strategy
- Evaluated DenseNet-based CNN architectures for fruit freshness classification
- Compared scratch training with ImageNet transfer learning
- Analyzed model accuracy, loss, convergence behavior, and generalization
- Presented the results section and contributed to the final project documentation

## Authors

- Yeqiao Yu
- Jiachen Wang
- John Liu
