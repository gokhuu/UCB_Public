# Histopathologic Cancer Detection

## Problem Description
This project addresses the challenge of automatically detecting metastatic cancer in small image patches taken from larger digital pathology scans. The dataset consists of 220,025 images (96x96 pixels, RGB channels) divided into two classes: cancer (40.5%, 89,117 images) and non-cancer (59.5%, 130,908 images). Each image represents a small region of a lymph node section stained with Hematoxylin and Eosin (H&E), requiring the model to identify subtle cellular patterns indicative of cancerous tissue.

## Exploratory Data Analysis
The dataset exploration included several key visualizations and analyses:

- **Class Distribution Analysis**: Identified moderate class imbalance (59.5% non-cancer vs. 40.5% cancer)
- **Sample Visualization**: Displayed representative images from both classes, revealing distinctive visual patterns
- **Color Channel Analysis**: Examined pixel value distributions across RGB channels, showing the importance of color information in H&E stained images
- **Class Average Visualization**: Generated average images for both classes and compared their differences
- **Pixel-level Differences**: Calculated and visualized the absolute pixel-wise differences between class averages
- **Channel Distribution Comparison**: Compared RGB channel distributions between cancer and non-cancer samples

Data preprocessing involved:
- Normalization of pixel values (rescaling to 0-1 range)
- Data augmentation techniques including rotation, flipping, shifting, and zooming to improve model generalization
- Implementation of class weights to address the moderate class imbalance

## Model Architecture
A Convolutional Neural Network (CNN) architecture was implemented for this binary classification task. The model consisted of:

- **Input Layer**: Accepting 96x96x3 images (RGB color channels preserved)
- **Convolutional Blocks**: Four blocks, each containing:
  - Conv2D layers with increasing filter sizes (32→64→128→256)
  - Batch normalization for training stability
  - MaxPooling2D layers to reduce spatial dimensions
  - Dropout layers (0.25-0.5) to prevent overfitting
- **Fully Connected Layers**:
  - Flattened output from convolutional layers
  - Dense layer with 256 neurons and ReLU activation
  - Dropout (0.5)
  - Output layer with sigmoid activation for binary classification

The model was compiled with:
- Adam optimizer with learning rate of 0.001
- Binary crossentropy loss function
- Accuracy and AUC metrics for performance monitoring

Training utilized:
- Early stopping to prevent overfitting
- Learning rate reduction when performance plateaued
- Model checkpointing to save the best performing model
- Class weights to address the class imbalance

### Model Summary
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 96, 96, 32)     │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 96, 96, 32)     │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 48, 48, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 48, 48, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 48, 48, 64)     │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 24, 24, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 24, 24, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (None, 24, 24, 128)    │           512 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 12, 12, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 12, 12, 256)    │       295,168 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_3           │ (None, 12, 12, 256)    │         1,024 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_3 (MaxPooling2D)  │ (None, 6, 6, 256)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 9216)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 256)            │     2,359,552 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 1)              │           257 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```

The total number of trainable parameters is 2,750,145, with the majority in the dense layer after flattening. This architecture progressively reduces spatial dimensions while increasing feature depth, allowing the model to learn hierarchical patterns from low-level features to high-level cancer-specific indicators.

## Results and Analysis
The model achieved excellent performance on the validation and test sets:

- **Overall Accuracy**: 92%
- **ROC AUC Score**: 0.98
- **Precision**: 
  - Non-Cancer: 0.90
  - Cancer: 0.97
- **Recall**:
  - Non-Cancer: 0.98
  - Cancer: 0.84
- **F1-Score**:
  - Non-Cancer: 0.94
  - Cancer: 0.90

The confusion matrix revealed:
- 25,780 true negatives (correctly identified non-cancer)
- 402 false positives
- 2,911 false negatives
- 14,912 true positives (correctly identified cancer)

Analysis of test predictions showed:
- Highly confident predictions for most samples (86.5%)
- 64.6% confidently classified as non-cancer (p < 0.2)
- 21.9% confidently classified as cancer (p > 0.8)
- Only 13.5% of predictions fell in the uncertain range (0.2 ≤ p ≤ 0.8)

Training exhibited some volatility in early epochs but stabilized, indicating effective learning despite the challenging nature of the classification task.

## Conclusion
The developed CNN model demonstrated strong performance in automatically detecting metastatic cancer in histopathologic scan patches. Key findings and takeaways include:

- The model achieved high discriminative ability (AUC = 0.98) on this challenging medical imaging task
- Data augmentation and class weighting techniques successfully addressed the moderate class imbalance
- The model makes confident predictions for the majority of test samples (86.5%)
- The higher precision for cancer detection (0.97) makes the model particularly valuable for minimizing false positives

Potential future improvements include:
- Implementing ensemble methods combining multiple models
- Exploring more advanced architectures like EfficientNet or Vision Transformers
- Developing visualization tools for model interpretability (e.g., GradCAM)
- Integrating the model into a clinical workflow with a confidence-based triage system
- Extending the model to detect specific cancer subtypes or grades
- Conducting clinical validation studies with practicing pathologists

This project demonstrates the potential of deep learning approaches to assist pathologists in cancer detection, potentially improving diagnostic efficiency and accuracy in clinical settings.