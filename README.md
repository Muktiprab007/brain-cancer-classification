# Brain Cancer Detection — Transfer Learning Comparison

Binary classification of brain MRI images (Tumor vs. Healthy) using transfer learning with InceptionV3, ResNet50, and DenseNet121.

## Results

| Model       | Accuracy | ROC-AUC | Sensitivity | Specificity |
|-------------|----------|---------|-------------|-------------|
| InceptionV3 | 98.75%   | 0.9936  | 0.974       | 1.000       |
| DenseNet121 | 93.75%   | 0.9936  | 0.872       | 1.000       |
| ResNet50    | 90.62%   | 0.9936  | 0.808       | 1.000       |

> **Best model: InceptionV3** — highest accuracy and sensitivity with zero false positives.

## Dataset

- Source: [Brain Cancer Detection MRI Images](https://www.kaggle.com/datasets/hamzahabib47/brain-cancer-detection-mri-images)
- Classes: `Tumor`, `Healthy`
- Input size: 224×224 pixels (grayscale → converted to 3-channel for pretrained models)
- Split: 80% train / 20% validation (stratified)

## Pipeline

1. Data loading and visualization
2. Preprocessing — resize, normalize, grayscale to RGB conversion
3. Augmentation — rotation, shift, zoom, horizontal flip
4. Model training — frozen pretrained base + custom classification head
5. Evaluation — confusion matrix, classification report, ROC-AUC
6. Inference — random sample prediction from validation set

## Model Architecture

All three models share the same classification head:

```
GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)
```

Pretrained weights from ImageNet. Base layers frozen during training.

## Training Configuration

| Parameter     | Value                  |
|---------------|------------------------|
| Optimizer     | Adam (lr = 1e-4)       |
| Loss          | Binary Crossentropy    |
| Batch size    | 32                     |
| Max epochs    | 50                     |
| Early stopping| Patience 5 (val_loss)  |
| LR scheduler  | ReduceLROnPlateau ×0.5 |

## Requirements

```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy pandas ipywidgets
```

## Usage

Open `brain-cancer-using.ipynb` in Kaggle or Jupyter and run all cells sequentially.  
To test classification on a random validation sample, run the inference cell at the bottom of the notebook.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Transfer%20Learning-red)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF)

## Author

**Mukti Prabowo**  
[GitHub](https://github.com/Muktiprab007) · [Kaggle](https://www.kaggle.com/muktiprab007) · [LinkedIn](https://linkedin.com/in/muktiprabowo)
