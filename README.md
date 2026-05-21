# Brain Cancer Detection — Transfer Learning Comparison

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Transfer%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

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

Open the ipynb file in Kaggle or Jupyter and run all cells sequentially.  
To test classification on a random validation sample, run the inference cell at the bottom of the notebook.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Transfer%20Learning-red)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF)

## Author

**Mukti Prabowo**  
[GitHub](https://github.com/Muktiprab007) · [Kaggle](https://www.kaggle.com/muktiprab007) · [LinkedIn](https://linkedin.com/in/muktiprabowo)
