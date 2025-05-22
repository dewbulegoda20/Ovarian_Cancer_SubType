# 🧬 Ovarian Cancer Subtype Classification

A deep learning-based project aimed at classifying ovarian cancer into subtypes using histopathological images. This work leverages convolutional neural networks (CNNs) to improve diagnostic precision and aid in personalized treatment strategies.

## 📌 Project Overview

Ovarian cancer is among the most lethal gynecological malignancies, and accurate subtype classification is critical for effective treatment planning. This project uses a curated dataset of labeled ovarian cancer images to train and evaluate a deep learning model capable of distinguishing between different cancer subtypes.

## 🚀 Objectives

- Build a CNN model to classify ovarian cancer into distinct subtypes.
- Improve prediction accuracy through data preprocessing and class imbalance handling.
- Evaluate model performance using appropriate classification metrics.

## 🧪 Technologies & Tools

- **Python**
- **Jupyter Notebook**
- **TensorFlow / Keras**
- **OpenCV / PIL**
- **NumPy / Pandas / Matplotlib / Seaborn**
- **scikit-learn**


## 📊 Dataset

- The dataset contains labeled histopathological images of ovarian cancer tissue.
- Each image corresponds to a known subtype (e.g., serous, mucinous, endometrioid, clear cell).
- Data sourced from [Kaggle]([https://www.kaggle.com](https://www.kaggle.com/datasets/sunilthite/ovarian-cancer-classification-dataset)).
- Preprocessing includes resizing, normalization, and data augmentation.

## 🔍 Key Features

- **Custom CNN Architecture**: Built and trained from scratch for robust image classification.
- **Data Augmentation**: Techniques like rotation, zoom, and flipping to improve generalization.
- **Class Imbalance Handling**: Used oversampling and/or class weights to tackle skewed class distribution.
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix for evaluation.

## 📈 Results

- Model demonstrates strong classification performance across subtypes with minimal overfitting.

## 💡 Future Improvements

- Experiment with transfer learning (e.g., ResNet, EfficientNet).
- Integrate Grad-CAM for visual explanation of model predictions.
- Deploy model via a web app using Streamlit or Flask.


