# Machine Learning Assignment - 2
## Classification Models and Evaluation

## a. Problem Statement

The objective of this assignment is to implement five machine learning classification models on the selected dataset and compare their performance using six evaluation metrics.

## b. Dataset Description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository
- **Total Instances:** 569
- **Number of Input Features:** 30
- **Problem Type:** Binary Classification
- **Target Variable:** Diagnosis (Malignant or Benign)

## c. GitHub Repository Link

**GitHub Repository:** Add your actual GitHub repository URL here.

## d. Models Used

The following five classification models were implemented and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Model Performance Comparison

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |   F1 Score |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-----------:|-------:|
| Logistic Regression    |     0.9825 | 0.9954 |      0.9825 |   0.9825 |     0.9825 | 0.9623 |
| Decision Tree          |     0.9123 | 0.9157 |      0.9161 |   0.9123 |     0.913  | 0.8174 |
| KNN                    |     0.9561 | 0.9788 |      0.9561 |   0.9561 |     0.956  | 0.9054 |
| Naive Bayes            |     0.9386 | 0.9878 |      0.9384 |   0.9386 |     0.9384 | 0.8676 |
| Support Vector Machine |     0.9825 | 0.9950 |      0.9825 |   0.9825 |     0.9825 | 0.9623 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performs well with good overall accuracy and balanced results. |
| Decision Tree | Provides good classification results but may be less consistent. |
| KNN | Provides reasonable results based on similarity between data points. |
| Naive Bayes | Offers simple and fast classification with reasonable performance. |
| Support Vector Machine | Provides strong classification performance with balanced results. |

### Overall Winner for the Dataset

**Logistic Regression**

The overall winner is selected based on the highest weighted F1 score among the five models.

## Streamlit Application

The Streamlit application allows users to upload test data, select a classification model and view evaluation metrics, confusion matrix and classification report.

The application loads the saved models and evaluates the uploaded test data without retraining.
