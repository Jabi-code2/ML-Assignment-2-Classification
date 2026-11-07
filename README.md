# Machine Learning Assignment - 2

## a. Problem Statement

The objective is to implement five classification models on one dataset and evaluate them using six performance metrics. The trained models are also demonstrated through a Streamlit application.

## b. Dataset Description

- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Source: UCI Machine Learning Repository
- Instances: 569
- Input features: 30
- Problem type: Binary classification
- Target: Diagnosis (malignant or benign)

## c. GitHub Repository Link

Add your actual GitHub repository URL here.

## d. Models Used

The five models are Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine.

### Model Comparison

         ML Model Name  Accuracy    AUC  Precision  Recall     F1    MCC
   Logistic Regression    0.9825 0.9954     0.9825  0.9825 0.9825 0.9623
         Decision Tree    0.9123 0.9157     0.9161  0.9123 0.9130 0.8174
                   KNN    0.9561 0.9788     0.9561  0.9561 0.9560 0.9054
           Naive Bayes    0.9386 0.9878     0.9384  0.9386 0.9384 0.8676
Support Vector Machine    0.9825 0.9950     0.9825  0.9825 0.9825 0.9623

### Model Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy: 0.9825; AUC: 0.9954; F1: 0.9825; MCC: 0.9623. Discuss the confusion matrix and errors for this model. |
| Decision Tree | Accuracy: 0.9123; AUC: 0.9157; F1: 0.9130; MCC: 0.8174. Discuss the confusion matrix and errors for this model. |
| KNN | Accuracy: 0.9561; AUC: 0.9788; F1: 0.9560; MCC: 0.9054. Discuss the confusion matrix and errors for this model. |
| Naive Bayes | Accuracy: 0.9386; AUC: 0.9878; F1: 0.9384; MCC: 0.8676. Discuss the confusion matrix and errors for this model. |
| Support Vector Machine | Accuracy: 0.9825; AUC: 0.9950; F1: 0.9825; MCC: 0.9623. Discuss the confusion matrix and errors for this model. |

### Overall Winner

The top model by weighted F1 score in this run is **Logistic Regression**. Review the other metrics and confusion matrices before writing the final conclusion.

## Streamlit Application

The app supports CSV upload, model selection, evaluation metrics, confusion matrix and classification report. It loads saved models and does not retrain on the uploaded test data.
