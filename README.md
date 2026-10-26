# Milling Machine Predictive Maintenance Failure Classification

## a. Problem Statement
The objective of this project is to develop and evaluate five supervised machine learning models to predict mechanical equipment failure in manufacturing operations. Utilizing multivariate sensor readings (rotational speed, torque, process temperatures, and tool wear), the models provide early warning signals to prevent unscheduled downtime, optimize maintenance scheduling, and mitigate equipment damage.

## b. Dataset Description
- **Source**: AI4I 2020 Predictive Maintenance Dataset (Kaggle / UCI Machine Learning Repository)
- **Total Records**: 10,000 instances
- **Target Variable**: `Machine failure` (0 = Operational / Normal, 1 = Failure)
- **Engineered Predictive Features (14 total)**:
  - `Air temperature [K]`
  - `Process temperature [K]`
  - `Rotational speed [rpm]`
  - `Torque [Nm]`
  - `Tool wear [min]`
  - `Temp_diff` (Process temp minus Air temp)
  - `Power_W` (Calculated milling power consumption)
  - `Overstrain` (Tool wear multiplied by Torque)
  - `Temp_ratio` (Thermal balance ratio)
  - `Speed_Torque_ratio` (Mechanical load ratio)
  - `Wear_Speed_prod` (Dynamic tool load)
  - `Type_H`, `Type_L`, `Type_M` (One-hot encoded product quality variants)

*(Note: Row identifiers and failure mode flags TWF, HDF, PWF, OSF, and RNF were removed to eliminate data leakage).*

## c. Github Repository Link
- https://github.com/Asmi-P/predictive-maintenance-classifier

## d. Models Used

### Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8550 | 0.9414 | 0.1754 | 0.8824 | 0.2927 | 0.3545 |
| Decision Tree | 0.9550 | 0.8576 | 0.4179 | 0.8235 | 0.5545 | 0.5677 |
| KNN | 0.9735 | 0.8678 | 0.8261 | 0.2794 | 0.4176 | 0.4714 |
| Naive Bayes | 0.9510 | 0.9201 | 0.3214 | 0.3971 | 0.3553 | 0.3321 |
| Support Vector Machine | 0.9200 | 0.9697 | 0.2870 | 0.9118 | 0.4366 | 0.4858 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves a high Recall (0.8824) and AUC (0.9414) with balanced weighting, effectively catching true machine failures. However, it incurs noticeable false alarms on the severe class imbalance, resulting in low Precision (0.1754). |
| **Decision Tree** | Captures non-linear thresholds and multi-condition operational limits (torque, overstrain, temperature difference) effectively. It produces the highest overall F1 Score (0.5545) and MCC (0.5677), maintaining balanced precision and high recall. |
| **KNN** | Achieves high Accuracy (0.9735) and Precision (0.8261), but struggles severely with Recall (0.2794). Due to data imbalance, minority failure instances get outvoted by dense clusters of operational data points. |
| **Naive Bayes** | Moderate performance across metrics (MCC 0.3321). Its conditional independence assumption is challenged by inter-dependent sensor parameters (e.g., speed, torque, mechanical power, and temperatures). |
| **Support Vector Machine** | Achieves the highest AUC (0.9697) and top Recall (0.9118) using the RBF kernel. It establishes clean decision boundaries in the non-linear feature space, though precision is penalized due to the rare failure class distribution. |
| **Overall Winner for your dataset?** | **Decision Tree Classifier**. In predictive maintenance, balancing false negatives with false alarms is critical. The Decision Tree achieves the strongest combined performance with the highest F1 Score (0.5545) and MCC Score (0.5677) while maintaining over 82% Recall without excessive false positives. |