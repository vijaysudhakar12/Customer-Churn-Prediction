# Customer Churn Prediction using Machine Learning

A machine learning project to predict whether a customer is likely to leave a service provider using customer demographics, subscription details, service usage, and billing information.

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who may leave helps businesses understand churn patterns and develop better customer retention strategies.

This project builds a **K-Nearest Neighbors (KNN) classification model** to predict customer churn. It covers data preprocessing, exploratory data preparation, handling missing values and outliers, categorical encoding, class imbalance handling, feature scaling, and model evaluation.

## Objectives

* Analyze customer data to prepare it for machine learning.
* Handle missing values and duplicate records.
* Detect and handle outliers using the Interquartile Range (IQR) method.
* Convert categorical variables into numerical features.
* Address class imbalance using random oversampling.
* Train a KNN classification model.
* Evaluate predictions using precision, recall, F1-score, accuracy, and a confusion matrix.

## Dataset Description

The dataset initially contains **1,008 records and 13 columns**.

| Feature                | Description                                             |
| ---------------------- | ------------------------------------------------------- |
| `customer_id`          | Unique customer identifier                              |
| `age`                  | Customer age                                            |
| `gender`               | Customer gender                                         |
| `tenure`               | Customer tenure                                         |
| `monthly_charges`      | Monthly service charges                                 |
| `total_charges`        | Total customer charges                                  |
| `contract_type`        | Type of subscription contract                           |
| `payment_method`       | Customer payment method                                 |
| `tech_support`         | Whether technical support is enabled                    |
| `online_security`      | Whether online security is enabled                      |
| `internet_service`     | Type of internet service                                |
| `number_of_complaints` | Number of customer complaints                           |
| `churn`                | Target variable indicating whether the customer churned |

**Target variable:** `churn`

* `0` — No churn
* `1` — Churn

*Note: The dataset file must be available locally to run the notebook. Check whether you have included it in the repository before publishing the project.*

## Technologies Used

* Python
* Pandas
* NumPy
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Project Workflow

### 1. Data Loading and Inspection

* Load the dataset using Pandas.
* Inspect the dataset dimensions, column data types, and missing values.
* Review the dataset structure before preprocessing.

### 2. Data Cleaning

**Duplicate records**

* Identify duplicate rows.
* Remove duplicate records to reduce redundant observations.

**Missing values**

The notebook uses different imputation strategies:

| Column            | Handling technique              |
| ----------------- | ------------------------------- |
| `gender`          | Remove rows with missing values |
| `payment_method`  | Remove rows with missing values |
| `monthly_charges` | Mean imputation                 |
| `total_charges`   | Median imputation               |
| `tech_support`    | Mode imputation                 |
| `online_security` | Mode imputation                 |

### 3. Outlier Detection and Handling

The Interquartile Range (IQR) method is applied to `monthly_charges` and `total_charges`.

The IQR is calculated as:

$$
IQR = Q_3 - Q_1
$$

The lower and upper boundaries are:

$$
\text{Lower Bound} = Q_1 - 1.5 \times IQR
$$

$$
\text{Upper Bound} = Q_3 + 1.5 \times IQR
$$

Observations outside these boundaries are removed for the selected columns.

### 4. Categorical Encoding

Categorical variables are converted into numerical features so that the model can process them.

* **One-Hot Encoding:** `contract_type`, `payment_method`, and `internet_service`
* **Binary Encoding:** `gender`, `tech_support`, `online_security`, and the target variable `churn`
* **Identifier Removal:** `customer_id` is removed before model training.

### 5. Train-Test Split

The dataset is divided into training and testing sets.

* Training set: 80%
* Testing set: 20%
* Random state: `42`
* Stratification: Enabled using the target variable

Stratification helps preserve the original churn-class proportions across the training and testing sets.

### 6. Handling Class Imbalance

The notebook uses **random oversampling** on the training data to balance the two target classes.

| Class          | Before oversampling | After oversampling |
| -------------- | ------------------: | -----------------: |
| No churn (`0`) |                 554 |                554 |
| Churn (`1`)    |                 216 |                554 |

Random oversampling duplicates minority-class training examples to balance the class distribution. The test set remains separate and unchanged by this oversampling step.

### 7. Feature Scaling

`StandardScaler` is used to standardize the features.

Standardization helps KNN because the algorithm calculates distances between observations, and features with larger numerical scales can otherwise dominate the distance calculation.

The scaler is fitted on the oversampled training features and then applied to the test features.

### 8. Model Training

The notebook uses the K-Nearest Neighbors (KNN) classifier with the following configuration:

```python
KNeighborsClassifier(
    n_neighbors=6,
    metric="euclidean"
)
```

The model predicts churn based on the nearest observations in the scaled feature space.

### 9. Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

These metrics help assess overall classification performance and the model's ability to identify customers who churn.

## Model Results

The following results are from the notebook's recorded model evaluation.

| Metric    | No churn (`0`) | Churn (`1`) |
| --------- | -------------: | ----------: |
| Precision |           0.80 |        0.45 |
| Recall    |           0.76 |        0.52 |
| F1-score  |           0.78 |        0.48 |
| Support   |            139 |          54 |

**Overall accuracy: 69%**

### Confusion Matrix

```text
[[105  34]
 [ 26  28]]
```

For the class order `[No churn, Churn]`, the matrix represents:

| Actual / Predicted | No churn | Churn |
| ------------------ | -------: | ----: |
| No churn           |      105 |    34 |
| Churn              |       26 |    28 |

### Result Interpretation

* The model correctly classified 105 customers who did not churn.
* It correctly identified 28 customers who churned.
* It missed 26 customers who actually churned.
* Churn recall of 0.52 means the model identified approximately 52% of actual churn cases in the test set.
* The churn F1-score of 0.48 indicates that identifying churn remains a key area for improvement.

Accuracy alone is not sufficient for churn prediction because missing customers who are likely to leave can be costly for a business.

## Key Learnings

* Data cleaning and preprocessing are essential for reliable model training.
* Missing values can be handled using different statistical imputation techniques.
* IQR is a useful method for detecting potential numerical outliers.
* Categorical encoding converts non-numerical features into a machine-readable format.
* Class imbalance can affect the model's ability to identify minority-class observations.
* Feature scaling is particularly important for distance-based algorithms such as KNN.
* Precision, recall, and F1-score provide more insight into minority-class performance than accuracy alone.

## Future Improvements

The following improvements can help make the project more robust:

1. **Prevent data leakage:** Fit imputers and outlier-handling rules using training data only, after splitting the dataset.
2. **Use a reproducible preprocessing pipeline:** Combine preprocessing, scaling, and model training with Scikit-learn pipelines.
3. **Compare classification algorithms:** Evaluate Logistic Regression, Decision Tree, Random Forest, and other suitable classifiers against KNN.
4. **Tune hyperparameters:** Use cross-validation and `GridSearchCV` to identify suitable values for `n_neighbors`, distance metrics, and other model parameters.
5. **Evaluate imbalance-handling strategies:** Compare random oversampling with class weights and SMOTE, applying resampling only to training folds.
6. **Improve churn detection:** Track churn-class recall, precision, F1-score, and precision-recall curves to select a model suited to the business objective.
7. **Improve reproducibility:** Replace machine-specific file paths with relative paths and document dataset setup instructions.

These are proposed improvements, not results already implemented in the current notebook.

## How to Run the Project

### Prerequisites

Install Python and Jupyter Notebook, then install the required libraries:

```bash
pip install pandas numpy seaborn scikit-learn jupyter
```

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/vijaysudhakar12/Customer-Churn-Prediction.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Customer-Churn-Prediction
   ```

3. Ensure the dataset file, `customer_churn_classification_dataset.csv`, is available and update the CSV path in the notebook if necessary.

4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `Customer_Churn_Prediction.ipynb` and run the cells in order.

## Conclusion

This project demonstrates an end-to-end introductory machine learning workflow for customer churn classification, from data preparation to model evaluation.

The initial KNN model achieves 69% accuracy, but its churn recall of 0.52 shows that further experimentation is needed before it could be considered reliable for business use.

The project provides a foundation for improving churn prediction through better validation, model comparison, hyperparameter tuning, and business-focused evaluation.

## Author

**Vijay Sudhakar**

GitHub: [@vijaysudhakar12](https://github.com/vijaysudhakar12)

Project Repository: [Customer Churn Prediction](https://github.com/vijaysudhakar12/Customer-Churn-Prediction)
