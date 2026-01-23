# Bank Marketing Subscription Prediction

## 🚀 Project Goal
To develop a predictive model that identifies clients likely to subscribe to a term deposit. The focus is on optimizing business value by maximizing the **ROC-AUC** and **F1** with constraints of Precision and Recall for pos labels, ensuring high recall for potential subscribers without overwhelming the sales team with false positives.

## 📊 Data Source
The project utilizes the **Bank Marketing Dataset** (Portuguese banking institution).
- **Instances:** ~41,188 rows (full version)
- **Features:** 20 -> 15 (Demographic, Social/Economic, and Campaign-specific)
- **Target:** `y` (Binary: `yes`, `no`)

## 📁 Project Structure
```
bank-marketing/
├── data/
│   ├── bank-additional-full.csv    # Full dataset (~41K rows)
|   ├── bank-additional-full-cleaned.csv # Full dataset (~41K rows), after cleaning and feature engineering (bank_marketing_eda.ipynb)
│   └── bank-additional.csv         # Smaller subset for quick testing (not used here)
├── model/
│   ├── xgboost_model.joblib
│   ├── lightgbm_model.joblib
│   ├── random_forest_model.joblib
│   ├── logistic_regression_model.joblib
│   ├── decision_tree_model.joblib
│   ├── linear_svc_model.joblib
│   └── model_results.joblib
├── bank_marketing_eda.ipynb        # Exploratory Data Analysis and Data Cleaning
├── bank_marketing_analysis.ipynb   # Model Training & Evaluation
├── deployment.ipynb                # Model Local Deployment
├── app.py                          # Streamlit Web Application
└── README.md
```

## 🛠 Model Pipeline
1. **Data Ingestion:** Loads data with `;` delimiter and maps target to binary (1/0).
2. **Preprocessing:**
   - **Numerical:** Standard Scaling for stability in linear models.
   - **Categorical:** One-Hot Encoding (with `drop='first'` for Logistic Regression & SVC).
3. **Cross-Validation:** 5-fold Stratified Cross-Validation using `cross_val_predict` to ensure generalization.
4. **Threshold Moving:** Moving beyond the default 0.5 threshold to a data-derived optimum (~0.75) to maximize the F1-score with precision (>.5) and recall (>.8/.7/.6) constraints.

## 📈 Performance Summary
- **Best Model:** LightGBM (Tuned)
- **Test AUC-ROC:** 0.953
- **Recall:** 0.82
- **Threshold Optimization:** Threshold from 0.5 to 0.75

## 💻 Technical Setup

### Prerequisites
- Python 3.12.7
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- lightgbm, xgboost, imblearn, tensorflow
- streamlit (for web app), joblib (for model saving)

### Execution
Run the cells in `bank_marketing_analysis.ipynb` sequentially. The notebook will:
1. Preprocess the CSV data.
2. Establish baseline performance across multiple models.
3. Fine tune hyperparameters through systematic grid search for the models.
4. Select the best performing model based on ROC-AUC.
5. Save the best performing model to a joblib file.

### Web Application
Launch the Streamlit app for interactive predictions:
```bash
streamlit run app.py
```

## 🔑 Key Findings
1. **Threshold Optimization:** Moving from the default 0.5 threshold to ~0.75 improved F1-score by better balancing precision and recall.
2. **Feature Importance:** Campaign-related features (e.g., `duration`, `campaign`) and economic indicators (e.g., `euribor3m`) were among the most important.
3. **Class Imbalance:** The dataset has significant class imbalance (~88% non-subscribers), making accuracy a less appropriate metric. Sampling methods (e.g., SMOTE) do not improve ensemble models' performance, yet they are still somewhat useful for linear models.
4. **Model Comparison:** Ensemble models (LightGBM > XGBoost) showed strongest performance over other types of models.