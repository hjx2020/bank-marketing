# Bank Marketing Subscription Prediction

## 🚀 Project Goal
To develop a predictive model that identifies clients likely to subscribe to a term deposit. The focus is on optimizing business value by maximizing the **macro-averaged F1-score**, ensuring high recall for potential subscribers without overwhelming the sales team with false positives.

## 📊 Data Source
The project utilizes the **Bank Marketing Dataset** (Portuguese banking institution).
- **Instances:** ~41,188 rows (full version)
- **Features:** 20 (Demographic, Social, Economic, and Campaign-specific)
- **Target:** `y` (Binary: 'yes', 'no')

## 📁 Project Structure
```
bank-marketing/
├── data/
│   ├── bank-additional-full.csv    # Full dataset (~41K rows)
|   ├── bank-additional-full-cleaned.csv # Full dataset (~41K rows), after cleaning and feature engineering (bank_marketing_eda.ipynb)
│   └── bank-additional.csv         # Smaller subset for quick testing (not used here)
├── model/
│   ├── xgboost_model.joblib
│   └── lightgbm_model.joblib
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
4. **Threshold Moving:** Moving beyond the default 0.5 threshold to a data-derived optimum (~0.76) to maximize the F1-score.

## 📈 Performance Summary
- **Best Model:** LightGBM (Tuned)
- **Test AUC-ROC:** 0.9524
- **Macro Avg Recall:** 0.80
- **F1 Score Optimization:** Threshold from 0.5 to 0.64

## 💻 Technical Setup

### Prerequisites
- Python 3.12.7
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- streamlit (for web app), joblib (for model saving)
- lightgbm, xgboost, imblearn, tensorflow

### Execution
Run the cells in `bank_marketing_analysis.ipynb` sequentially. The notebook will:
1. Preprocess the raw CSV data.
2. Establish baseline performance across multiple models.
3. Select top 2 best performing models and optimize hyperparameters through systematic grid search.
4. Select the best performing model.
5. Save the best performing model to a joblib file.

### Web Application
Launch the Streamlit app for interactive predictions:
```bash
streamlit run app.py
```

## 🔍 Main Visualizations
- **Confusion Matrix:** Evaluates model errors on a per-class basis.
- **Precision-Recall Curve:** Demonstrates the trade-off between capturing all subscribers vs. accuracy.
- **F1 vs. Threshold Plot:** Identifies the mathematical "sweet spot" for classification.

## 🔑 Key Findings
1. **Threshold Optimization:** Moving from the default 0.5 threshold to ~0.72 improved F1-score by better balancing precision and recall.
2. **Feature Importance:** Campaign-related features (e.g., `duration`, `pdays`) and economic indicators (e.g., `euribor3m`) were among the most predictive.
3. **Class Imbalance:** The dataset has significant class imbalance (~88% non-subscribers), making F1-score a more appropriate metric than accuracy. Sampling methods (e.g., SMOTE) do not improve ensemble models' performance, yet they are still somewhat useful for linear models.
4. **Model Comparison:** Ensemble models (LightGBM > XGBoost) showed strongest performance over other types of models.