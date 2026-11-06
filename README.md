# Credit Risk Assessment: Loan Default Prediction

Predicts whether a loan applicant is **high-risk** (likely to default) or **low-risk**. The project compares three models (Logistic Regression, Random Forest and a cost-sensitive XGBoost) and serves the Random Forest as an interactive **Streamlit** app.

- **Live app:** [Live app](https://credit-risk-prediction-mqkt.onrender.com/)
- **Dataset:** [Credit Risk Dataset](https://drive.google.com/file/d/1sDXZpAvkhTe9OurKGkSQSrckjbtFb6H0/view?usp=sharing), 32,581 loan records, 12 columns (32,262 rows after cleaning)

## Results

Evaluated on a stratified, held-out 20% test set (6,453 applicants). Class 1 = default (high-risk), about 22% of the data.

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|---|---|---|---|---|
| Logistic Regression | 86.1% | 75% | 55% | 63% |
| **Random Forest (200 trees)** | **93.66%** | 97% | 74% | **84%** |
| XGBoost (cost-sensitive) | 92.0% | 83% | **81%** | 82% |

**Takeaways**

- Random Forest has the highest accuracy and F1 for defaults, and is the model served in the app.
- Because only about 22% of loans default, accuracy alone hides how many defaults are missed. The standard Random Forest misses about a quarter of them (74% recall).
- Cost-sensitive XGBoost (`scale_pos_weight` = 3.56) raises default recall to 81% at the cost of about 1.7 points of accuracy. For risk-averse lending, where missing a default is the costly error, it is the stronger choice.

## Pipeline

1. **Cleaning**
   - Median-impute `person_emp_length` (895 missing values).
   - Drop `loan_int_rate` (3,116 missing values, about 10%).
   - Remove duplicate rows.
   - Remove outliers: `person_age` > 100, `person_income` > $300,000, `person_emp_length` > 50 years.
2. **Feature engineering:** recompute `loan_percent_income` as `loan_amnt / person_income` (rounded to 2 decimals) after the cleaning steps.
3. **Encoding**
   - Binary: `cb_person_default_on_file` (Y=1, N=0).
   - Ordinal: `loan_grade` (A=1 ... G=7).
   - One-hot (`drop_first=True`): `person_home_ownership`, `loan_intent`.
4. **Split and scale:** stratified 80/20 train/test split (`random_state=42`), then `StandardScaler` fitted on the training set and applied to the numeric features (`person_age`, `person_income`, `person_emp_length`, `loan_amnt`, `loan_percent_income`, `cb_person_cred_hist_length`).
5. **Modeling**
   - Logistic Regression (`max_iter=1000`).
   - Random Forest (`n_estimators=200`, `random_state=42`).
   - XGBoost (`n_estimators=200`) with `scale_pos_weight` set to the ratio of non-defaults to defaults in the training set, to handle class imbalance.
6. **Evaluation:** accuracy, classification report, confusion matrix, and 2D PCA decision-boundary plots for each model.

All random seeds are fixed (`random_state=42`) so results are reproducible.

## Project structure

```
app.py                         # Streamlit app
rf_model.pkl                   # trained Random Forest
scaler.pkl                     # StandardScaler fitted on the training set
columns.pkl                    # exact training column order
requirements.txt
credit_risk_assessment.ipynb   # full analysis
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To re-run the notebook, you also need `xgboost` (and `matplotlib`, `seaborn`) installed.

## Deploy on Render

1. Push this repo to GitHub (all three `.pkl` files must be committed).
2. Render → **New +** → **Web Service** → connect the repo.
3. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - **Environment variable:** `PYTHON_VERSION` = `3.11.9`
4. Deploy. The free tier sleeps when idle, so the first load can take about a minute.

`requirements.txt` pins the scikit-learn version the model was trained with. A pickled model must be loaded with the same version.
