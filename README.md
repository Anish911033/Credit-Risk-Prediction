# Credit Risk Assessment: Loan Default Prediction

Predicts whether a loan applicant is **high-risk** (likely to default) or **low-risk**, and serves the best model (Random Forest) as an interactive **Streamlit** app.

- **Live app:** _add your Render URL here_
- **Dataset:** [Credit Risk Dataset (Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset), 32,581 loan records

## Results

Accuracy on a held-out 20% test set (6,453 applicants):

| Model | Accuracy |
|---|---|
| Perceptron | 80.2% |
| Logistic Regression | 86.1% |
| SVM (RBF kernel) | 91.4% |
| **Random Forest (200 trees)** | **93.46%** |

## Pipeline

1. **Cleaning:** median-impute employment length, drop `loan_int_rate` (~10% missing), remove duplicates and outliers (age > 100, income > $300k, employment > 50 years)
2. **Feature engineering:** recompute `loan_percent_income` from loan amount and income
3. **Encoding:** binary (default history), ordinal (loan grade A→G), one-hot (home ownership, loan intent)
4. **Split and scale:** stratified 80/20 split, `StandardScaler` on numeric features
5. **Modeling:** Perceptron, Logistic Regression, SVM, Random Forest (`random_state=42`)

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

## Deploy on Render

1. Push this repo to GitHub (all three `.pkl` files must be committed).
2. Render → **New +** → **Web Service** → connect the repo.
3. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - **Environment variable:** `PYTHON_VERSION` = `3.11.9`
4. Deploy. The free tier sleeps when idle, so the first load can take about a minute.

`requirements.txt` pins the scikit-learn version the model was trained with. A pickled model must be loaded with the same version.
