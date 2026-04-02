# Linear Regression — Insurance Charges
Predicting individual medical insurance charges using a linear regression model built with scikit-learn.

---

## Problem Statement

Insurance companies need to estimate medical costs for customers before setting premium prices. This project builds a linear regression model that predicts individual medical charges billed by health insurance, based on a customer's personal attributes — mirroring real-world actuarial and pricing models used in the insurance industry.

---

## Dataset

- **Source:** [Medical Cost Personal Dataset](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv)
- **Size:** 1,338 rows × 7 columns

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Age of the primary beneficiary |
| `sex` | Categorical | Male / Female |
| `bmi` | Numeric | Body mass index |
| `children` | Numeric | Number of dependents covered |
| `smoker` | Categorical | Yes / No |
| `region` | Categorical | Northeast, Southeast, Southwest, Northwest |
| `charges` | **Target** | Medical costs billed by insurance (USD) |

---

## Project Structure

```
LinearRegression/
├── insurance_charges.ipynb   # Main notebook
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Approach

1. **EDA** — Distribution analysis, categorical value counts, smoker vs charges boxplot
2. **Preprocessing** — Log transformation on target (`charges`), label encoding for `sex` and `smoker`, dropped `region` for first iteration
3. **Train / Test Split** — 80/20 split with `random_state=42`
4. **Feature Scaling** — `StandardScaler` fitted on training data only (no data leakage)
5. **Model** — `LinearRegression` from scikit-learn
6. **Evaluation** — R², RMSE, MAE on test set
7. **Residual Analysis** — Residuals vs predicted, distribution of residuals

---

## Results

| Metric | Value |
|---|---|
| R² | 0.7983 |
| RMSE | 0.4259 |
| MAE | 0.2755 |

The model explains ~80% of the variance in insurance charges using just 5 features. The residual plot reveals a curved pattern, indicating the data contains two distinct populations (smokers and non-smokers) that a single straight line can't fully separate. Future improvements would include interaction terms (e.g. `smoker × bmi`) or tree-based models.

---

## Key Findings

- **Smoker status** is by far the strongest predictor of charges — consistent with the EDA boxplot
- **Age** and **BMI** have positive relationships with charges
- **Sex** has a small negative coefficient (male → slightly lower charges)
- Log-transforming the target significantly improved model fit by compressing the right-skewed distribution

---

## Setup

```bash
# Activate the shared virtual environment
source /path/to/mlvenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook insurance_charges.ipynb
```
