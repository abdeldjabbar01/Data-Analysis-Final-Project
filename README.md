# Medical Insurance Cost Study - Final Report

## Dataset
- File: insurance.csv
- 1338 rows, 7 columns (age, sex, bmi, children, smoker, region, charges)
- No missing values, 1 duplicate removed

## Descriptive Statistics
- Average age: 39 years
- Average BMI: 30.7
- Average charge: $13,270
- Charges are right-skewed (max $63,770)

## Key Finding (T-test)
- Smokers pay significantly more than non-smokers
- t = 46.64, p < 0.001 → statistically significant

## Models Performance (test set)

| Model  | R²   | MAE ($) | RMSE ($) |
|--------|------|---------|----------|
| OLS    | 0.807| 4,177   | 5,956    |
| Ridge  | 0.806| 4,194   | 5,972    |
| Lasso  | 0.807| 4,177   | 5,956    |

## Top Drivers (OLS coefficients)
1. smoker_yes → +$23,077
2. age (per std) → +$3,473
3. bmi (per std) → +$1,928
4. children → +$710

Sex and region have very small effects.

## Deployment
- Streamlit dashboard created
- Uses saved model (ols_model.pkl) and scaler (scaler.pkl)
- Run locally: `streamlit run app.py`

## Conclusion
OLS explains 80.7% of variance. Smoking is the strongest predictor.