# Python for Data Science

Coursework repository containing three assignments covering the core Python data-science stack — NumPy, pandas, unsupervised learning, linear models, and logistic regression from scratch.

## Contents

### Assignment 1 — Fraud Detection & Customer Segmentation

- **Fraud detection**: Loads a credit card transaction dataset (`creditcard_for_fraud_detection.csv`), scales `Time` and `Amount` with `StandardScaler`, and applies an `IsolationForest` (contamination = 0.002) to flag anomalous transactions. Visualizes flagged anomalies vs. normal transactions on a `Time` vs. `Amount` scatter plot.
- **Customer profiling**: Works with `Credit Card Customer Data.csv` (average credit limit, number of cards, bank/online visits, calls made). Rescales all features to a 1–10 range, then implements **K-Means clustering from scratch** (random centroid init, geometric-mean centroid updates, PCA-based 2D visualization of clusters per iteration). Uses the elbow method (WCSS vs. k) to justify a choice of **k = 7**.
- **Part (b)**: Drops weakly/redundantly correlated columns (`Sl_No`, `Customer Key`, `Total_visits_bank`, `Total_calls_made`) based on a correlation matrix, then implements **multiple linear regression from scratch** (closed-form normal equation) to predict `Avg_Credit_Limit` from the remaining features, evaluated with mean absolute error.

### Assignment 2 — NumPy & Video Game Sales EDA

- **NumPy fundamentals**: Random 5×5 integer matrix — element indexing, per-row means, boolean masking (values above the overall mean), and a hand-written **spiral traversal** function.
- **Video Game Sales dataset** (`videogamesales.csv`):
  - Computes `global_sales` as the row-wise sum of regional sales columns.
  - Bar chart of total copies sold by genre.
  - Filters and reshapes all "Grand Theft Auto" entries into a per-platform/per-year sales summary.
  - Pie chart of total sales split across North America, Europe, Japan, and other regions (with the Japan slice exploded for emphasis).

### Final Assignment — Logistic Regression from Scratch

Implements a `Logistic_Regression` class using **batch gradient descent** (no scikit-learn), including:
- Sigmoid-based `fit` / `update_weights` / `predict` methods
- A custom `accuracy` scorer

Applied to the UCI **Heart Disease dataset** (`heart.csv`) to predict presence of heart disease from clinical features (age, sex, chest pain type, resting BP, cholesterol, max heart rate, etc.). Trained on the first 242 rows, evaluated on the remaining 61, achieving **~98.4% accuracy** at `learning_rate=0.001`, `iterations=300`.

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

## Notes

- The K-Means and linear regression implementations in Assignment 1, and the logistic regression implementation in the Final Assignment, are all written from first principles (no `sklearn` model fitting) as part of the learning exercise — `sklearn` is only used for `IsolationForest`, `StandardScaler`, and `PCA` (for visualization).
