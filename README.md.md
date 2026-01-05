# 📘 Best practice Machine Learning Models

This document defines the **minimum rules, structure, and best practices** for developing Machine Learning models within this project.

👉 It is designed for partners with **limited experience**, while following **real, professional ML standards**.

---

## 🎯 Objective

- Ensure **all models are comparable**
- Avoid common mistakes (data leakage, inconsistent metrics, confusing names)
- Enable easy analysis, visualization, and decision‑making
- Reduce review and iteration time

> **Golden rule:** if your model cannot be compared easily with another one, it is incorrectly implemented.

---

## 📁 Mandatory project structure

```
project/
├── data/
│   ├── raw/            # Original data (DO NOT modify)
│   ├── processed/      # Cleaned data
│   └── features/       # Final datasets for modeling
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   └── 03_models.ipynb
├── src/
│   ├── data_split.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── models/
│   └── metrics_summary.csv
├── outputs/
│   └── plots/
└── README.md
```

📌 **Important**
- Do not modify `data/raw`
- All final metrics must be stored in `models/metrics_summary.csv`
- All plots must be saved in `outputs/plots/`

---

## 🔤 Naming conventions (CRITICAL)

### 🔹 Datasets and splits

Always use these exact names:

```python
X_train, X_test
y_train, y_test
```

If validation is used:

```python
X_train, X_val, X_test
y_train, y_val, y_test
```

❌ Incorrect:
```python
X1, X2, train, test, ytemp
```

---

### 🔹 Feature variables

Variable names must describe **what they contain**, not how they were created.

✅ Correct:
```python
features_numeric
features_categorical
X_features_final
```

❌ Incorrect:
```python
df2, temp, x
```

---

## 🧠 Model naming convention

Each model must have a unique and descriptive name.

```python
model_name = "logreg_v1_baseline"
model_name = "rf_v2_class_weight"
model_name = "xgb_v3_feature_eng"
```

### 📌 Recommended format

```
<model>_<version>_<key_detail>
```

Examples:
- `logreg_v1_baseline`
- `rf_v1_no_balance`
- `xgb_v2_tuned`

---

## 🏋️ Model training (single mandatory pattern)

All models must follow this exact flow:

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

🚫 Forbidden:
- Training on `X_test`
- Hyperparameter tuning using the test set
- Evaluating without clearly stating the dataset used

---

## 📊 Metrics (mandatory standard)

All models must report metrics using the same structure.

```python
metrics = {
    "model": model_name,
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}
```

📌 If a metric is not listed here, **it must not be compared or plotted**.

---

## 💾 Metrics storage (MANDATORY)

Each model execution must append results to the shared file:

```python
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(
    "models/metrics_summary.csv",
    mode="a",
    header=False,
    index=False
)
```

✔ This enables:
- Model comparison
- Automated plotting
- Result versioning

---

## 📈 Plots

### Rules

- One metric per plot
- X‑axis = model name
- Do not mix metrics in the same chart

Example:

```python
sns.barplot(
    data=metrics_df,
    x="model",
    y="f1"
)
plt.xticks(rotation=45)
```

---

## ⚠️ Common mistakes and how to avoid them

### ❌ Changing variable names across notebooks

```python
Xtrain, X_test2
```

✔ Always use the standard naming.

---

### ❌ Overwriting predictions

```python
y_pred = model1.predict()
y_pred = model2.predict()
```

✔ Correct approach:
- Do not store predictions
- Store metrics directly

---

### ❌ Non‑reproducible results

✔ Always set a seed:

```python
RANDOM_STATE = 42
```

---

## 👥 Collaborative work in the SAME notebook (MANDATORY)

Since two people work **in the same notebook**, all shared parameters must be defined **once at the top** of the notebook.

👉 This prevents inconsistent results and unnecessary discussions.

---

## ⚙️ Global parameter block (TOP of the notebook)

This block must be the **first executable cell** and must **never be duplicated**.

```python
# =========================
# Global parameters
# =========================

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Metrics
SCORING_CLASSIFICATION = ["accuracy", "precision", "recall", "f1"]

# KNN
KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = "distance"

# Random Forest
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None

# Paths
METRICS_PATH = "models/metrics_summary.csv"
```

📌 **Rule**:
- No hard‑coded values inside models
- Any change must be done **only here**

---

## 🧩 Standard function: `train_and_log_model()`

All models **must be trained and evaluated using this function**.

```python
def train_and_log_model(
    model,
    model_name: str,
    X_train,
    y_train,
    X_test,
    y_test,
    metrics_path: str = "models/metrics_summary.csv"
):
    """Train a model, compute standard metrics, and log results."""

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        metrics_path,
        mode="a",
        header=not os.path.exists(metrics_path),
        index=False
    )

    return metrics
```

---

## 🚫 What NOT to do (real examples)

### ❌ Train on the test set (DATA LEAKAGE)

```python
model.fit(X_test, y_test)  # ❌
```

✔ Correct:
```python
model.fit(X_train, y_train)
```

---

### ❌ Define parameters inside the model

```python
RandomForestClassifier(n_estimators=100)  # ❌
```

✔ Correct:
```python
RandomForestClassifier(n_estimators=RF_N_ESTIMATORS)
```

---

### ❌ Change shared parameters without visibility

- Modifying `KNN_N_NEIGHBORS` locally
- Changing `TEST_SIZE` in a hidden cell

✔ All changes must be made in the global block.

---

## 🧠 Final reinforced rule

> **If a parameter is not defined in the global block, it does not exist.**
>
> **If two people get different results, the notebook is incorrectly structured.**

