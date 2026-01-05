# 📘 README – Guía para Modelos de Machine Learning

Este documento define **las reglas mínimas, estructura y buenas prácticas** para trabajar en el desarrollo de modelos de Machine Learning dentro del proyecto.

👉 Está diseñado para socios con **experiencia limitada**, aunque sigue **estándares ML reales y profesionales**.

---

## 🎯 Objetivo

- Garantizar que **todos los modelos sean comparables**
- Evitar errores comunes (data leakage, métricas inconsistentes, nombres confusos)
- Facilitar análisis, gráficos y toma de decisiones
- Ahorrar tiempo en revisiones

> **Regla clave:** si tu modelo no se puede comparar fácilmente con otro, está mal implementado.

---

## 📁 Estructura obligatoria del proyecto

```
project/
├── data/
│   ├── raw/            # Datos originales (no modificar)
│   ├── processed/      # Datos limpios
│   └── features/       # Datasets finales para modelos
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

📌 **Importante**
- No modificar `data/raw`
- Todas las métricas finales deben guardarse en `models/metrics_summary.csv`
- Todos los gráficos van en `outputs/plots/`

---

## 🔤 Convención de nombres (CRÍTICO)

### 🔹 Datasets y splits

Usar **siempre** estos nombres:

```python
X_train, X_test
y_train, y_test
```

Si hay validación:

```python
X_train, X_val, X_test
y_train, y_val, y_test
```

❌ Incorrecto:
```python
X1, X2, train, test, ytemp
```

---

### 🔹 Features

Los nombres deben indicar **qué contienen**, no cómo se crearon.

✅ Correcto:
```python
features_numeric
features_categorical
X_features_final
```

❌ Incorrecto:
```python
df2, temp, x
```

---

## 🧠 Convención de nombres de modelos

Cada modelo debe tener un nombre único y descriptivo.

```python
model_name = "logreg_v1_baseline"
model_name = "rf_v2_class_weight"
model_name = "xgb_v3_feature_eng"
```

### 📌 Formato recomendado

```
<modelo>_<versión>_<detalle_clave>
```

Ejemplos:
- `logreg_v1_baseline`
- `rf_v1_no_balance`
- `xgb_v2_tuned`

---

## 🏋️ Entrenamiento del modelo (patrón único)

Todos los modelos deben seguir exactamente este flujo:

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

🚫 Prohibido:
- Entrenar con `X_test`
- Ajustar hiperparámetros usando `test`
- Evaluar sin aclarar el set usado

---

## 📊 Métricas (estándar obligatorio)

Todas las métricas deben guardarse con el mismo formato.

```python
metrics = {
    "model": model_name,
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}
```

📌 Si una métrica no está aquí, **no se compara ni se grafica**.

---

## 💾 Guardado de métricas (OBLIGATORIO)

Todas las ejecuciones deben agregarse al archivo común:

```python
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(
    "models/metrics_summary.csv",
    mode="a",
    header=False,
    index=False
)
```

✔ Esto permite:
- Comparar modelos
- Generar gráficos automáticos
- Versionar resultados

---

## 📈 Gráficos

### Reglas

- Un gráfico por métrica
- Eje X = modelo
- No mezclar métricas

Ejemplo:

```python
sns.barplot(
    data=metrics_df,
    x="model",
    y="f1"
)
plt.xticks(rotation=45)
```

---

## ⚠️ Errores comunes y cómo evitarlos

### ❌ Cambiar nombres entre modelos

```python
Xtrain, X_test2
```

✔ Usar siempre nombres estándar.

---

### ❌ Sobrescribir predicciones

```python
y_pred = model1.predict()
y_pred = model2.predict()
```

✔ Soluciones:
- Guardar métricas directamente
- O usar nombres distintos (`y_pred_lr`, `y_pred_rf`)

---

### ❌ Resultados no reproducibles

✔ Siempre fijar seed:

```python
RANDOM_STATE = 42
```

---

## ✅ Checklist antes de entregar un modelo

Antes de subir tu trabajo, verificá:

- [ ] Usé `X_train / X_test`
- [ ] Definí `model_name`
- [ ] Guardé métricas en `metrics_summary.csv`
- [ ] No entrené con test
- [ ] Fijé `random_state`
- [ ] Los gráficos son claros y legibles

---

## 🧠 Regla final

> **Si otra persona no puede entender, reproducir y comparar tu modelo en menos de 30 segundos, hay que corregirlo.**

---

📌 Ante dudas, **no improvisar**: preguntar antes de cambiar la estructura o las convenciones.

---

## 🧩 Función estándar: `train_and_log_model()`

Para evitar errores y asegurar consistencia, **todos los modelos deben entrenarse y evaluarse usando esta función**.

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
    """
    Entrena un modelo, evalúa métricas estándar y las guarda para comparación.
    """
    # Entrenamiento
    model.fit(X_train, y_train)

    # Predicción
    y_pred = model.predict(X_test)

    # Métricas estándar
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    # Guardado
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        metrics_path,
        mode="a",
        header=not os.path.exists(metrics_path),
        index=False
    )

    return metrics
```

### ✔ Uso correcto

```python
metrics_lr = train_and_log_model(
    model=logreg,
    model_name="logreg_v1_baseline",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)
```

📌 Beneficios:
- Evita data leakage
- Evita métricas inconsistentes
- Facilita gráficos automáticos
- Permite comparar modelos sin esfuerzo

---

## 🚫 Qué NO hacer (errores reales y comunes)

### ❌ Entrenar con test (DATA LEAKAGE)

```python
model.fit(X_test, y_test)  # ❌
```

✔ Correcto:
```python
model.fit(X_train, y_train)
```

---

### ❌ Cambiar nombres entre notebooks

```python
Xtrain, ytrain
X_test_final
```

✔ Correcto:
```python
X_train, y_train
X_test, y_test
```

---

### ❌ Comparar modelos con métricas distintas

```python
accuracy_lr = accuracy_score(...)
f1_rf = f1_score(...)
```

✔ Correcto: todos los modelos deben reportar las **mismas métricas** usando la función estándar.

---

### ❌ Sobrescribir resultados

```python
y_pred = model1.predict()
y_pred = model2.predict()
```

✔ Correcto:
- No guardar predicciones
- Guardar métricas directamente

---

### ❌ No fijar `random_state`

```python
RandomForestClassifier()
```

✔ Correcto:
```python
RandomForestClassifier(random_state=42)
```

---

### ❌ Gráficos sin contexto

```python
plt.plot(values)
```

✔ Correcto:
- Eje X = modelo
- Título con métrica
- Labels claros

---

## 👥 Trabajo colaborativo en el MISMO notebook (OBLIGATORIO)

Como el trabajo se realiza **entre dos personas en un mismo notebook**, es obligatorio usar **parámetros compartidos**, definidos una sola vez al inicio.

👉 Esto evita resultados inconsistentes y discusiones innecesarias.

---

## ⚙️ Bloque único de parámetros (AL INICIO DEL NOTEBOOK)

Este bloque debe estar en la **primera celda ejecutable** del notebook y **NO debe duplicarse**.

```python
# =========================
# Parámetros globales
# =========================

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Métricas
SCORING_REGRESSION = ["rmse", "mae", "r2"]
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

📌 **Regla**:
- Nadie hardcodea valores dentro del modelo
- Si se cambia un parámetro, se cambia **solo acá**

---

## 🤝 Reglas de convivencia (MUY IMPORTANTES)

### 👤 Partner A – Regresión

- Modelos de regresión (Linear / Ridge / Lasso, etc.)
- Usa **exactamente** los parámetros globales
- No redefine `random_state`, `test_size` ni métricas

Ejemplo:

```python
from sklearn.linear_model import LinearRegression

reg_model = LinearRegression()

metrics_reg = train_and_log_model(
    model=reg_model,
    model_name="linreg_v1_baseline",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    metrics_path=METRICS_PATH
)
```

---

### 👤 Partner B – KNN y Random Forest

- Modelos KNN y Random Forest
- Usa **solo** parámetros del bloque global

Ejemplo KNN:

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=KNN_N_NEIGHBORS,
    weights=KNN_WEIGHTS
)
```

Ejemplo Random Forest:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    random_state=RANDOM_STATE
)
```

---

## 🚫 Qué NO hacer en trabajo colaborativo

### ❌ Definir parámetros dentro del modelo

```python
RandomForestClassifier(n_estimators=100)  # ❌
```

✔ Correcto:
```python
RandomForestClassifier(n_estimators=RF_N_ESTIMATORS)
```

---

### ❌ Cambiar parámetros sin avisar

- Cambiar `KNN_N_NEIGHBORS` sin comunicarlo
- Ajustar `TEST_SIZE` localmente

✔ Todo cambio debe hacerse en el bloque global y quedar visible.

---

### ❌ Duplicar celdas de entrenamiento

- Una persona no re-entrena el modelo de la otra
- Cada modelo se ejecuta **una sola vez** y se loguea

---

## 🧠 Regla final (reforzada)

> **Si un parámetro no está definido en el bloque global, no existe.**
>
> **Si dos personas obtienen resultados distintos, el notebook está mal estructurado.**

