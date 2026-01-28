# 📘 Logistic Regression — Detailed Theory

## 📌 What is Logistic Regression?

**Logistic Regression** is a **supervised machine learning algorithm** used for **classification problems**, mainly **binary classification**.

Examples:

* Spam ❌ / Not Spam ✅
* Pass ❌ / Fail ✅
* Disease ❌ / No Disease ✅

Despite the name, it is used for **classification**, not regression.

---

## 🧠 Core Idea (Intuition)

Linear regression gives output from **–∞ to +∞**
But classification needs output **between 0 and 1**

So we use the **Sigmoid (Logistic) Function**.

---

## 1️⃣ Why Logistic Regression Exists

Linear Regression predicts **continuous values**:
[
y = wx + b
]

Problem:

* Output can be **< 0 or > 1**
* Cannot represent **probability**

But classification problems need:
[
0 \le P(y=1|x) \le 1
]

👉 **Solution:** Pass linear output through a **sigmoid function**.

---

## 2️⃣ Sigmoid (Logistic) Function — Heart of the Model

[
\sigma(z) = \frac{1}{1 + e^{-z}}
]

### Properties:

* Output range: **(0, 1)**
* S-shaped curve
* Differentiable (important for gradient descent)

### Behavior:

* Large positive `z` → output ≈ 1
* Large negative `z` → output ≈ 0
* `z = 0` → output = 0.5

---

## 3️⃣ Logistic Regression Model Equation

### Step 1: Linear combination

[
z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
]

### Step 2: Apply sigmoid

[
\hat{y} = \sigma(z)
]

Here:

* `ŷ` = predicted probability that class = 1
* `w` = weights
* `b` = bias

---

## 4️⃣ Probabilistic Interpretation (Very Important)

Logistic regression models:
[
P(y=1|x) = \sigma(z)
]

And:
[
P(y=0|x) = 1 - \sigma(z)
]

So output is **probability**, not just class label.

---

## 5️⃣ Decision Boundary

The decision boundary occurs when:
[
P(y=1|x) = 0.5
]

Which means:
[
z = 0
\Rightarrow wx + b = 0
]

👉 This forms a **linear decision boundary** (line / plane / hyperplane).

---

## 6️⃣ Why NOT Mean Squared Error (MSE)?

Using MSE with sigmoid:

* Cost function becomes **non-convex**
* Multiple local minima
* Slow & unstable training

👉 Hence, **Log Loss** is used.

---

## 7️⃣ Cost Function (Log Loss / Binary Cross Entropy)

[
J(w) = -\frac{1}{m} \sum_{i=1}^{m}
\Big[y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\Big]
]

### Intuition:

* Strong penalty for confident wrong predictions
* Convex function → single global minimum

---

# 📂 File-Structure GitHub repo

Create file:

```
machine-learning-journey/
├── logistic_regression_sklearn.py
├── logistic_regression_from_scratch.py
└── logestic-regression-notes.md

```

## 8️⃣ Training Logistic Regression

### Optimization Algorithm:

* **Gradient Descent**
* **Stochastic Gradient Descent**
* **Newton’s Method** (used internally by some solvers)

### Weight Update Rule:

[
w = w - \alpha \frac{\partial J}{\partial w}
]

Where:

* `α` = learning rate
* `∂J/∂w` = gradient

---

## 9️⃣ Regularization (Overfitting Control)

### Why needed?

* High-dimensional data
* Noisy features
* Prevents large weights

---

### 🔹 L2 Regularization (Ridge)

[
J = \text{Log Loss} + \lambda \sum w^2
]

* Shrinks weights
* Keeps all features

---

### 🔹 L1 Regularization (Lasso)

[
J = \text{Log Loss} + \lambda \sum |w|
]

* Forces some weights to **zero**
* Feature selection

---

## 🔟 Multiclass Logistic Regression

### One-Vs-Rest (OvR)

* Train `k` binary classifiers
* Choose class with highest probability

### Softmax Regression (Multinomial)

[
P(y=j|x) = \frac{e^{z_j}}{\sum e^{z_k}}
]

Used when:

* More than 2 classes
* Classes are mutually exclusive

---

## 1️⃣1️⃣ Assumptions of Logistic Regression

✔ Binary dependent variable
✔ Independent observations
✔ No multicollinearity
✔ Linear relationship between features and log-odds

---

## 1️⃣2️⃣ Log-Odds (Advanced Theory – Interview Favorite)

[
\log\left(\frac{P(y=1)}{P(y=0)}\right) = wx + b
]

This means:

* Logistic regression is **linear in log-odds**
* Explains why it’s interpretable

---

## 1️⃣3️⃣ Evaluation Metrics (Important)

Accuracy is not enough ❌

| Metric    | Use                    |
| --------- | ---------------------- |
| Precision | False positives matter |
| Recall    | False negatives matter |
| F1-Score  | Balance                |
| ROC-AUC   | Probability ranking    |

---

## 1️⃣4️⃣ Advantages & Limitations (Theory Answer)

### Advantages

✔ Simple
✔ Fast
✔ Interpretable
✔ Probabilistic output

### Limitations

❌ Linear boundary
❌ Sensitive to outliers
❌ Needs feature scaling

---

## 📌 One-Line Exam Definition

> **Logistic Regression is a supervised classification algorithm that uses the sigmoid function to model the probability of a binary outcome and is trained using log-loss.**

---
