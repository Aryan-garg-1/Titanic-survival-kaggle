# Titanic Survival Prediction 🚢

Kaggle competition entry predicting passenger survival using feature engineering and ensemble models.  
**Best CV accuracy: 83.7% (XGBoost)**

---

## Model Comparison

| Model               | Mean CV Accuracy |
|---------------------|-----------------|
| Logistic Regression | 81.5%           |
| Random Forest       | 81.9%           |
| XGBoost             | **83.7%**       |

---

## Evaluation (Validation Fold)

|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Not Survived  | 0.85      | 0.90   | 0.87     | 110     |
| Survived      | 0.82      | 0.74   | 0.78     | 69      |
| **Accuracy**  |           |        | **0.84** | 179     |

---

## Feature Engineering Highlights

- **Title extraction** — parsed from passenger names, rare titles grouped into one category
- **HasCabin** — binary flag for whether a cabin was recorded (correlates with class & survival)
- **FamilySize & IsAlone** — derived from SibSp + Parch
- **Grouped Age imputation** — median filled within Pclass × Sex groups instead of a global median
- **LogFare** — log1p transform to reduce Fare skew
- **One-hot Embarked** — avoids false ordinal assumption (S=0, C=1, Q=2)
- **AgeBand** — age bucketed into 5 life-stage bins

---

## Notebook Structure

1. Data loading & missingness overview
2. Exploratory Data Analysis (survival by Sex, Pclass, Embarked, Age, Fare, SibSp + correlation heatmap)
3. Feature engineering
4. Model training & cross-validation comparison (Logistic Regression, Random Forest, XGBoost)
5. Confusion matrix & classification report on held-out validation fold
6. Submission generation

---

## Stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `matplotlib` · `seaborn`
