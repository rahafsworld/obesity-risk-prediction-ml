# Obesity Classification Using Machine Learning

Predicting obesity levels based on eating habits and physical activity. Group project comparing 5 ML models (KNN, Decision Tree, SVM, Random Forest, LightGBM).

## Dataset

**Source:** [UCI ML Repository - Obesity Levels](https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)

- **2,111 samples** → 2,087 after removing duplicates
- **16 features** + engineered BMI
- **7 obesity classes** (Insufficient Weight to Obesity Type III)

## Critical Data Limitation: 77% Synthetic

**This dataset is mostly artificial (77% SMOTE-generated), so it creates unrealistic patterns:**

### Synthetic Data Artifacts:
1. **Gender shows artificially strong correlation** (Cramér's V = 0.558)
   - Males concentrated in Obesity Type II (295/323 samples)
   - Females concentrated in Obesity Type III (323/324 samples)
   - This gender split is **an artifact of SMOTE balancing** so it obviously doesn't reflect real-world obesity patterns

2. **Family history appears in 81.8% of samples**
   - Likely overrepresented due to SMOTE targeting this feature for class balancing
   - Shows 26.9x risk amplification that may be exaggerated

3. **Sequential clustering in raw data**
   - Samples aren't randomly distributed
   - Gender and obesity categories appear in grouped sequences
   - Suggests synthetic generation process rather than natural data collection

### Why This Matters Here:
- Models may learn **synthetic patterns instead of real medical relationships**
- Gender-based predictions should be interpreted cautiously
- Feature importance rankings may be distorted
- Generalization to real patients is uncertain

**Our approach:** For the sake of research, we acknowledge these limitations throughout the analysis and focus on methodology rather than claiming clinical validity.

---

## What We Did

### 1. Exploratory Data Analysis
Despite synthetic artifacts, found patterns:
- **Weight** (46.7% importance) and **Height** (26.2%) dominate predictions
- **Physical activity** (FAF) inversely correlates with obesity (mean: 1.25 → 0.64)
- **Vegetable consumption** (FCVC) shows moderate association (Cramér's V = 0.486)

### 2. Data Preparation
- Removed 24 duplicates
- Engineered BMI feature
- One-hot encoded categorical variables (8 features)
- Standardized numeric features (9 features)
- 80-20 stratified split

### 3. Model Comparison

| Model | Test Accuracy | Strengths | Weaknesses |
|-------|--------------|-----------|------------|
| **LightGBM** | **98%** | Highest accuracy, fast | Black box, may overfit synthetic patterns |
| SVM | 96% | Strong with high dimensions | Hard to interpret |
| Random Forest | 94% | Balanced, interpretable | Slower training |
| Decision Tree | 94% | Fully transparent | Overfits (100% train acc) |
| KNN | 86% | Simple, explainable | Struggles with overlap |

**Deployed:** Random Forest (chosen for better stability on small datasets despite lower accuracy than LightGBM. Not available in this repository)

---

## Key Findings

### Feature Importance Consensus:
1. Weight / BMI (dominates all models)
2. Height
3. Gender *likely inflated due to synthetic data*
4. Age
5. Physical Activity (FAF)

### Model Behavior:
- All models struggle with **adjacent classes** (e.g., Overweight Level I vs II)
- Extreme categories easier to classify (Insufficient Weight, Obesity Type III)
- Misclassifications rarely cross distant categories

---

## Tech Stack

Anaconda, Python, scikit-learn, LightGBM, pandas, matplotlib, seaborn

---

## Limitations

**Dataset Issues:**
- 77% synthetic data creates unrealistic correlations
- Small sample size (2,087 records)
- Self-reported behavioral data (subject to bias)
- Sequential clustering suggests non-random generation

**Model Issues:**
- May have learned synthetic artifacts rather than real patterns
- Gender importance likely overstated
- Limited generalizability to real clinical populations

---

## What We Learned

1. **Synthetic data detection matters** - always check for sequential patterns and unrealistic correlations
2. **Accuracy isn't everything** - deployed Random Forest (94%) over LightGBM (98%) for stability
3. **Interpretability vs performance tradeoff** - Decision Tree (transparent) vs LightGBM (accurate)
4. **Evaluation needs multiple metrics** - accuracy, precision, recall, F1, confusion matrices all tell different stories

---

## Future Improvements

- Test on **real clinical datasets** to validate findings
- Apply **cross-validation** instead of single train-test split
- Use **SHAP values** to explain black-box model predictions
- Address class imbalance without synthetic oversampling
- Investigate why gender correlation is so strong in synthetic data

---

**Academic Context:** University group project for Data Science course. Focus on methodology and model comparison rather than clinical deployment. In the code file, I only included MY work, and not my peer's.

Feedback welcome!
