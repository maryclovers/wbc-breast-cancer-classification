# Breast cancer classification: a full machine learning workflow

AI as a Medical Device module, MSc AI for Healthcare, University of Hull, 2026. Module mark: 85%.

This is my laboratory notebook from the module. It takes the Wisconsin Breast Cancer dataset through every stage of a clinical machine learning workflow, from data preparation to choosing an operating threshold. The tuned support vector machine (SVM) reached 0.94 cross-validated accuracy, and lowering its decision threshold to 0.3 caught every malignant tumour in the held-out split (sensitivity 1.0) at a specificity of 0.84.

## The dataset

569 breast masses described by 30 features computed from digitised fine needle aspirate images (radius, texture, concavity and similar measures), each labelled benign or malignant. The module's copy has missing values added on purpose, so handling them was part of the exercise. The final models in Session 09 use the 266 complete cases, split 212 for training and 54 for testing, stratified by class. In this setting a false negative is a missed cancer, so sensitivity is the metric I watched most closely.

## What the notebook covers

| Stage | What I did |
|---|---|
| Understanding the data | Distributions, class balance, correlation analysis, removal of highly correlated features |
| Preparation | Scaling, and three ways of handling class imbalance compared side by side: random oversampling, SMOTE and ADASYN |
| Feature selection | Variance threshold, univariate selection, recursive feature elimination (RFE) and principal component analysis (PCA), with the overlap between methods checked |
| Unsupervised learning | An autoencoder and clustering, to see whether benign and malignant cases separate without labels |
| Probability | Bayes' theorem in practice: how disease prevalence changes the positive predictive value of a screening test |
| Models | Naive Bayes, random forest, SVM, and a small PyTorch neural network (30 inputs, two hidden layers) |
| Validation and tuning | Cross-validation and GridSearchCV. Best SVM: linear kernel, C = 10, 0.94 cross-validated accuracy |
| Operating point | Decision threshold analysis. At a threshold of 0.3, sensitivity 1.0 and specificity 0.84 |
| Evidence | A structured literature search through PubMed to compare published AI breast cancer tools with these results |

## Why the threshold matters

The default threshold of 0.5 treats a missed cancer and a false alarm as equally costly. In screening they are not. Moving the threshold to 0.3 accepts more false positives, which lead to further investigation, in exchange for not missing malignant cases in this split. The test split is small, so this is evidence for the approach, not a claim of perfect sensitivity in practice.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook wbc_breast_cancer_classification.ipynb
```

The notebook reads the module's copy of the public [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17) dataset, which is not included here. The public version, also available through `sklearn.datasets.load_breast_cancer`, has no missing values, so results will differ slightly. The PubMed section needs an email address for the NCBI Entrez API: replace `your_email@example.com` with your own.

## What I learned

- Resampling belongs inside cross-validation, on the training folds only. In the Session 03 pipeline I oversampled before splitting, which lets synthetic copies of the same cases sit in both training and test sets and inflates the scores. The Session 09 models avoid this by working on the original data.
- Different feature selection methods agree on only part of the feature set, so the overlap is more trustworthy than any single ranking.
- A threshold is a clinical decision, not a default setting.

## Related

[tcga-glioma-grading](https://github.com/maryclovers/tcga-glioma-grading), the assessed project from the same module.

Clovers Akitoye · [GitHub](https://github.com/maryclovers) · [LinkedIn](https://www.linkedin.com/in/cloversakitoye)
