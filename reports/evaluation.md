# Evaluation — my CalorieCNN on a clean session-grouped split

Test set: **650 dishes**, session-disjoint from training. Majority-class baseline: **33.8%**.

| Class | Precision | Recall (sens.) | Specificity | F1 | AUC | Support |
|---|---|---|---|---|---|---|
| Low | 0.854 | 0.768 | 0.933 | 0.809 | 0.954 | 220 |
| Medium | 0.588 | 0.668 | 0.784 | 0.626 | 0.827 | 205 |
| High | 0.813 | 0.791 | 0.904 | 0.802 | 0.939 | 225 |

**Accuracy:** 0.745 · **Macro-F1:** 0.745

**Off-by-two rate** (Low↔High, the maximal error): **0.3%** — low means the ordinal head keeps mistakes adjacent.

![confusion matrix](figures/confusion_matrix.png)

![ROC curves](figures/roc_curves.png)

*These numbers are from the ~990K-param CalorieCNN at 160px on the clean split — reproducible via this script, independent of the group's Kaggle weights. They are not the group's reported 74.1%.*
