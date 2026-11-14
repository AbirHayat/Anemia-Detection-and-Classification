Macro-F1, development CV (25 outer folds, mean ± std). Original = pre-registered calibration (scikit-learn uses raw decision scores when a model has them). Amended = post-results amendment of 14 Nov 2026: sigmoid calibration on predicted probabilities for every model. Same folds, seeds and hyperparameters. Sources: results/phase3/summary_metrics_cal.csv, summary_metrics_un.csv and results/phase3/amended/summary_metrics_cal.csv.

| Feature set | Model | Arm | Calibrated on (original) | Uncalibrated | Calibrated, ORIGINAL (pre-registered) | Calibrated, AMENDED (on probabilities) | Change |
|---|---|---|---|---|---|---|---|
| F-all | dummy | none | probabilities | 0.047 ± 0.000 | 0.047 ± 0.000 | 0.047 ± 0.000 | +0.000 |
| F-all | forest | smote | probabilities | 0.960 ± 0.033 | 0.954 ± 0.039 | 0.954 ± 0.039 | +0.000 |
| F-all | forest | weights | probabilities | 0.949 ± 0.037 | 0.947 ± 0.042 | 0.947 ± 0.042 | +0.000 |
| F-all | hgb | smote | raw scores | 0.944 ± 0.040 | 0.947 ± 0.038 | 0.941 ± 0.038 | -0.005 |
| F-all | hgb | weights | raw scores | 0.942 ± 0.040 | 0.933 ± 0.044 | 0.941 ± 0.040 | +0.008 |
| F-all | logreg | smote | raw scores | 0.828 ± 0.057 | 0.525 ± 0.053 | 0.836 ± 0.058 | +0.311 |
| F-all | logreg | weights | raw scores | 0.826 ± 0.049 | 0.514 ± 0.047 | 0.831 ± 0.052 | +0.317 |
| F-all | tree | smote | probabilities | 0.969 ± 0.028 | 0.969 ± 0.028 | 0.969 ± 0.028 | +0.000 |
| F-all | tree | weights | probabilities | 0.972 ± 0.028 | 0.972 ± 0.028 | 0.972 ± 0.028 | +0.000 |
| F-all+idx | dummy | none | probabilities | 0.047 ± 0.000 | 0.047 ± 0.000 | 0.047 ± 0.000 | +0.000 |
| F-all+idx | forest | smote | probabilities | 0.956 ± 0.037 | 0.948 ± 0.044 | 0.948 ± 0.044 | +0.000 |
| F-all+idx | forest | weights | probabilities | 0.956 ± 0.036 | 0.954 ± 0.046 | 0.954 ± 0.046 | +0.000 |
| F-all+idx | hgb | smote | raw scores | 0.950 ± 0.038 | 0.948 ± 0.042 | 0.947 ± 0.042 | -0.002 |
| F-all+idx | hgb | weights | raw scores | 0.938 ± 0.032 | 0.939 ± 0.036 | 0.939 ± 0.034 | +0.000 |
| F-all+idx | logreg | smote | raw scores | 0.839 ± 0.052 | 0.522 ± 0.062 | 0.843 ± 0.052 | +0.321 |
| F-all+idx | logreg | weights | raw scores | 0.838 ± 0.042 | 0.502 ± 0.044 | 0.845 ± 0.045 | +0.343 |
| F-all+idx | tree | smote | probabilities | 0.967 ± 0.023 | 0.967 ± 0.022 | 0.967 ± 0.022 | +0.000 |
| F-all+idx | tree | weights | probabilities | 0.969 ± 0.029 | 0.969 ± 0.029 | 0.969 ± 0.029 | +0.000 |
| F-core | dummy | none | probabilities | 0.047 ± 0.000 | 0.047 ± 0.000 | 0.047 ± 0.000 | +0.000 |
| F-core | forest | smote | probabilities | 0.977 ± 0.020 | 0.975 ± 0.022 | 0.975 ± 0.022 | +0.000 |
| F-core | forest | weights | probabilities | 0.967 ± 0.026 | 0.968 ± 0.024 | 0.968 ± 0.024 | +0.000 |
| F-core | hgb | smote | raw scores | 0.945 ± 0.045 | 0.950 ± 0.040 | 0.945 ± 0.045 | -0.005 |
| F-core | hgb | weights | raw scores | 0.939 ± 0.035 | 0.934 ± 0.035 | 0.940 ± 0.035 | +0.005 |
| F-core | logreg | smote | raw scores | 0.865 ± 0.045 | 0.482 ± 0.053 | 0.863 ± 0.050 | +0.381 |
| F-core | logreg | weights | raw scores | 0.848 ± 0.037 | 0.490 ± 0.050 | 0.854 ± 0.043 | +0.364 |
| F-core | tree | smote | probabilities | 0.962 ± 0.026 | 0.962 ± 0.026 | 0.962 ± 0.026 | +0.000 |
| F-core | tree | weights | probabilities | 0.970 ± 0.029 | 0.970 ± 0.029 | 0.970 ± 0.029 | +0.000 |
| F-core+idx | dummy | none | probabilities | 0.047 ± 0.000 | 0.047 ± 0.000 | 0.047 ± 0.000 | +0.000 |
| F-core+idx | forest | smote | probabilities | 0.971 ± 0.026 | 0.967 ± 0.027 | 0.967 ± 0.027 | +0.000 |
| F-core+idx | forest | weights | probabilities | 0.973 ± 0.021 | 0.971 ± 0.024 | 0.971 ± 0.024 | +0.000 |
| F-core+idx | hgb | smote | raw scores | 0.949 ± 0.039 | 0.948 ± 0.040 | 0.950 ± 0.036 | +0.002 |
| F-core+idx | hgb | weights | raw scores | 0.942 ± 0.030 | 0.938 ± 0.033 | 0.944 ± 0.029 | +0.006 |
| F-core+idx | logreg | smote | raw scores | 0.875 ± 0.039 | 0.465 ± 0.054 | 0.872 ± 0.049 | +0.407 |
| F-core+idx | logreg | weights | raw scores | 0.868 ± 0.038 | 0.473 ± 0.047 | 0.869 ± 0.048 | +0.396 |
| F-core+idx | tree | smote | probabilities | 0.954 ± 0.033 | 0.954 ± 0.033 | 0.954 ± 0.033 | +0.000 |
| F-core+idx | tree | weights | probabilities | 0.968 ± 0.028 | 0.968 ± 0.028 | 0.968 ± 0.028 | +0.000 |

Pre-registered rules applied to each version:

- Imbalance arm (P2), original: logreg weights; tree weights; forest smote; hgb smote
- Imbalance arm (P2), amended: logreg smote; tree weights; forest smote; hgb smote
- H2 model (P3), original: forest with smote
- H2 model (P3), amended: forest with smote
