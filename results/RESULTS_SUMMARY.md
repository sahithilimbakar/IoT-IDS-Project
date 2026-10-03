# Experimental Results Summary (auto-generated)

## Dataset
- Rows used: 300,000 (train 240,000 / unseen test 60,000)
- NORMAL: 8,526 | ATTACK: 291,474 | distinct labels: 34
- Features before selection: 39 | after selection: 5 (87.2% reduction)

## Selected features (ranked by Random Forest importance)
1. Number
2. ack_flag_number
3. HTTPS
4. Tot size
5. Rate

## Best model on unseen test data
| Feature set | Model | #Features | Accuracy | Precision | Recall | F1 | Train time (s) | Inference (us/sample) |
|---|---|---|---|---|---|---|---|---|
| All features | Random Forest | 39 | 0.9897 | 0.9947 | 0.9947 | 0.9947 | 6.10 | 1.90 |
| Selected features | Random Forest | 5 | 0.9828 | 0.9914 | 0.9909 | 0.9912 | 4.73 | 2.53 |

## All models: before vs after (F1-score)
| Model | F1 (all) | F1 (selected) | Change | Train speed-up | Inference speed-up |
|---|---|---|---|---|---|
| Decision Tree | 0.9933 | 0.9900 | -0.0033 | 3.34x | 1.15x |
| Random Forest | 0.9947 | 0.9912 | -0.0036 | 1.29x | 0.75x |
| Hist Gradient Boosting | 0.9913 | 0.9863 | -0.0051 | 1.53x | 1.34x |
| Logistic Regression | 0.9848 | 0.9829 | -0.0019 | 5.53x | 5.16x |

## Key sentence for the paper (fill from your real numbers)
Reducing the feature set from 39 to 5 features (87.2% fewer) changed the best model's F1-score from 0.9947 to 0.9912 (change -0.0035), while training time went from 6.1s to 4.7s.

## Files
- results/tables/*.csv  - all numeric tables
- results/figures/*.png - all graphs
- models/ - trained models and detector bundles