# AI-Based Intrusion Detection for IoT Networks

Pipeline: Dataset -> Data Preparation -> Machine Learning -> Attack Detection -> Testing -> Feature Selection

## Setup (VS Code terminal, inside the IoT_IDS_Project folder)
    pip install -r requirements.txt

## Folder layout
    IoT_IDS_Project/
      dataset/MERGED_CSV/*.csv     <- your data (already there)
      config.py  data_prep.py  train_models.py  detector.py
      evaluate.py  feature_selection.py  compare_results.py  run_all.py

## Run everything
    python run_all.py --quick      # 2-5 min sanity check first
    python run_all.py              # normal run (1% of every CSV)
    python run_all.py --frac 0.03 --max-rows 600000   # bigger, more accurate

## Outputs
    results/figures/*.png     graphs
    results/tables/*.csv      metrics, feature ranking, before/after comparison
    results/RESULTS_SUMMARY.md  numbers ready for the paper
    models/                   trained models + ids_detector_full / ids_detector_reduced

## Use the detector
    python detector.py --demo 10
    python detector.py --input new_traffic.csv --mode reduced

## Notes
- Test set is never used for training or for choosing features (fair before/after comparison).
- If the label column is not detected, set LABEL_CANDIDATES / NORMAL_NAMES in config.py.
- make_demo_data.py creates FAKE data only for testing the code. Never use it for the paper.
