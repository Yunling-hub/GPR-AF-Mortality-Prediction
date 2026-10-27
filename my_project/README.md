# GPR-based Mortality Prediction in AF Patients

## Data Source
- MIMIC-IV (PhysioNet, https://physionet.org)
- NWICU (PhysioNet, https://physionet.org)
- Users must obtain data use agreements from PhysioNet

## Requirements
- Python 3.13.9 (see requirements.txt)
- R 4.5.2

## Workflow
1. R_scripts/01_data_preprocessing_MIMIC_traditional_statistical_analysis.R
2. R_scripts/01_data_preprocessing_NWICU_traditional_statistical_analysis.R
3. python_scripts/02_original_dataset_split.py
4. R_scripts/03_data_preprocessing_machine_learning_analysis.R
5. R_scripts/04_feature_selection.R
6. python_scripts/05_model_training_validation_30d.py
7. python_scripts/05_model_training_validation_90d.py
8. python_scripts/05_model_training_validation_365d.py
9. R_scripts/06_nomogram.R
10. python_scripts/07_nomogram_model_comparison_ROC_Calibration.py

## Reproducibility
- All random seeds are fixed.
- For ML analyses, winsorization and MICE imputation were fitted on the
  training set only and applied unchanged to the internal validation set
  and the NWICU cohort.
- Raw data are not included.