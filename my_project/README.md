# GPR-based Mortality Prediction in AF Patients

## Data Source
- MIMIC-IV (PhysioNet, https://physionet.org)
- NWICU (PhysioNet, https://physionet.org)
- Users must obtain data use agreements from PhysioNet

## Requirements
- Python 3.13.9 (see requirements.txt)
- R 4.5.2

## Workflow
1. Run R scripts in `R_scripts/` sequentially (01 → 03 → 05)
2. Run Python scripts in `python_scripts/` sequentially (02 → 04 → 06)

## Reproducibility
All random seeds are fixed. Raw data not included.