import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from scipy.stats import norm
from lifelines import CoxPHFitter, KaplanMeierFitter
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# ----- read data -----
train_df = pd.read_csv("./data/splits/train_set.csv")
val_df = pd.read_csv("./data/splits/val_set.csv")

os.makedirs("./results/", exist_ok=True)

print("=" * 60)
print("Data loaded")
print("=" * 60)
print(f"Training: {train_df.shape}")
print(f"Validation: {val_df.shape}")

# ----- features -----
features = ['Age', 'BUN', 'GPR', 'RDW', 'Sepsis', 'RR', 'Platelets']

for feat in features:
    if train_df[feat].isna().sum() > 0:
        med_val = train_df[feat].median()
        train_df[feat] = train_df[feat].fillna(med_val)
        val_df[feat] = val_df[feat].fillna(med_val)

# ----- prepare survival data -----
train_df['time'] = train_df['icu_survival_day_365']
train_df['status'] = train_df['death_within_icu_365days']

# ----- fit Cox models -----
cph1 = CoxPHFitter()
cph1.fit(train_df[['time', 'status'] + features], duration_col='time', event_col='status')

cph2 = CoxPHFitter()
cph2.fit(train_df[['time', 'status', 'SOFA']], duration_col='time', event_col='status')

cph3 = CoxPHFitter()
cph3.fit(train_df[['time', 'status', 'Charlson']], duration_col='time', event_col='status')

cph4 = CoxPHFitter()
cph4.fit(train_df[['time', 'status', 'SOFA', 'Charlson']], duration_col='time', event_col='status')

# ----- baseline survival -----
kmf = KaplanMeierFitter()
kmf.fit(train_df['time'], train_df['status'])

base_surv_30 = kmf.survival_function_at_times(30).values[0] if 30 in kmf.survival_function_.index else 0.5
base_surv_90 = kmf.survival_function_at_times(90).values[0] if 90 in kmf.survival_function_.index else 0.5
base_surv_365 = kmf.survival_function_at_times(365).values[0] if 365 in kmf.survival_function_.index else 0.5

def calc_pred(cph, newdata, base_surv):
    lp = cph.predict_partial_hazard(newdata)
    return 1 - base_surv ** lp.values

# ----- predictions on validation set -----
val_7f = val_df[features]
val_sofa = val_df[['SOFA']]
val_charlson = val_df[['Charlson']]
val_sofa_charlson = val_df[['SOFA', 'Charlson']]

pred1_30 = calc_pred(cph1, val_7f, base_surv_30)
pred1_90 = calc_pred(cph1, val_7f, base_surv_90)
pred1_365 = calc_pred(cph1, val_7f, base_surv_365)

pred2_30 = calc_pred(cph2, val_sofa, base_surv_30)
pred2_90 = calc_pred(cph2, val_sofa, base_surv_90)
pred2_365 = calc_pred(cph2, val_sofa, base_surv_365)

pred3_30 = calc_pred(cph3, val_charlson, base_surv_30)
pred3_90 = calc_pred(cph3, val_charlson, base_surv_90)
pred3_365 = calc_pred(cph3, val_charlson, base_surv_365)

pred4_30 = calc_pred(cph4, val_sofa_charlson, base_surv_30)
pred4_90 = calc_pred(cph4, val_sofa_charlson, base_surv_90)
pred4_365 = calc_pred(cph4, val_sofa_charlson, base_surv_365)

# ----- outcomes -----
y_30 = ((val_df['icu_survival_day_30'] <= 30) & (val_df['death_within_icu_30days'] == 1)).astype(int)
y_90 = ((val_df['icu_survival_day_90'] <= 90) & (val_df['death_within_icu_90days'] == 1)).astype(int)
y_365 = val_df['death_within_icu_365days'].astype(int)

def safe_auc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_pred)

def safe_brier(y_true, y_pred):
    try:
        return brier_score_loss(y_true, y_pred)
    except:
        return np.nan

auc_30 = [safe_auc(y_30, pred1_30), safe_auc(y_30, pred2_30),
          safe_auc(y_30, pred3_30), safe_auc(y_30, pred4_30)]
auc_90 = [safe_auc(y_90, pred1_90), safe_auc(y_90, pred2_90),
          safe_auc(y_90, pred3_90), safe_auc(y_90, pred4_90)]
auc_365 = [safe_auc(y_365, pred1_365), safe_auc(y_365, pred2_365),
           safe_auc(y_365, pred3_365), safe_auc(y_365, pred4_365)]

brier_30 = [safe_brier(y_30, pred1_30), safe_brier(y_30, pred2_30),
            safe_brier(y_30, pred3_30), safe_brier(y_30, pred4_30)]
brier_90 = [safe_brier(y_90, pred1_90), safe_brier(y_90, pred2_90),
            safe_brier(y_90, pred3_90), safe_brier(y_90, pred4_90)]
brier_365 = [safe_brier(y_365, pred1_365), safe_brier(y_365, pred2_365),
             safe_brier(y_365, pred3_365), safe_brier(y_365, pred4_365)]

model_names = ["GPR‑based nomogram", "SOFA", "Charlson", "SOFA+Charlson"]

# ----- DeLong test -----
def delong_bootstrap(y_true, pred1, pred2, n_bootstrap=2000, random_state=42):
    np.random.seed(random_state)
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        y_boot = y_true.iloc[idx] if hasattr(y_true, 'iloc') else y_true[idx]
        p1_boot = pred1[idx]
        p2_boot = pred2[idx]
        try:
            auc1 = roc_auc_score(y_boot, p1_boot)
            auc2 = roc_auc_score(y_boot, p2_boot)
            diffs.append(auc1 - auc2)
        except:
            continue
    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    p_value = 2 * (1 - norm.cdf(np.abs(mean_diff) / np.std(diffs)))
    return mean_diff, ci_lower, ci_upper, p_value

diff_30, ci_lower_30, ci_upper_30, p_30 = delong_bootstrap(y_30, pred1_30, pred4_30)
diff_90, ci_lower_90, ci_upper_90, p_90 = delong_bootstrap(y_90, pred1_90, pred4_90)
diff_365, ci_lower_365, ci_upper_365, p_365 = delong_bootstrap(y_365, pred1_365, pred4_365)

print("\nDeLong test: GPR-based vs SOFA+Charlson")
print(f"30-day: diff={diff_30:.3f}, p={p_30:.4f}")
print(f"90-day: diff={diff_90:.3f}, p={p_90:.4f}")
print(f"365-day: diff={diff_365:.3f}, p={p_365:.4f}")

# ----- plot ROC -----
def plot_roc(y_true, pred_list, auc_list, time_label, p_value, output_path):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    plt.figure(figsize=(10, 8))
    for i, (pred, auc_val) in enumerate(zip(pred_list, auc_list)):
        if np.isnan(auc_val):
            continue
        fpr, tpr, _ = roc_curve(y_true, pred)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{model_names[i]} (AUC={auc_val:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    if not np.isnan(p_value):
        plt.plot([], [], ' ', label=f'DeLong: GPR-based vs SOFA+Charlson: P = {p_value:.3f}')
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.title(f'Validation Set - {time_label} Mortality', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='tiff')
    plt.close()

# ----- plot calibration -----
def plot_calibration(y_true, pred_list, brier_list, time_label, output_path):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    plt.figure(figsize=(10, 8))
    for i, (pred, brier) in enumerate(zip(pred_list, brier_list)):
        if np.isnan(pred).all() or np.isnan(brier):
            continue
        try:
            frac_pos, mean_pred = calibration_curve(y_true, pred, n_bins=10, strategy='uniform')
            plt.plot(mean_pred, frac_pos, 'o-', color=colors[i], lw=2, markersize=8,
                     label=f'{model_names[i]} (Brier={brier:.3f})')
        except:
            continue
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect calibration')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Observed Proportion of Deaths', fontsize=12)
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.title(f'Calibration Curves - Validation Set ({time_label} Mortality)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='tiff')
    plt.close()

# ----- generate figures -----
print("\nGenerating figures...")

plot_roc(y_30, [pred1_30, pred2_30, pred3_30, pred4_30], auc_30, "30-day", p_30, "./results/ROC_Val_30day.tiff")
plot_calibration(y_30, [pred1_30, pred2_30, pred3_30, pred4_30], brier_30, "30-day", "./results/Calibration_Val_30day.tiff")

plot_roc(y_90, [pred1_90, pred2_90, pred3_90, pred4_90], auc_90, "90-day", p_90, "./results/ROC_Val_90day.tiff")
plot_calibration(y_90, [pred1_90, pred2_90, pred3_90, pred4_90], brier_90, "90-day", "./results/Calibration_Val_90day.tiff")

plot_roc(y_365, [pred1_365, pred2_365, pred3_365, pred4_365], auc_365, "365-day", p_365, "./results/ROC_Val_365day.tiff")
plot_calibration(y_365, [pred1_365, pred2_365, pred3_365, pred4_365], brier_365, "365-day", "./results/Calibration_Val_365day.tiff")

# ----- save results -----
auc_df = pd.DataFrame({
    'Model': model_names,
    'AUC_30d': auc_30,
    'AUC_90d': auc_90,
    'AUC_365d': auc_365
})
auc_df.to_csv("./results/AUC_Validation_Set.csv", index=False)

brier_df = pd.DataFrame({
    'Model': model_names,
    'Brier_30d': brier_30,
    'Brier_90d': brier_90,
    'Brier_365d': brier_365
})
brier_df.to_csv("./results/Brier_Validation_Set.csv", index=False)

delong_df = pd.DataFrame({
    'Timepoint': ['30-day', '90-day', '365-day'],
    'GPR_based_AUC': [auc_30[0], auc_90[0], auc_365[0]],
    'SOFA_Charlson_AUC': [auc_30[3], auc_90[3], auc_365[3]],
    'Difference': [diff_30, diff_90, diff_365],
    'CI_lower': [ci_lower_30, ci_lower_90, ci_lower_365],
    'CI_upper': [ci_upper_30, ci_upper_90, ci_upper_365],
    'P_value': [p_30, p_90, p_365]
})
delong_df.to_csv("./results/DeLong_GPRbased_vs_SOFACharlson.csv", index=False)

print("\nAll done!")
print("ROC_Val_30day.tiff, ROC_Val_90day.tiff, ROC_Val_365day.tiff")
print("Calibration_Val_30day.tiff, Calibration_Val_90day.tiff, Calibration_Val_365day.tiff")