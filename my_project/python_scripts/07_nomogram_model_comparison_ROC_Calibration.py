# ============================================================================
# ROC and calibration curves - validation set (one figure per time point)
# Three models: GPR-based nomogram vs SOFA vs Charlson
# ============================================================================

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

# ============================================================================
# 1. Load data
# ============================================================================
train_path = 'data/processed/ML/train_processed.csv'
val_path   = 'data/processed/ML/val_processed.csv'
output_dir = 'results/nomogram'

os.makedirs(output_dir, exist_ok=True)

train_df = pd.read_csv(train_path)
val_df   = pd.read_csv(val_path)

print("=" * 60)
print("Data loaded")
print("=" * 60)
print(f"Training set: {train_df.shape}")
print(f"Validation set: {val_df.shape}")

# ============================================================================
# 2. Define variables (9 features)
# ============================================================================
features = ['Age', 'RDW', 'Sepsis', 'ARF', 'RR', 'GPR', 'Anion_gap', 'Vasopressor', 'HR']

for feat in features:
    if train_df[feat].isna().sum() > 0:
        med_val = train_df[feat].median()
        train_df[feat] = train_df[feat].fillna(med_val)
        val_df[feat]   = val_df[feat].fillna(med_val)

print("\nMissing value handling complete")

# ============================================================================
# 3. Prepare survival data for the training set
# ============================================================================
train_df['time']   = train_df['icu_survival_day_365']
train_df['status'] = train_df['death_within_icu_365days']

# ============================================================================
# 4. Fit Cox proportional hazards models (three models)
# ============================================================================
cph1 = CoxPHFitter()
cph1.fit(train_df[['time', 'status'] + features], duration_col='time', event_col='status')

cph2 = CoxPHFitter()
cph2.fit(train_df[['time', 'status', 'SOFA']], duration_col='time', event_col='status')

cph3 = CoxPHFitter()
cph3.fit(train_df[['time', 'status', 'Charlson']], duration_col='time', event_col='status')

print("\nThree Cox models fitted (GPR-based nomogram, SOFA, Charlson)")

# ============================================================================
# 5. Baseline survival
# ============================================================================
kmf = KaplanMeierFitter()
kmf.fit(train_df['time'], train_df['status'])

base_surv_30  = kmf.survival_function_at_times(30).values[0]
base_surv_90  = kmf.survival_function_at_times(90).values[0]
base_surv_365 = kmf.survival_function_at_times(365).values[0]

print(f"Baseline survival: 30-day={base_surv_30:.4f}, 90-day={base_surv_90:.4f}, 365-day={base_surv_365:.4f}")


def calc_pred(cph, newdata, base_surv):
    lp = cph.predict_partial_hazard(newdata)
    return 1 - base_surv ** lp.values


# ============================================================================
# 6. Predicted probabilities on the validation set (three models)
# ============================================================================
val_9f       = val_df[features]
val_sofa     = val_df[['SOFA']]
val_charlson = val_df[['Charlson']]

pred1_30  = calc_pred(cph1, val_9f, base_surv_30)
pred1_90  = calc_pred(cph1, val_9f, base_surv_90)
pred1_365 = calc_pred(cph1, val_9f, base_surv_365)

pred2_30  = calc_pred(cph2, val_sofa, base_surv_30)
pred2_90  = calc_pred(cph2, val_sofa, base_surv_90)
pred2_365 = calc_pred(cph2, val_sofa, base_surv_365)

pred3_30  = calc_pred(cph3, val_charlson, base_surv_30)
pred3_90  = calc_pred(cph3, val_charlson, base_surv_90)
pred3_365 = calc_pred(cph3, val_charlson, base_surv_365)

# ============================================================================
# 7. True outcomes on the validation set
# ============================================================================
y_30  = val_df['death_within_icu_30days'].astype(int)
y_90  = val_df['death_within_icu_90days'].astype(int)
y_365 = val_df['death_within_icu_365days'].astype(int)

print(f"\nValidation set event counts:")
print(f"  30-day deaths: {y_30.sum()}")
print(f"  90-day deaths: {y_90.sum()}")
print(f"  365-day deaths: {y_365.sum()}")


# ============================================================================
# 8. AUC and Brier scores (three models)
# ============================================================================
def safe_auc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_pred)


def safe_brier(y_true, y_pred):
    try:
        return brier_score_loss(y_true, y_pred)
    except:
        return np.nan


auc_30  = [safe_auc(y_30,  p) for p in [pred1_30,  pred2_30,  pred3_30]]
auc_90  = [safe_auc(y_90,  p) for p in [pred1_90,  pred2_90,  pred3_90]]
auc_365 = [safe_auc(y_365, p) for p in [pred1_365, pred2_365, pred3_365]]

brier_30  = [safe_brier(y_30,  p) for p in [pred1_30,  pred2_30,  pred3_30]]
brier_90  = [safe_brier(y_90,  p) for p in [pred1_90,  pred2_90,  pred3_90]]
brier_365 = [safe_brier(y_365, p) for p in [pred1_365, pred2_365, pred3_365]]

model_names = ["GPR-based nomogram", "SOFA", "Charlson"]

print("\n" + "=" * 60)
print("Validation set AUC summary")
print("=" * 60)
print(f"{'Model':<22} {'30-day':<10} {'90-day':<10} {'365-day':<10}")
print("-" * 55)
for i in range(3):
    print(f"{model_names[i]:<22} {auc_30[i]:<10.3f} {auc_90[i]:<10.3f} {auc_365[i]:<10.3f}")

print("\n" + "=" * 60)
print("Validation set Brier score summary")
print("=" * 60)
print(f"{'Model':<22} {'30-day':<10} {'90-day':<10} {'365-day':<10}")
print("-" * 55)
for i in range(3):
    print(f"{model_names[i]:<22} {brier_30[i]:<10.3f} {brier_90[i]:<10.3f} {brier_365[i]:<10.3f}")


# ============================================================================
# 9. DeLong test: nomogram vs SOFA, nomogram vs Charlson
# ============================================================================
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
            diffs.append(roc_auc_score(y_boot, p1_boot) - roc_auc_score(y_boot, p2_boot))
        except:
            continue
    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    p_value = 2 * (1 - norm.cdf(np.abs(mean_diff) / np.std(diffs)))
    return mean_diff, ci_lower, ci_upper, p_value


# Nomogram vs SOFA
diff_30_sofa,  ci_lower_30_sofa,  ci_upper_30_sofa,  p_30_sofa  = delong_bootstrap(y_30,  pred1_30,  pred2_30)
diff_90_sofa,  ci_lower_90_sofa,  ci_upper_90_sofa,  p_90_sofa  = delong_bootstrap(y_90,  pred1_90,  pred2_90)
diff_365_sofa, ci_lower_365_sofa, ci_upper_365_sofa, p_365_sofa = delong_bootstrap(y_365, pred1_365, pred2_365)

# Nomogram vs Charlson
diff_30_ch,  ci_lower_30_ch,  ci_upper_30_ch,  p_30_ch  = delong_bootstrap(y_30,  pred1_30,  pred3_30)
diff_90_ch,  ci_lower_90_ch,  ci_upper_90_ch,  p_90_ch  = delong_bootstrap(y_90,  pred1_90,  pred3_90)
diff_365_ch, ci_lower_365_ch, ci_upper_365_ch, p_365_ch = delong_bootstrap(y_365, pred1_365, pred3_365)

print("\n" + "=" * 60)
print("DeLong test: GPR-based nomogram vs SOFA")
print("=" * 60)
print(f"{'Timepoint':<12} {'GPR-based':<12} {'SOFA':<10} {'Diff':<10} {'P':<10}")
print("-" * 60)
print(f"{'30-day':<12} {auc_30[0]:<12.3f} {auc_30[1]:<10.3f} {diff_30_sofa:<10.3f} {p_30_sofa:<10.4f}")
print(f"{'90-day':<12} {auc_90[0]:<12.3f} {auc_90[1]:<10.3f} {diff_90_sofa:<10.3f} {p_90_sofa:<10.4f}")
print(f"{'365-day':<12} {auc_365[0]:<12.3f} {auc_365[1]:<10.3f} {diff_365_sofa:<10.3f} {p_365_sofa:<10.4f}")

print("\n" + "=" * 60)
print("DeLong test: GPR-based nomogram vs Charlson")
print("=" * 60)
print(f"{'Timepoint':<12} {'GPR-based':<12} {'Charlson':<10} {'Diff':<10} {'P':<10}")
print("-" * 60)
print(f"{'30-day':<12} {auc_30[0]:<12.3f} {auc_30[2]:<10.3f} {diff_30_ch:<10.3f} {p_30_ch:<10.4f}")
print(f"{'90-day':<12} {auc_90[0]:<12.3f} {auc_90[2]:<10.3f} {diff_90_ch:<10.3f} {p_90_ch:<10.4f}")
print(f"{'365-day':<12} {auc_365[0]:<12.3f} {auc_365[2]:<10.3f} {diff_365_ch:<10.3f} {p_365_ch:<10.4f}")


# ============================================================================
# 10. Plot ROC curves (three models)
# ============================================================================
def plot_roc_single(y_true, pred_list, model_names, auc_list, time_label, p_values, output_path):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    plt.figure(figsize=(10, 8))

    for i, (pred, auc_val) in enumerate(zip(pred_list, auc_list)):
        if np.isnan(auc_val):
            continue
        fpr, tpr, _ = roc_curve(y_true, pred)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{model_names[i]} (AUC={auc_val:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)

    from matplotlib.lines import Line2D
    legend_elements = []
    for i, auc_val in enumerate(auc_list):
        if not np.isnan(auc_val):
            legend_elements.append(Line2D([0], [0], color=colors[i], lw=2,
                                          label=f'{model_names[i]} (AUC={auc_val:.3f})'))

    p_sofa, p_ch = p_values
    legend_elements.append(Line2D([0], [0], color='black', lw=0,
                                   label=f'DeLong vs SOFA: P = {p_sofa:.3f}'))
    legend_elements.append(Line2D([0], [0], color='black', lw=0,
                                   label=f'DeLong vs Charlson: P = {p_ch:.3f}'))

    plt.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.title(f'Validation Set - {time_label} Mortality', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='tiff')
    plt.close()
    print(f"  {os.path.basename(output_path)}")


# ============================================================================
# 11. Plot calibration curves (three models)
# ============================================================================
def plot_calibration_single(y_true, pred_list, model_names, brier_list, time_label, output_path):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    plt.figure(figsize=(10, 8))

    for i, (pred, brier) in enumerate(zip(pred_list, brier_list)):
        if np.isnan(pred).all() or np.isnan(brier):
            continue
        try:
            fraction_positive, mean_predicted = calibration_curve(y_true, pred, n_bins=10, strategy='uniform')
            plt.plot(mean_predicted, fraction_positive, 'o-',
                     color=colors[i], lw=2, markersize=8,
                     label=f'{model_names[i]} (Brier={brier:.3f})')
        except:
            continue

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect calibration')
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Observed Proportion of Deaths', fontsize=12)
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.title(f'Calibration Curves - Validation Set ({time_label} Mortality)',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='tiff')
    plt.close()
    print(f"  {os.path.basename(output_path)}")


# ============================================================================
# 12. Generate ROC and calibration curves for the three time points
# ============================================================================
print("\n" + "=" * 60)
print("Generating ROC and calibration curves...")
print("=" * 60)

plot_roc_single(y_30,  [pred1_30,  pred2_30,  pred3_30],
                model_names, auc_30,  "30-day",  (p_30_sofa,  p_30_ch),
                os.path.join(output_dir, 'Figure_ROC_Val_30day.tiff'))
plot_calibration_single(y_30, [pred1_30, pred2_30, pred3_30],
                        model_names, brier_30, "30-day",
                        os.path.join(output_dir, 'Figure_Calibration_Val_30day.tiff'))

plot_roc_single(y_90,  [pred1_90,  pred2_90,  pred3_90],
                model_names, auc_90,  "90-day",  (p_90_sofa,  p_90_ch),
                os.path.join(output_dir, 'Figure_ROC_Val_90day.tiff'))
plot_calibration_single(y_90, [pred1_90, pred2_90, pred3_90],
                        model_names, brier_90, "90-day",
                        os.path.join(output_dir, 'Figure_Calibration_Val_90day.tiff'))

plot_roc_single(y_365, [pred1_365, pred2_365, pred3_365],
                model_names, auc_365, "365-day", (p_365_sofa, p_365_ch),
                os.path.join(output_dir, 'Figure_ROC_Val_365day.tiff'))
plot_calibration_single(y_365, [pred1_365, pred2_365, pred3_365],
                        model_names, brier_365, "365-day",
                        os.path.join(output_dir, 'Figure_Calibration_Val_365day.tiff'))

# ============================================================================
# 13. Save results
# ============================================================================
auc_df = pd.DataFrame({'Model': model_names, 'AUC_30d': auc_30, 'AUC_90d': auc_90, 'AUC_365d': auc_365})
auc_df.to_csv(os.path.join(output_dir, 'AUC_Validation_Set.csv'), index=False)

brier_df = pd.DataFrame({'Model': model_names, 'Brier_30d': brier_30, 'Brier_90d': brier_90, 'Brier_365d': brier_365})
brier_df.to_csv(os.path.join(output_dir, 'Brier_Validation_Set.csv'), index=False)

delong_df = pd.DataFrame({
    'Timepoint': ['30-day', '90-day', '365-day'],
    'GPR_based_AUC': [auc_30[0], auc_90[0], auc_365[0]],
    'SOFA_AUC': [auc_30[1], auc_90[1], auc_365[1]],
    'Charlson_AUC': [auc_30[2], auc_90[2], auc_365[2]],
    'Diff_vs_SOFA': [diff_30_sofa, diff_90_sofa, diff_365_sofa],
    'P_vs_SOFA': [p_30_sofa, p_90_sofa, p_365_sofa],
    'Diff_vs_Charlson': [diff_30_ch, diff_90_ch, diff_365_ch],
    'P_vs_Charlson': [p_30_ch, p_90_ch, p_365_ch],
})
delong_df.to_csv(os.path.join(output_dir, 'DeLong_GPRbased_vs_SOFA_Charlson.csv'), index=False)

print("\n" + "=" * 60)
print("All analyses complete.")
print("=" * 60)
print(f"Output directory: {output_dir}")