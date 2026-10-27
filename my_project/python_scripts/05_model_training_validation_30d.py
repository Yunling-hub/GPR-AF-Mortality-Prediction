# ============================================================================
# Comparison and evaluation of nine machine learning models
# Outcome: 30-day mortality, 15 features
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss,
                             confusion_matrix, accuracy_score, f1_score)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap
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
nwicu_path = 'data/processed/ML/nwicu_processed.csv'
output_dir = 'results/models/30days'

os.makedirs(output_dir, exist_ok=True)

train_df = pd.read_csv(train_path)
val_df   = pd.read_csv(val_path)
nwicu_df = pd.read_csv(nwicu_path)

print("=" * 60)
print("Data loaded")
print("=" * 60)
print(f"Training set: {train_df.shape}")
print(f"Validation set: {val_df.shape}")
print(f"External set: {nwicu_df.shape}")

# ============================================================================
# 2. 15 features and outcome
# ============================================================================
feature_vars = [
    'GPR', 'Age', 'Gender', 'HR', 'RR',
    'WBC',  'RDW','Serum_chloride','Anion_gap',
    'Diabetes', 'MT', 'ARF', 'Sepsis',
    'Vasopressor', 'CRRT'
]
target = 'death_within_icu_30days'

missing_vars = [v for v in feature_vars if v not in train_df.columns]
if missing_vars:
    print(f"WARNING - missing variables: {missing_vars}")
else:
    print("All feature variables present")

X_train = train_df[feature_vars]
y_train = train_df[target]
X_val   = val_df[feature_vars]
y_val   = val_df[target]
X_nwicu = nwicu_df[feature_vars]
y_nwicu = nwicu_df[target]

print(f"\nTraining set: {X_train.shape[0]} cases, mortality: {y_train.mean() * 100:.2f}%")
print(f"Validation set: {X_val.shape[0]} cases, mortality: {y_val.mean() * 100:.2f}%")
print(f"External set: {X_nwicu.shape[0]} cases, mortality: {y_nwicu.mean() * 100:.2f}%")

for df in [X_train, X_val, X_nwicu]:
    for col in df.columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_nwicu_scaled = scaler.transform(X_nwicu)

# ============================================================================
# 3. Nine models and colors
# ============================================================================
model_names = [
    'Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest',
    'AdaBoost', 'GBDT', 'XGBoost', 'LightGBM', 'CatBoost'
]

model_colors = {
    'Logistic Regression': '#1F77B4',
    'SVM': '#FF7F0E',
    'Decision Tree': '#2CA02C',
    'Random Forest': '#D62728',
    'AdaBoost': '#9467BD',
    'GBDT': '#8C564B',
    'XGBoost': '#E377C2',
    'LightGBM': '#7F7F7F',
    'CatBoost': '#BCBD22'
}

models = {}
for name in model_names:
    if name == 'Logistic Regression':
        models[name] = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', solver='lbfgs')
    elif name == 'SVM':
        models[name] = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42, class_weight='balanced')
    elif name == 'Decision Tree':
        models[name] = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=20, min_samples_leaf=10, class_weight='balanced')
    elif name == 'Random Forest':
        models[name] = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=20, min_samples_leaf=10, class_weight='balanced', n_jobs=-1)
    elif name == 'AdaBoost':
        models[name] = AdaBoostClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
    elif name == 'GBDT':
        models[name] = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5, min_samples_split=20, min_samples_leaf=10, subsample=0.8)
    elif name == 'XGBoost':
        models[name] = xgb.XGBClassifier(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1,
                                         subsample=0.8, colsample_bytree=0.8,
                                         scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                                         use_label_encoder=False, eval_metric='logloss', verbosity=0)
    elif name == 'LightGBM':
        models[name] = lgb.LGBMClassifier(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1,
                                          subsample=0.8, colsample_bytree=0.8, class_weight='balanced', verbosity=-1)
    elif name == 'CatBoost':
        models[name] = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, auto_class_weights='Balanced',
                                          random_seed=42, verbose=0, thread_count=-1)

models_need_scaled = ['Logistic Regression', 'SVM']

# ============================================================================
# 4. Train + internal validation
# ============================================================================
print("\nTraining all models...")

results = []
train_pred_proba = {}
val_pred_proba = {}
train_pred = {}
val_pred = {}

for model_name, model in models.items():
    print(f"  Training: {model_name}")

    if model_name in models_need_scaled:
        X_train_use = X_train_scaled
        X_val_use = X_val_scaled
    else:
        X_train_use = X_train
        X_val_use = X_val

    model.fit(X_train_use, y_train)

    train_proba = model.predict_proba(X_train_use)[:, 1]
    train_pred_label = model.predict(X_train_use)
    train_pred_proba[model_name] = train_proba
    train_pred[model_name] = train_pred_label

    val_proba = model.predict_proba(X_val_use)[:, 1]
    val_pred_label = model.predict(X_val_use)
    val_pred_proba[model_name] = val_proba
    val_pred[model_name] = val_pred_label

    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, train_pred_label).ravel()
    train_auc = roc_auc_score(y_train, train_proba)
    train_acc = accuracy_score(y_train, train_pred_label)
    train_sens = tp_train / (tp_train + fn_train) if (tp_train + fn_train) > 0 else 0
    train_spec = tn_train / (tn_train + fp_train) if (tn_train + fp_train) > 0 else 0
    train_ppv = tp_train / (tp_train + fp_train) if (tp_train + fp_train) > 0 else 0
    train_npv = tn_train / (tn_train + fn_train) if (tn_train + fn_train) > 0 else 0
    train_f1 = f1_score(y_train, train_pred_label)
    train_brier = brier_score_loss(y_train, train_proba)

    tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_val, val_pred_label).ravel()
    val_auc = roc_auc_score(y_val, val_proba)
    val_acc = accuracy_score(y_val, val_pred_label)
    val_sens = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0
    val_spec = tn_val / (tn_val + fp_val) if (tn_val + fp_val) > 0 else 0
    val_ppv = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0
    val_npv = tn_val / (tn_val + fn_val) if (tn_val + fn_val) > 0 else 0
    val_f1 = f1_score(y_val, val_pred_label)
    val_brier = brier_score_loss(y_val, val_proba)

    results.append({
        'Model': model_name,
        'Train_AUC': train_auc, 'Train_Accuracy': train_acc,
        'Train_Sensitivity': train_sens, 'Train_Specificity': train_spec,
        'Train_PPV': train_ppv, 'Train_NPV': train_npv,
        'Train_F1': train_f1, 'Train_Brier': train_brier,
        'Val_AUC': val_auc, 'Val_Accuracy': val_acc,
        'Val_Sensitivity': val_sens, 'Val_Specificity': val_spec,
        'Val_PPV': val_ppv, 'Val_NPV': val_npv,
        'Val_F1': val_f1, 'Val_Brier': val_brier
    })

results_df = pd.DataFrame(results)
print("\nModel performance summary:")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(os.path.join(output_dir, 'All_models_results_30days.csv'), index=False)

# ============================================================================
# Model selection: exclude sensitivity < 5%, then take highest AUC
# ============================================================================
MIN_SENSITIVITY = 0.05

candidates = results_df[results_df['Val_Sensitivity'] >= MIN_SENSITIVITY].copy()

if len(candidates) == 0:
    print(f"WARNING - all models have sensitivity < {MIN_SENSITIVITY}; falling back to highest AUC")
    candidates = results_df.copy()

best_row = candidates.loc[candidates['Val_AUC'].idxmax()]
best_model_name = best_row['Model']
best_model = models[best_model_name]
best_auc = best_row['Val_AUC']

excluded = set(results_df['Model']) - set(candidates['Model'])
if excluded:
    print(f"WARNING - excluded due to sensitivity < {MIN_SENSITIVITY}: {', '.join(excluded)}")

print(f"\nBest model: {best_model_name} (validation AUC={best_auc:.4f}, sensitivity={best_row['Val_Sensitivity']:.4f})")


# ============================================================================
# 5. External validation of the best model
# ============================================================================
if best_model_name in models_need_scaled:
    best_model.fit(X_train_scaled, y_train)
    y_nwicu_proba_best = best_model.predict_proba(X_nwicu_scaled)[:, 1]
    y_nwicu_pred_best  = best_model.predict(X_nwicu_scaled)
    y_train_proba_best = best_model.predict_proba(X_train_scaled)[:, 1]
    y_train_pred_best  = best_model.predict(X_train_scaled)
    y_val_proba_best   = best_model.predict_proba(X_val_scaled)[:, 1]
    y_val_pred_best    = best_model.predict(X_val_scaled)
else:
    best_model.fit(X_train, y_train)
    y_nwicu_proba_best = best_model.predict_proba(X_nwicu)[:, 1]
    y_nwicu_pred_best  = best_model.predict(X_nwicu)
    y_train_proba_best = best_model.predict_proba(X_train)[:, 1]
    y_train_pred_best  = best_model.predict(X_train)
    y_val_proba_best   = best_model.predict_proba(X_val)[:, 1]
    y_val_pred_best    = best_model.predict(X_val)

def calc_metrics(y_true, y_pred, y_proba, name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Dataset': name,
        'AUC': roc_auc_score(y_true, y_proba),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'F1': f1_score(y_true, y_pred),
        'Brier': brier_score_loss(y_true, y_proba)
    }

train_metrics = calc_metrics(y_train, y_train_pred_best, y_train_proba_best, 'Training')
val_metrics   = calc_metrics(y_val, y_val_pred_best, y_val_proba_best, 'Internal Validation')
nwicu_metrics = calc_metrics(y_nwicu, y_nwicu_pred_best, y_nwicu_proba_best, 'External Validation')

optimal_eval_df = pd.DataFrame([train_metrics, val_metrics, nwicu_metrics])
optimal_eval_df.to_csv(os.path.join(output_dir, 'Optimal_model_evaluation_30days.csv'), index=False)

print("\nBest model performance across the three datasets:")
print(optimal_eval_df.round(4).to_string(index=False))

# ============================================================================
# 6. AUC with CI
# ============================================================================
def bootstrap_auc(y_true, y_pred_proba, n_bootstrap=1000):
    np.random.seed(42)
    n_samples = len(y_true)
    indices = np.arange(n_samples)
    aucs = []
    for _ in range(n_bootstrap):
        bootstrap_idx = np.random.choice(indices, size=n_samples, replace=True)
        y_true_boot = y_true.iloc[bootstrap_idx] if hasattr(y_true, 'iloc') else y_true[bootstrap_idx]
        y_proba_boot = y_pred_proba[bootstrap_idx]
        try:
            aucs.append(roc_auc_score(y_true_boot, y_proba_boot))
        except:
            continue
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

forest_data = []
for model_name in models.keys():
    auc = roc_auc_score(y_val, val_pred_proba[model_name])
    lower, upper = bootstrap_auc(y_val, val_pred_proba[model_name])
    forest_data.append({'Model': model_name, 'AUC': auc, 'Lower_CI': lower, 'Upper_CI': upper})

forest_df = pd.DataFrame(forest_data)
forest_df_sorted = forest_df.sort_values('AUC', ascending=False)
forest_df_sorted['AUC_CI'] = forest_df_sorted.apply(
    lambda x: f"{x['AUC']:.3f} ({x['Lower_CI']:.3f}-{x['Upper_CI']:.3f})", axis=1
)
forest_df_sorted[['Model', 'AUC_CI']].to_csv(os.path.join(output_dir, 'AUC_with_CI_30days.csv'), index=False)

print("\nData files saved.")

# ============================================================================
# 7. Figure S4A: ROC on training set
# ============================================================================
plt.figure(figsize=(10, 8))
for model_name in models.keys():
    fpr, tpr, _ = roc_curve(y_train, train_pred_proba[model_name])
    auc = roc_auc_score(y_train, train_pred_proba[model_name])
    plt.plot(fpr, tpr, color=model_colors[model_name], lw=2, label=f'{model_name} (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
plt.xlim([0, 1]); plt.ylim([0, 1])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.legend(loc='lower right', fontsize=8, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.title('ROC curves on training set (30-day)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Figure_S4A_ROC_training_30days.tiff'), dpi=600, bbox_inches='tight', format='tiff')
plt.close()

# ============================================================================
# 8. Figure S4B: ROC on validation set
# ============================================================================
plt.figure(figsize=(10, 8))
for model_name in models.keys():
    fpr, tpr, _ = roc_curve(y_val, val_pred_proba[model_name])
    auc = roc_auc_score(y_val, val_pred_proba[model_name])
    plt.plot(fpr, tpr, color=model_colors[model_name], lw=2, label=f'{model_name} (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
plt.xlim([0, 1]); plt.ylim([0, 1])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.legend(loc='lower right', fontsize=8, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.title('ROC curves on validation set (30-day)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Figure_S4B_ROC_validation_30days.tiff'), dpi=600, bbox_inches='tight', format='tiff')
plt.close()

# ============================================================================
# 9. Figure S4C: Calibration curves
# ============================================================================
plt.figure(figsize=(10, 8))
for model_name in models.keys():
    fraction_pos, mean_pred = calibration_curve(y_val, val_pred_proba[model_name], n_bins=10, strategy='uniform')
    brier = results_df[results_df['Model'] == model_name]['Val_Brier'].values[0]
    plt.plot(mean_pred, fraction_pos, 'o-', color=model_colors[model_name], lw=2, markersize=6,
             label=f'{model_name} (Brier={brier:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect calibration')
plt.xlim([0, 1]); plt.ylim([0, 1])
plt.xlabel('Mean Predicted Probability', fontsize=12)
plt.ylabel('Observed Proportion of Deaths', fontsize=12)
plt.legend(loc='lower right', fontsize=8, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.title('Calibration curves (30-day)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Figure_S4C_Calibration_curves_30days.tiff'), dpi=600, bbox_inches='tight', format='tiff')
plt.close()

# ============================================================================
# 10. Figure S4D: AUC forest plot
# ============================================================================
forest_df_plot = forest_df.sort_values('AUC', ascending=True)
plt.figure(figsize=(10, 8))
y_pos = np.arange(len(forest_df_plot))
forest_colors = [model_colors[name] for name in forest_df_plot['Model']]
for i, (_, row) in enumerate(forest_df_plot.iterrows()):
    plt.errorbar(row['AUC'], i,
                 xerr=[[row['AUC'] - row['Lower_CI']], [row['Upper_CI'] - row['AUC']]],
                 fmt='o', color=forest_colors[i], ecolor='gray', capsize=4, markersize=8,
                 markeredgecolor='black', markeredgewidth=0.5)
    plt.text(row['AUC'] + 0.03, i,
             f"{row['AUC']:.3f} [{row['Lower_CI']:.3f}-{row['Upper_CI']:.3f}]",
             va='center', fontsize=9)
plt.xlabel('AUC with 95% Confidence Interval', fontsize=12)
plt.ylabel('Models', fontsize=12)
plt.yticks(y_pos, forest_df_plot['Model'])
plt.ylim([-1, len(forest_df_plot)])
plt.xlim((0.6, 0.9))
plt.grid(True, alpha=0.3, axis='x')
plt.title('Forest plot of AUCs (30-day)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Figure_S4D_Forest_plot_AUC_30days.tiff'), dpi=600, bbox_inches='tight', format='tiff')
plt.close()

# ============================================================================
# 11. Figure 5A: ROC of the best model
# ============================================================================
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba_best)
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba_best)
fpr_nwicu, tpr_nwicu, _ = roc_curve(y_nwicu, y_nwicu_proba_best)
train_auc = roc_auc_score(y_train, y_train_proba_best)
val_auc = roc_auc_score(y_val, y_val_proba_best)
nwicu_auc = roc_auc_score(y_nwicu, y_nwicu_proba_best)

plt.figure(figsize=(10, 8))
plt.plot(fpr_train, tpr_train, color='#1F77B4', lw=2, label=f'Training (AUC={train_auc:.3f})')
plt.plot(fpr_val, tpr_val, color='#FF7F0E', lw=2, label=f'Validation (AUC={val_auc:.3f})')
plt.plot(fpr_nwicu, tpr_nwicu, color='#2CA02C', lw=2, label=f'External (AUC={nwicu_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
plt.xlim([0, 1]); plt.ylim([0, 1])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.title(f'ROC curves for {best_model_name} (30-day)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Figure_5A_ROC_curves_30days.tiff'), dpi=600, bbox_inches='tight', format='tiff')
plt.close()

# ============================================================================
# 12. Figure 5B: KS curve
# ============================================================================
def calculate_ks(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    ks_values = tpr - fpr
    max_ks_idx = np.argmax(ks_values)
    return ks_values[max_ks_idx], thresholds[max_ks_idx], fpr, tpr, ks_values, thresholds

ks_value, opt_thresh, fpr_nw, tpr_nw, ks_values, thresholds = calculate_ks(y_nwicu, y_nwicu_proba_best)

plt.figure(figsize=(10, 8))
plt.plot(fpr_nw, tpr_nw, color='#2CA02C', lw=2, label=f'ROC (AUC={nwicu_auc:.3f})')
x_ks = np.linspace(0, 1, 100)
plt.plot(x_ks, x_ks + ks_value, '--', color='gray', lw=1.5, alpha=0.7, label=f'KS = {ks_value:.3f}')
idx_opt = np.where(thresholds == opt_thresh)[0][0]
plt.plot(fpr_nw[idx_opt], tpr_nw[idx_opt], 'o', color='#2CA02C', markersize=10,
         markeredgecolor='white', markeredgewidth=1, label=f'Threshold={opt_thresh:.3f}')
plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.3)
plt.xlim([0, 1]); plt.ylim([0, 1])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.title('KS curve on external validation (30-day)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Figure_5B_KS_curve_30days.tiff'), dpi=600, bbox_inches='tight', format='tiff')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(thresholds[:-1], ks_values[:-1], color='#2CA02C', lw=2)
plt.axhline(y=ks_value, color='gray', linestyle='--', alpha=0.7, label=f'Max KS = {ks_value:.3f}')
plt.axvline(x=opt_thresh, color='gray', linestyle='--', alpha=0.7, label=f'Threshold = {opt_thresh:.3f}')
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('KS value (TPR - FPR)', fontsize=12)
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.title('KS values at different thresholds (30-day)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Figure_5B_KS_thresholds_30days.tiff'), dpi=600, bbox_inches='tight', format='tiff')
plt.close()

# ============================================================================
# 13. Figure 5C: Calibration curves (best model)
# ============================================================================
def get_calibration(y_true, y_proba, n_bins=10):
    fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
    return fraction_pos, mean_pred, brier_score_loss(y_true, y_proba)

frac_train, mean_train, brier_train = get_calibration(y_train, y_train_proba_best)
frac_val, mean_val, brier_val = get_calibration(y_val, y_val_proba_best)
frac_nwicu, mean_nwicu, brier_nwicu = get_calibration(y_nwicu, y_nwicu_proba_best)

plt.figure(figsize=(10, 8))
plt.plot(mean_train, frac_train, 'o-', color='#1F77B4', lw=2, markersize=8, label=f'Training (Brier={brier_train:.3f})')
plt.plot(mean_val, frac_val, 'o-', color='#FF7F0E', lw=2, markersize=8, label=f'Validation (Brier={brier_val:.3f})')
plt.plot(mean_nwicu, frac_nwicu, 'o-', color='#2CA02C', lw=2, markersize=8, label=f'External (Brier={brier_nwicu:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect')
plt.xlim([0, 1]); plt.ylim([0, 1])
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Observed Proportion', fontsize=12)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.title(f'Calibration curves for {best_model_name} (30-day)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Figure_5C_Calibration_curves_30days.tiff'), dpi=600, bbox_inches='tight', format='tiff')
plt.close()


# ============================================================================
# 14. Figure 5D: DCA (with automatic threshold detection)
# ============================================================================
def calculate_net_benefit(y_true, y_pred_proba, threshold):
    y_true = np.asarray(y_true)
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)
    return (tp / n) - (fp / n) * (threshold / (1 - threshold))

def find_zero_crossing(thresholds, net_benefits, start_positive=True):
    """
    Find the first threshold at which net benefit crosses from positive to
    negative (start_positive=True) or from negative to positive.
    Returns an interpolated (threshold, net benefit) pair, or (None, None).
    """
    thresholds = np.asarray(thresholds, dtype=float)
    nb = np.asarray(net_benefits, dtype=float)
    for i in range(1, len(nb)):
        prev, curr = nb[i - 1], nb[i]
        if start_positive:
            if prev > 0 and curr <= 0:
                t0, t1 = thresholds[i - 1], thresholds[i]
                b0, b1 = prev, curr
                t_cross = t0 + (0 - b0) * (t1 - t0) / (b1 - b0) if b1 != b0 else t1
                return t_cross, 0.0
        else:
            if prev < 0 and curr >= 0:
                t0, t1 = thresholds[i - 1], thresholds[i]
                b0, b1 = prev, curr
                t_cross = t0 + (0 - b0) * (t1 - t0) / (b1 - b0) if b1 != b0 else t1
                return t_cross, 0.0
    return None, None

def max_positive_threshold(thresholds, net_benefits):
    """Return the largest threshold with positive net benefit (discrete points)."""
    thresholds = np.asarray(thresholds, dtype=float)
    nb = np.asarray(net_benefits, dtype=float)
    pos_mask = nb > 0
    if pos_mask.any():
        return thresholds[pos_mask].max(), nb[pos_mask].max()
    return None, None

thresholds_dca = np.arange(0.01, 1.00, 0.01)
nb_train = np.array([calculate_net_benefit(y_train, y_train_proba_best, t) for t in thresholds_dca])
nb_val   = np.array([calculate_net_benefit(y_val,   y_val_proba_best,   t) for t in thresholds_dca])
nb_nwicu = np.array([calculate_net_benefit(y_nwicu, y_nwicu_proba_best, t) for t in thresholds_dca])
treat_all = np.array([(y_nwicu.mean() - (1 - y_nwicu.mean()) * t / (1 - t)) for t in thresholds_dca])

# ---- Automatic zero-crossing detection on the external set ----
cross_ext, _ = find_zero_crossing(thresholds_dca, nb_nwicu, start_positive=True)
max_pos_ext, max_pos_nb_ext = max_positive_threshold(thresholds_dca, nb_nwicu)

# Also record training / validation (optional, for comparison)
cross_train, _ = find_zero_crossing(thresholds_dca, nb_train, start_positive=True)
cross_val,   _ = find_zero_crossing(thresholds_dca, nb_val,   start_positive=True)

print("\n" + "=" * 60)
print("DCA threshold detection (net benefit crossing zero)")
print("=" * 60)
print(f"Training  : {cross_train if cross_train is not None else 'NA'}")
print(f"Validation: {cross_val if cross_val is not None else 'NA'}")
print(f"External  : {cross_ext if cross_ext is not None else 'NA'}")
print(f"External max positive threshold: {max_pos_ext if max_pos_ext is not None else 'NA'}"
      f" (net benefit = {max_pos_nb_ext if max_pos_nb_ext is not None else 'NA'})")

# Save threshold results
dca_threshold_df = pd.DataFrame({
    'Dataset': ['Training', 'Validation', 'External'],
    'Zero_Crossing_Threshold': [cross_train, cross_val, cross_ext],
    'Max_Positive_Threshold': [
        max_positive_threshold(thresholds_dca, nb_train)[0],
        max_positive_threshold(thresholds_dca, nb_val)[0],
        max_pos_ext
    ]
})
dca_threshold_df.to_csv(os.path.join(output_dir, 'DCA_thresholds_30days.csv'), index=False)

# Save full DCA curve data
dca_curve_df = pd.DataFrame({
    'threshold': thresholds_dca,
    'net_benefit_train': nb_train,
    'net_benefit_val': nb_val,
    'net_benefit_external': nb_nwicu,
    'net_benefit_treat_all': treat_all
})
dca_curve_df.to_csv(os.path.join(output_dir, 'DCA_curve_data_30days.csv'), index=False)
# ---- Plot (same style as before, no threshold annotation lines) ----
plt.figure(figsize=(10, 8))
plt.plot(thresholds_dca, nb_train, color='#1F77B4', lw=2, label='Training')
plt.plot(thresholds_dca, nb_val, color='#FF7F0E', lw=2, label='Validation')
plt.plot(thresholds_dca, nb_nwicu, color='#2CA02C', lw=2, label='External')
plt.plot(thresholds_dca, treat_all, color='gray', linestyle='--', lw=1, alpha=0.5, label='Treat all')
plt.plot(thresholds_dca, np.zeros_like(thresholds_dca), color='gray', linestyle=':', lw=1, alpha=0.5, label='Treat none')
plt.xlabel('Threshold Probability', fontsize=12)
plt.ylabel('Net Benefit', fontsize=12)
plt.title(f'Decision Curve Analysis for {best_model_name} (30-day)', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim([0, 1]); plt.ylim([-0.05, 0.35])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Figure_5D_DCA_curves_30days.tiff'), dpi=600, bbox_inches='tight', format='tiff')
plt.close()

print("DCA curve data and thresholds saved.")

# ============================================================================
# 15. SHAP analysis (with AdaBoost fallback)
# ============================================================================
if best_model_name in models_need_scaled:
    X_explain = pd.DataFrame(X_val_scaled, columns=feature_vars)
else:
    X_explain = X_val.copy()

if len(X_explain) > 500:
    X_explain = X_explain.sample(n=500, random_state=42)

tree_supported = ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest', 'GBDT', 'Decision Tree']

if best_model_name in tree_supported:
    if best_model_name in models_need_scaled:
        explainer = shap.TreeExplainer(best_model, X_train_scaled)
    else:
        explainer = shap.TreeExplainer(best_model, X_train)
    shap_values = explainer.shap_values(X_explain)
else:
    print(f"  {best_model_name} does not support TreeExplainer; using KernelExplainer")
    bg_size = min(100, len(X_train))
    if best_model_name in models_need_scaled:
        background = shap.sample(X_train_scaled, bg_size, random_state=42)
    else:
        background = shap.sample(X_train, bg_size, random_state=42)
    f = lambda x: best_model.predict_proba(x)[:, 1]
    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(X_explain, nsamples=100)
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        explainer.expected_value = float(np.array(explainer.expected_value).ravel()[0])

if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]

# SHAP summary
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_explain, feature_names=feature_vars, show=False, max_display=len(feature_vars))
plt.gca().set_title('')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Figure_3C_SHAP_summary_30days.tiff'), dpi=600, bbox_inches='tight', format='tiff')
plt.close()

# SHAP importance bar
shap_importance = np.abs(shap_values).mean(axis=0)
feature_importance_df = pd.DataFrame({'feature': feature_vars, 'importance': shap_importance}).sort_values('importance', ascending=True)
feature_importance_df = feature_importance_df.reset_index(drop=True)
feature_importance_df['rank'] = len(feature_importance_df) - feature_importance_df.index
feature_importance_df.to_csv(os.path.join(output_dir, 'shap_feature_importance_30days.csv'), index=False)

plt.figure(figsize=(10, 8))
bars = plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='steelblue', alpha=0.8)
plt.xlabel('Mean |SHAP value|', fontsize=12)
plt.grid(True, alpha=0.3, axis='x')
for bar, v, rank in zip(bars, feature_importance_df['importance'], feature_importance_df['rank']):
    plt.text(v + 0.001, bar.get_y() + bar.get_height() / 2, f'{v:.4f} (rank {int(rank)})', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Figure_3D_SHAP_importance_bar_30days.tiff'), dpi=600, bbox_inches='tight', format='tiff')
plt.close()

# SHAP force plot
if isinstance(explainer.expected_value, (list, np.ndarray)):
    ev = np.array(explainer.expected_value).ravel()
    base_value = ev[0] if len(ev) == 1 else ev[1]
else:
    base_value = float(explainer.expected_value)

if best_model_name in models_need_scaled:
    y_proba_sample = best_model.predict_proba(X_explain)[:, 1]
else:
    y_proba_sample = best_model.predict_proba(X_explain)[:, 1]

moderate_mask = (y_proba_sample >= 0.4) & (y_proba_sample <= 0.6)
moderate_indices = np.where(moderate_mask)[0]

if len(moderate_indices) > 0:
    patient_idx = moderate_indices[0]
    patient_data = X_explain.iloc[[patient_idx]]
    patient_shap = shap_values[patient_idx:patient_idx + 1, :]

    plt.figure(figsize=(18, 4))
    shap.force_plot(base_value, patient_shap[0, :], patient_data.iloc[0, :], feature_names=feature_vars, matplotlib=True, show=False)
    plt.subplots_adjust(bottom=0.3, top=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_3E_SHAP_Force_plot_30days.tiff'), dpi=600, bbox_inches='tight', format='tiff')
    plt.close()

# SHAP GPR dependence
plt.figure(figsize=(10, 6))
shap.dependence_plot('GPR', shap_values, X_explain, feature_names=feature_vars, show=False, title=None)
plt.xlabel('GPR', fontsize=12)
plt.ylabel('SHAP value for GPR', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'SHAP_dependence_GPR_30days.tiff'), dpi=600, bbox_inches='tight', format='tiff')
plt.close()

print("\n" + "=" * 60)
print("30-day analysis complete.")
print(f"Best model: {best_model_name}")
print(f"   Validation AUC: {val_auc:.4f}")
print(f"   External AUC: {nwicu_auc:.4f}")
print(f"Output directory: {output_dir}")
print("=" * 60)