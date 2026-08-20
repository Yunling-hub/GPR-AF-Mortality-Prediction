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
                             confusion_matrix, accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap
import os
import warnings
warnings.filterwarnings('ignore')

# font settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# ----- read data -----
train_df = pd.read_csv("./data/splits/train_set.csv")
val_df = pd.read_csv("./data/splits/val_set.csv")
nwicu_df = pd.read_csv("./data/processed/NWICU/df_nwicu_cleaned.csv")

os.makedirs("./results/", exist_ok=True)

print("=" * 60)
print("Data loaded")
print("=" * 60)
print(f"Training: {train_df.shape}")
print(f"Validation: {val_df.shape}")
print(f"External: {nwicu_df.shape}")

# ----- features and target -----
feature_vars = [
    'GPR', 'Age', 'Gender', 'HR', 'RR',
    'WBC', 'Platelets', 'RDW',
    'Serum_sodium',
    'PT', 'APTT', 'BUN', 'Serum_creatinine',
    'Diabetes', 'HF', 'MT', 'CKD', 'ARF', 'Sepsis',
    'Vasopressor', 'CRRT'
]
target = 'death_within_icu_365days'

X_train = train_df[feature_vars]
y_train = train_df[target]
X_val = val_df[feature_vars]
y_val = val_df[target]
X_nwicu = nwicu_df[feature_vars]
y_nwicu = nwicu_df[target]

print(f"\nTraining: {X_train.shape[0]} cases, mortality {y_train.mean()*100:.2f}%")
print(f"Validation: {X_val.shape[0]} cases, mortality {y_val.mean()*100:.2f}%")
print(f"External: {X_nwicu.shape[0]} cases, mortality {y_nwicu.mean()*100:.2f}%")

# fill missing if any (should be none after preprocessing)
for df in [X_train, X_val, X_nwicu]:
    for col in df.columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

# scale for SVM and LR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_nwicu_scaled = scaler.transform(X_nwicu)

# ----- models -----
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
        models[name] = LogisticRegression(random_state=42, max_iter=1000,
                                          class_weight='balanced', solver='lbfgs')
    elif name == 'SVM':
        models[name] = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                           random_state=42, class_weight='balanced')
    elif name == 'Decision Tree':
        models[name] = DecisionTreeClassifier(random_state=42, max_depth=5,
                                              min_samples_split=20, min_samples_leaf=10,
                                              class_weight='balanced')
    elif name == 'Random Forest':
        models[name] = RandomForestClassifier(n_estimators=100, random_state=42,
                                              max_depth=10, min_samples_split=20,
                                              min_samples_leaf=10, class_weight='balanced', n_jobs=-1)
    elif name == 'AdaBoost':
        models[name] = AdaBoostClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
    elif name == 'GBDT':
        models[name] = GradientBoostingClassifier(n_estimators=100, random_state=42,
                                                  max_depth=5, min_samples_split=20,
                                                  min_samples_leaf=10, subsample=0.8)
    elif name == 'XGBoost':
        models[name] = xgb.XGBClassifier(n_estimators=100, random_state=42, max_depth=5,
                                         learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                         scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                                         use_label_encoder=False, eval_metric='logloss', verbosity=0)
    elif name == 'LightGBM':
        models[name] = lgb.LGBMClassifier(n_estimators=100, random_state=42, max_depth=5,
                                          learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                          class_weight='balanced', verbosity=-1)
    elif name == 'CatBoost':
        models[name] = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1,
                                          auto_class_weights='Balanced', random_seed=42,
                                          verbose=0, thread_count=-1)

models_need_scaled = ['Logistic Regression', 'SVM']

# ----- train all models -----
print("\nTraining models...")
results = []
train_pred_proba = {}
val_pred_proba = {}

for name, model in models.items():
    print(f"  {name}")
    if name in models_need_scaled:
        X_train_use, X_val_use = X_train_scaled, X_val_scaled
    else:
        X_train_use, X_val_use = X_train, X_val

    model.fit(X_train_use, y_train)

    train_proba = model.predict_proba(X_train_use)[:, 1]
    val_proba = model.predict_proba(X_val_use)[:, 1]
    train_pred = model.predict(X_train_use)
    val_pred = model.predict(X_val_use)

    train_pred_proba[name] = train_proba
    val_pred_proba[name] = val_proba

    # metrics
    tn, fp, fn, tp = confusion_matrix(y_train, train_pred).ravel()
    train_auc = roc_auc_score(y_train, train_proba)
    train_acc = accuracy_score(y_train, train_pred)
    train_sens = tp/(tp+fn) if (tp+fn)>0 else 0
    train_spec = tn/(tn+fp) if (tn+fp)>0 else 0
    train_brier = brier_score_loss(y_train, train_proba)

    tn, fp, fn, tp = confusion_matrix(y_val, val_pred).ravel()
    val_auc = roc_auc_score(y_val, val_proba)
    val_acc = accuracy_score(y_val, val_pred)
    val_sens = tp/(tp+fn) if (tp+fn)>0 else 0
    val_spec = tn/(tn+fp) if (tn+fp)>0 else 0
    val_brier = brier_score_loss(y_val, val_proba)

    results.append({
        'Model': name,
        'Train_AUC': train_auc, 'Train_Accuracy': train_acc,
        'Train_Sensitivity': train_sens, 'Train_Specificity': train_spec,
        'Train_Brier': train_brier,
        'Val_AUC': val_auc, 'Val_Accuracy': val_acc,
        'Val_Sensitivity': val_sens, 'Val_Specificity': val_spec,
        'Val_Brier': val_brier
    })

results_df = pd.DataFrame(results)
results_df.to_csv("./results/All_models_results_365days.csv", index=False)
print(f"\nResults saved.")

# best model
best_name = results_df.loc[results_df['Val_AUC'].idxmax(), 'Model']
best_model = models[best_name]
print(f"Best model: {best_name} (AUC={results_df['Val_AUC'].max():.4f})")

# ----- evaluate best model on all datasets -----
def calc_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'AUC': roc_auc_score(y_true, y_proba),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Sensitivity': tp/(tp+fn) if (tp+fn)>0 else 0,
        'Specificity': tn/(tn+fp) if (tn+fp)>0 else 0,
        'Brier': brier_score_loss(y_true, y_proba)
    }

if best_name in models_need_scaled:
    best_model.fit(X_train_scaled, y_train)
    y_train_proba = best_model.predict_proba(X_train_scaled)[:, 1]
    y_val_proba = best_model.predict_proba(X_val_scaled)[:, 1]
    y_nwicu_proba = best_model.predict_proba(X_nwicu_scaled)[:, 1]
    y_train_pred = best_model.predict(X_train_scaled)
    y_val_pred = best_model.predict(X_val_scaled)
    y_nwicu_pred = best_model.predict(X_nwicu_scaled)
else:
    best_model.fit(X_train, y_train)
    y_train_proba = best_model.predict_proba(X_train)[:, 1]
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    y_nwicu_proba = best_model.predict_proba(X_nwicu)[:, 1]
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_nwicu_pred = best_model.predict(X_nwicu)

train_metrics = calc_metrics(y_train, y_train_pred, y_train_proba)
val_metrics = calc_metrics(y_val, y_val_pred, y_val_proba)
nwicu_metrics = calc_metrics(y_nwicu, y_nwicu_pred, y_nwicu_proba)

optimal_eval = pd.DataFrame([
    {'Dataset': 'Training', **train_metrics},
    {'Dataset': 'Validation', **val_metrics},
    {'Dataset': 'External', **nwicu_metrics}
])
optimal_eval.to_csv("./results/Optimal_model_evaluation_365days.csv", index=False)

# ----- bootstrap AUC CI -----
def bootstrap_auc(y_true, y_proba, n_bootstrap=1000):
    np.random.seed(42)
    n = len(y_true)
    aucs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            aucs.append(roc_auc_score(y_true.iloc[idx], y_proba[idx]))
        except:
            continue
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

forest_data = []
for name in model_names:
    auc = roc_auc_score(y_val, val_pred_proba[name])
    lower, upper = bootstrap_auc(y_val, val_pred_proba[name])
    forest_data.append({'Model': name, 'AUC': auc, 'Lower_CI': lower, 'Upper_CI': upper})

forest_df = pd.DataFrame(forest_data).sort_values('AUC', ascending=False)
forest_df['AUC_CI'] = forest_df.apply(
    lambda x: f"{x['AUC']:.3f} ({x['Lower_CI']:.3f}-{x['Upper_CI']:.3f})", axis=1)
forest_df[['Model', 'AUC_CI']].to_csv("./results/AUC_with_CI_365days.csv", index=False)

print("All results saved.")

# ----- ROC training -----
plt.figure(figsize=(10,8))
for name in model_names:
    fpr, tpr, _ = roc_curve(y_train, train_pred_proba[name])
    plt.plot(fpr, tpr, color=model_colors[name], lw=2,
             label=f"{name} (AUC={roc_auc_score(y_train, train_pred_proba[name]):.3f})")
plt.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./results/ROC_training.tiff", dpi=600, format='tiff')
plt.close()

# ----- ROC validation -----
plt.figure(figsize=(10,8))
for name in model_names:
    fpr, tpr, _ = roc_curve(y_val, val_pred_proba[name])
    plt.plot(fpr, tpr, color=model_colors[name], lw=2,
             label=f"{name} (AUC={roc_auc_score(y_val, val_pred_proba[name]):.3f})")
plt.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right', fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./results/ROC_validation.tiff", dpi=600, format='tiff')
plt.close()

# ----- Calibration curves -----
plt.figure(figsize=(10,8))
for name in model_names:
    frac_pos, mean_pred = calibration_curve(y_val, val_pred_proba[name], n_bins=10)
    brier = results_df[results_df['Model']==name]['Val_Brier'].values[0]
    plt.plot(mean_pred, frac_pos, 'o-', color=model_colors[name], lw=2, markersize=6,
             label=f"{name} (Brier={brier:.3f})")
plt.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Perfect')
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Proportion')
plt.legend(loc='lower right', fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./results/Calibration_curves.tiff", dpi=600, format='tiff')
plt.close()

# ----- Forest plot AUC -----
forest_plot = forest_df.sort_values('AUC', ascending=True)
plt.figure(figsize=(10,8))
for i, (_, row) in enumerate(forest_plot.iterrows()):
    plt.errorbar(row['AUC'], i,
                 xerr=[[row['AUC']-row['Lower_CI']], [row['Upper_CI']-row['AUC']]],
                 fmt='o', color=model_colors[row['Model']], ecolor='gray',
                 capsize=4, markersize=8)
    plt.text(row['AUC']+0.02, i, f"{row['AUC']:.3f} [{row['Lower_CI']:.3f}-{row['Upper_CI']:.3f}]",
             va='center', fontsize=8)
plt.xlabel('AUC with 95% CI')
plt.ylabel('Models')
plt.yticks(range(len(forest_plot)), forest_plot['Model'])
plt.xlim(0.6, 0.9)
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("./results/Forest_plot_AUC.tiff", dpi=600, format='tiff')
plt.close()

# ----- ROC best model -----
plt.figure(figsize=(10,8))
for data, prob, label, color in zip(
    [y_train, y_val, y_nwicu],
    [y_train_proba, y_val_proba, y_nwicu_proba],
    ['Training', 'Validation', 'External'],
    ['#1F77B4', '#FF7F0E', '#2CA02C']):
    fpr, tpr, _ = roc_curve(data, prob)
    plt.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={roc_auc_score(data, prob):.3f})")
plt.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./results/ROC_best_model.tiff", dpi=600, format='tiff')
plt.close()

# ----- Calibration best model -----
def get_calib(y_true, y_proba):
    frac, mean = calibration_curve(y_true, y_proba, n_bins=10)
    return frac, mean, brier_score_loss(y_true, y_proba)

plt.figure(figsize=(10,8))
for data, prob, label, color in zip(
    [y_train, y_val, y_nwicu],
    [y_train_proba, y_val_proba, y_nwicu_proba],
    ['Training', 'Validation', 'External'],
    ['#1F77B4', '#FF7F0E', '#2CA02C']):
    frac, mean, brier = get_calib(data, prob)
    plt.plot(mean, frac, 'o-', color=color, lw=2, markersize=8, label=f"{label} (Brier={brier:.3f})")
plt.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Perfect')
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Proportion')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./results/Calibration_best_model.tiff", dpi=600, format='tiff')
plt.close()

# ----- DCA -----
def net_benefit(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)
    return (tp/n) - (fp/n) * (threshold/(1-threshold))

thresholds_dca = np.arange(0.01, 1.00, 0.01)
nb_train = [net_benefit(y_train, y_train_proba, t) for t in thresholds_dca]
nb_val = [net_benefit(y_val, y_val_proba, t) for t in thresholds_dca]
nb_nwicu = [net_benefit(y_nwicu, y_nwicu_proba, t) for t in thresholds_dca]
treat_all = [(y_nwicu.mean() - (1-y_nwicu.mean()) * t/(1-t)) for t in thresholds_dca]

plt.figure(figsize=(10,8))
plt.plot(thresholds_dca, nb_train, color='#1F77B4', lw=2, label='Training')
plt.plot(thresholds_dca, nb_val, color='#FF7F0E', lw=2, label='Validation')
plt.plot(thresholds_dca, nb_nwicu, color='#2CA02C', lw=2, label='External')
plt.plot(thresholds_dca, treat_all, 'gray', '--', lw=1, alpha=0.5, label='Treat all')
plt.plot(thresholds_dca, np.zeros_like(thresholds_dca), 'gray', ':', lw=1, alpha=0.5, label='Treat none')
plt.xlabel('Threshold')
plt.ylabel('Net Benefit')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("./results/DCA_curves.tiff", dpi=600, format='tiff')
plt.close()

# save optimal threshold
opt_nb_idx = np.argmax(nb_nwicu)
opt_thresh_dca = thresholds_dca[opt_nb_idx]
pd.DataFrame([{
    'Optimal_Threshold': opt_thresh_dca,
    'Max_Net_Benefit': nb_nwicu[opt_nb_idx]
}]).to_csv("./results/DCA_optimal_threshold_365days.csv", index=False)

# ----- SHAP analysis -----
print("\nSHAP analysis...")
if best_name in models_need_scaled:
    explainer = shap.TreeExplainer(best_model, X_train_scaled)
    X_sample = X_val_scaled
else:
    explainer = shap.TreeExplainer(best_model, X_train)
    X_sample = X_val

if len(X_sample) > 500:
    X_sample = X_sample.sample(n=500, random_state=42)

shap_values = explainer.shap_values(X_sample)
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]

# SHAP summary
plt.figure(figsize=(12,8))
shap.summary_plot(shap_values, X_sample, feature_names=feature_vars, show=False)
plt.tight_layout()
plt.savefig("./results/SHAP_summary.tiff", dpi=600, format='tiff')
plt.close()

# SHAP importance
importance = np.abs(shap_values).mean(axis=0)
imp_df = pd.DataFrame({'feature': feature_vars, 'importance': importance})
imp_df = imp_df.sort_values('importance', ascending=True)
imp_df.to_csv("./results/shap_feature_importance.csv", index=False)

plt.figure(figsize=(10,8))
plt.barh(imp_df['feature'], imp_df['importance'], color='steelblue', alpha=0.8)
plt.xlabel('Mean |SHAP value|')
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("./results/SHAP_importance.tiff", dpi=600, format='tiff')
plt.close()

print("\nAll done!")
print(f"Best model: {best_name}")
print(f"Validation AUC: {val_metrics['AUC']:.4f}")
print(f"External AUC: {nwicu_metrics['AUC']:.4f}")