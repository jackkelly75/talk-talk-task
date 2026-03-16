# ============================================================
# UK Telecoms Churn Prediction Model
# ============================================================
# ---  Package import ---
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import duckdb
import matplotlib.pyplot as plt
from pathlib import Path
from dateutil.relativedelta import relativedelta
#modelling packages
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve,
    precision_score, recall_score, f1_score, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

import os
os.chdir(r"C:\Users\jackk\talktalk")
from data_loader import load_all_data
from churn_dashboard import generate_churn_dashboard
from build_features import build_churn_dataset
# ============================================================
# DATA LOADING (using read_csv, read_parquet and duckdb)
# ============================================================
DATA_DIR = Path(r"C:\Users\jackk\talktalk")
data = load_all_data(DATA_DIR, verbose=True)
cease_df = data["cease"]
customer_df = data["customer"]
calls_df = data["calls"]
usage_df = data["usage"]

# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================
all_customers = generate_churn_dashboard(customer_df, cease_df)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
CHURN_WINDOW_DAYS = 60

model_df = build_churn_dataset(
    customer_df = customer_df,
    cease_df         = cease_df,
    calls_df         = calls_df,
    usage_df         = usage_df,
    churn_window_days=30,
    lookback_days    =30
)
# Speed deficit — are they getting what they pay for?
model_df["speed_deficit"] = model_df["speed"] - model_df["line_speed"]

print(f"Total rows: {len(model_df):,}")
print(f"Unique customers: {model_df['unique_customer_identifier'].nunique():,}")
print(f"\nChurn rate: {model_df['churned'].mean():.2%}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
# generate features for model
CATEGORICAL_COLS = ["contract_status", "technology", "sales_channel"]
model_df = pd.get_dummies(model_df, columns=CATEGORICAL_COLS, dummy_na=True)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
FEATURE_COLS = ['contract_dd_cancels', 'dd_cancel_60_day',
       'ooc_days', 'speed', 'line_speed', 'tenure_days',
       'speed_deficit', 'speed_attainment_pct',
       'poor_speed_flag', 'near_ooc_flag', 'ooc_days_positive',
       'days_until_ooc', 'dd_cancel_rate', 'total_calls',
       'loyalty_calls', 'csb_calls', 'ever_called_loyalty', 'avg_talk_time',
       'avg_hold_time', 'total_hold_time', 'days_since_last_call',
       'repeat_contact_flag', 'hold_to_talk_ratio', 'calls_last_30d',
       'calls_prior_60d', 'call_velocity', 'avg_download_mbs',
       'avg_upload_mbs', 'total_download_mbs', 'total_upload_mbs',
       'download_trend_pct', 'days_since_last_usage', 'zero_usage_days_30d',
       'calls_3m_rolling', 'loyalty_call_ever_3m',
       'consecutive_zero_months', 'download_trend_3m',
       'contract_status_01 Early Contract', 'contract_status_02 In Contract',
       'contract_status_03 Soon to be OOC', 'contract_status_04 Coming OOC',
       'contract_status_05 Newly OOC', 'contract_status_06 OOC',
       'technology_FTTC', 'technology_FTTP',
       'technology_GFAST', 'technology_MPF',
       'sales_channel_Field', 'sales_channel_Inbound',
       'sales_channel_Migrated Customer', 'sales_channel_Online - Affiliate',
       'sales_channel_Online - Ambient', 'sales_channel_Online - Other',
       'sales_channel_Online - Search', 'sales_channel_Other',
       'sales_channel_Outbound', 'sales_channel_Partners',
       'sales_channel_Retail', 'sales_channel_Unknown',
       'sales_channel_Webchat']

print(f"There are {len(model_df[model_df['churned'] == 1])} churned users before removal of missing data")
model_df = model_df.dropna(subset=list(FEATURE_COLS))
print(f"There are {len(model_df[model_df['churned'] == 1])} churned users after removal of missing data")

#Split data by time to avoid data leakage between train and test
# Use earlier months to train, most recent months to test
cutoff = model_df["snapshot_date"].quantile(0.8)  # 80/20 time split  -will balance to 82/18 later once months between removed
two_months_before = cutoff - relativedelta(months=2) # leave a 1 month gap ebtween for no data leakage
X_train = model_df[model_df["snapshot_date"] <= two_months_before][FEATURE_COLS]
y_train = model_df[model_df["snapshot_date"] <= two_months_before]['churned']
X_test  = model_df[model_df["snapshot_date"] >  cutoff][FEATURE_COLS]
y_test  = model_df[model_df["snapshot_date"] >  cutoff]['churned']
print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

X_train = X_train.reset_index(drop = True)
X_test = X_test.reset_index(drop = True)
y_train = y_train.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)

# ============================================================
# Undersample using clustering to reduce imbalance
# ============================================================
from sklearn.neighbors import NearestNeighbors
import numpy as np
# isolate majority class
X_majority = X_train[y_train == 0]
def remove_near_duplicates(X, threshold=0.05):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, _ = nbrs.kneighbors(X)
    nearest_dist = distances[:, 1]  # distance to closest neighbour (not self)
    keep_mask = nearest_dist > threshold
    return X[keep_mask]

X_majority_under = remove_near_duplicates(X_majority, threshold=0.05)
y_majority_under = np.zeros(len(X_majority_under))
# minority stays the same
X_minority = X_train[y_train == 1]
y_minority = y_train[y_train == 1]
# combine
X_train = np.vstack([X_majority_under, X_minority])
X_train = pd.DataFrame(X_train)
X_train.columns = X_test.columns
y_train = np.concatenate([y_majority_under, y_minority])
y_train = pd.Series(y_train)

# ============================================================
# MODEL TRAINING — XGBoost (primary) + comparison
# ============================================================
##won't early stopping due to class imbalance for now
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() / 5
print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=1,
    colsample_bytree=1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=1,
    use_label_encoder=False,
    reg_lambda = 2,
    alpha = 2,
)

xgb_model.fit(
    X_train, y_train,
    verbose=50,
)

# -- random forest and LR for comparison
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=4, class_weight="balanced", random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)

lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
])
lr_model.fit(X_train, y_train)


# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(model, X_test, y_test, name="Model"):
	y_pred = model.predict(X_test)
	y_prob = model.predict_proba(X_test)[:, 1]
	auc = roc_auc_score(y_test, y_prob)
	print(f"\n{'='*40}")
	print(f"{name} — AUC: {auc:.4f}")
	print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
	return y_prob, auc

xgb_probs, xgb_auc = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
rf_probs, rf_auc = evaluate_model(rf_model, X_test, y_test, "Random Forest")
lr_probs, lr_auc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

#Plot roc curve
plt.figure(figsize=(7, 5))
for probs, auc, name in [
    (xgb_probs, xgb_auc, "XGBoost"),
    (rf_probs, rf_auc, "Random Forest"),
    (lr_probs, lr_auc, "Logistic Regression")
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()
#Plot pr curve
plt.figure(figsize=(7, 5))
for probs, name in [
    (xgb_probs, "XGBoost"),
    (rf_probs, "Random Forest"),
    (lr_probs, "Logistic Regression")
]:
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    plt.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.3f})")
# Baseline: positive class prevalence (4%)
baseline = 0.0444
plt.hlines(baseline, xmin=0, xmax=1, colors="k", linestyles="--", label="Baseline (4%)")
plt.title("Precision–Recall Curve Comparison")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# FIND BEST THREHOLD
# ============================================================
probs = np.asarray(xgb_probs)
y = np.asarray(y_test).astype(int)
thresholds = np.linspace(0.0, 1.0, 101)
pct_flagged = []
pct_churners_captured = []
precisions = []
recalls = []
f1s = []
total_churners = y.sum()
for t in thresholds:
    preds = (probs >= t).astype(int)
    pct_flagged.append(preds.mean() * 100)  # percent of users flagged
    if total_churners > 0:
        captured = ((preds == 1) & (y == 1)).sum()
        pct_churners_captured.append(captured / total_churners * 100)
    else:
        pct_churners_captured.append(0.0)
    precisions.append(precision_score(y, preds, zero_division=0))
    recalls.append(recall_score(y, preds, zero_division=0))
    f1s.append(f1_score(y, preds, zero_division=0))

pct_flagged = np.array(pct_flagged)
pct_churners_captured = np.array(pct_churners_captured)
precisions = np.array(precisions)
recalls = np.array(recalls)
f1s = np.array(f1s)

#best threshold by F1 (if ties, choose smallest threshold)
best_idx = np.flatnonzero(f1s == f1s.max()).min()
best_t = thresholds[best_idx]
best_precision = precisions[best_idx]
best_recall = recalls[best_idx]
best_f1 = f1s[best_idx]
best_pct_flagged = pct_flagged[best_idx]
best_pct_captured = pct_churners_captured[best_idx]

print(f"Best threshold by F1 = {best_t:.2f}")
print(f"Precision = {best_precision:.3f}, Recall = {best_recall:.3f}, F1 = {best_f1:.3f}")
print(f"Percent flagged = {best_pct_flagged:.2f}%, Percent churners captured = {best_pct_captured:.2f}%")

# Plot
plt.figure(figsize=(8, 6))
plt.plot(pct_flagged, pct_churners_captured, lw=2, label='capture curve')
plt.xlabel('Percent of users flagged as churners (%)')
plt.ylabel('Percent of real churners captured (%)')
plt.title('Percent flagged vs Percent of real churners captured')
plt.grid(alpha=0.3)
# Mark specific thresholds
for t in [0.1, 0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    idx = np.argmin(np.abs(thresholds - t))
    x_pt = pct_flagged[idx]
    y_pt = pct_churners_captured[idx]
    plt.scatter([x_pt], [y_pt], s=80, label=f'threshold {t:.1f}')
    plt.annotate(f'{t:.1f}\n({x_pt:.1f}%, {y_pt:.1f}%)',
                 xy=(x_pt, y_pt),
                 xytext=(5, -20),
                 textcoords='offset points',
                 fontsize=9,
                 arrowprops=dict(arrowstyle='->', lw=0.8))
# add F1 marker - move it over?
plt.scatter([best_pct_flagged], [best_pct_captured], s=160, color='C3', marker='X', label=f'best F1 (t={best_t:.2f})')
plt.annotate(
    f'best F1 score t={best_t:.2f}\nP={best_precision:.2f} R={best_recall:.2f} F1={best_f1:.2f}\n({best_pct_flagged:.1f}%, {best_pct_captured:.1f}%)',
    xy=(best_pct_flagged, best_pct_captured),
    xytext=(140, 20),
    textcoords='offset points',
    fontsize=9,
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.5'),
    arrowprops=dict(arrowstyle='->', lw=0.8)
)
plt.tight_layout()
plt.show()


# plot confustion matrix with best prob from f1 above
cm = confusion_matrix(y_test, xgb_probs > 0.45)
ConfusionMatrixDisplay(cm).plot()
plt.show()


# ============================================================
# FEATURE IMPORTANCE USING SHAP
# ============================================================
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.show()

#plot top 10 using beeswarm plot
top_idx = np.argsort(feat_imp)[-10:]
shap.summary_plot(shap_values[:, top_idx], 
                  X_test.iloc[:, top_idx], 
                  show=False)
plt.tight_layout()
plt.show()



# ============================================================
# BUSINESS IMPACT ANALYSIS
# ============================================================
# Precision/Recall curve — helps set the decision threshold
precision, recall, thresholds = precision_recall_curve(y_test, xgb_probs)

plt.figure(figsize=(9, 5))
plt.plot(recall, precision, marker=".")
plt.xlabel("Recall (% churners caught)")
plt.ylabel("Precision (% of calls that are churners)")
plt.title("Precision-Recall Tradeoff — Helps Set Call List Size")
plt.grid(True)
plt.tight_layout()
plt.show()
# Simulate: if we call the top N% by churn probability, how many churners do we catch?
def simulate_targeting(model_df, top_pct_range=np.arange(0.025, 0.275, 0.025)):
    results = []
    total_churners = model_df["churned"].sum()

    for pct in top_pct_range:
        n = int(len(model_df) * pct)
        top_n = model_df.nlargest(n, "churn_probability")
        caught = top_n["churned"].sum()

        results.append({
            "top_pct": f"{pct*100:.1f}%",                 # show percent called
            "customers_called": n,
            "churners_caught": caught,
            "recall_pct": caught / total_churners * 100,  # convert to %
            "precision_pct": caught / n * 100,            # convert to %
        })

    return pd.DataFrame(results)

targeting_sim = simulate_targeting(model_df)
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.bar(targeting_sim["top_pct"], targeting_sim["customers_called"],
        color="steelblue", alpha=0.5, label="Customers Called")
ax2.plot(targeting_sim["top_pct"], targeting_sim["recall_pct"],
         "r-o", label="Recall (%)")
ax2.plot(targeting_sim["top_pct"], targeting_sim["precision_pct"],
         "g-s", label="Precision (%)")
ax1.set_xlabel("Top % of Customers Called")
ax1.set_ylabel("Number of Customers Called")
ax2.set_ylabel("Rate (%)")
ax2.legend(loc="center right")
ax1.legend(loc="upper left")
plt.title("Business Targeting Simulation: Call List Size vs. Performance")
plt.tight_layout()
plt.show()
