import pandas as pd, numpy as np
from pathlib import Path
import warnings, lightgbm as lgb, pickle, json
from sklearn.metrics import roc_auc_score
from datetime import datetime
warnings.filterwarnings('ignore')

BASE_DIR = Path("C:/Users/Paolo/Desktop/NQ/NQdom")
TEST_DAY = "2026-03-19"

label_path = BASE_DIR / "output" / TEST_DAY / "direction_labels_N30_1tick.csv"
df = pd.read_csv(label_path)
EXCLUDE_STR = {'ts', 'signed_return_label', 'signed_return_delta'}
for col in df.columns:
    if col not in EXCLUDE_STR:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

df['vacuum_directional'] = (df['lob_vacuum_count_bid'] - df['lob_vacuum_count_ask']).astype(np.float32)
df['imbalance_velocity_10'] = df['imbalance_1'] - df['imbalance_1'].shift(10)
df['imbalance_velocity_10'] = df['imbalance_velocity_10'].fillna(0.0).astype(np.float32)
df['imbalance_velocity_30'] = df['imbalance_1'] - df['imbalance_1'].shift(30)
df['imbalance_velocity_30'] = df['imbalance_velocity_30'].fillna(0.0).astype(np.float32)
df['stack_net'] = df['stacked_imb_score'].astype(np.float32)
df['microprice_return_10'] = df['microprice'] - df['microprice'].shift(10)
df['microprice_return_10'] = df['microprice_return_10'].fillna(0.0).astype(np.float32)
df['microprice_return_30'] = df['microprice'] - df['microprice'].shift(30)
df['microprice_return_30'] = df['microprice_return_30'].fillna(0.0).astype(np.float32)

NEW_FEATURES = ['vacuum_directional','imbalance_velocity_10','imbalance_velocity_30',
                'stack_net','microprice_return_10','microprice_return_30']
new_feat_set = set(NEW_FEATURES)

core_feats = ['imbalance_1','exhaustion_ratio','pull_bid_1','pull_bid_2',
              'stacked_imb_score','microprice','vacuum_directional',
              'imbalance_velocity_10','imbalance_velocity_30',
              'stack_net','microprice_return_10','microprice_return_30']
for c in df.columns:
    if c not in EXCLUDE_STR and c not in core_feats:
        if any(x in c for x in ['vacuum','stack','ps_net']):
            core_feats.append(c)
core_feats = sorted(set(core_feats))
print(f"Core features ({len(core_feats)})")

all_feat_cols = [c for c in df.columns if c not in EXCLUDE_STR]
X_all = df[all_feat_cols].values.astype(np.float32)
X_core = df[core_feats].values.astype(np.float32)
y_dir = df['signed_return_label'].values.astype(np.int8)
split_idx = int(len(X_all) * 0.75)
X_all_train, X_all_val = X_all[:split_idx], X_all[split_idx:]
X_core_train, X_core_val = X_core[:split_idx], X_core[split_idx:]
y_train, y_val = y_dir[:split_idx], y_dir[split_idx:]

lgbm_core = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, num_leaves=15,
    min_child_samples=100, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.3, reg_lambda=0.3, class_weight='balanced',
    random_state=42, verbose=-1,
)
lgbm_core.fit(X_core_train, y_train, eval_set=[(X_core_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
               lgb.log_evaluation(period=0)])
core_proba = lgbm_core.predict_proba(X_core_val)[:, 1]
core_auc = roc_auc_score(y_val, core_proba)
print(f"CORE AUC: {core_auc:.4f}  std={core_proba.std():.3f}")

lgbm_full = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, num_leaves=15,
    min_child_samples=100, subsample=0.7, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=0.5, class_weight='balanced',
    random_state=42, verbose=-1,
)
lgbm_full.fit(X_all_train, y_train, eval_set=[(X_all_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
               lgb.log_evaluation(period=0)])
full_proba = lgbm_full.predict_proba(X_all_val)[:, 1]
full_auc = roc_auc_score(y_val, full_proba)
print(f"FULL AUC: {full_auc:.4f}  std={full_proba.std():.3f}")

best_auc = max(core_auc, full_auc)
best_name = "CORE" if core_auc >= full_auc else "FULL"
best_model = lgbm_core if core_auc >= full_auc else lgbm_full
best_feats = core_feats if core_auc >= full_auc else all_feat_cols
best_proba = core_proba if core_auc >= full_auc else full_proba
print(f"BEST: {best_name} ({best_auc:.4f})")

fi_df = pd.DataFrame({
    'feature': best_feats,
    'gain': best_model.booster_.feature_importance(importance_type='gain')
}).sort_values('gain', ascending=False).reset_index(drop=True)
total_gain = fi_df['gain'].sum()
fi_df['pct'] = fi_df['gain'] / total_gain * 100

def categorize(feat):
    if feat in new_feat_set: return 'NEW_directional'
    if 'vacuum' in feat: return 'LOB_vacuum'
    if 'imbalance' in feat: return 'imbalance'
    if any(x in feat for x in ['stack','pull','ps_']): return 'stack_pull'
    if 'flow' in feat: return 'flow'
    if 'exhaustion' in feat: return 'exhaustion'
    if any(x in feat for x in ['_1s','_5s','_30s','std','mean','sum']): return 'P4_temporal'
    if 'delta' in feat: return 'delta'
    if any(x in feat for x in ['microprice','spread','mid']): return 'price'
    return 'other'

fi_df['category'] = fi_df['feature'].apply(categorize)
cat_summary = fi_df.groupby('category')['pct'].sum().sort_values(ascending=False)

print(f"\n=== FEATURE IMPORTANCE ({best_name} MODEL) ===")
print(f"{'Rank':<5} {'Feature':<42} {'Gain%':>7}  Category")
print("-" * 72)
for rank, row in fi_df.head(20).iterrows():
    tag = " NEW" if row['feature'] in new_feat_set else ""
    print(f"{rank+1:<5} {row['feature']:<42} {row['pct']:>6.2f}%  {row['category']}{tag}")

print(f"\nGain by category:")
for cat, pct in cat_summary.items():
    bar = "#" * int(pct / 2)
    print(f"  {cat:<22}: {pct:>5.1f}% {bar}")

p4_gain = cat_summary.get('P4_temporal', 0)
new_gain_val = cat_summary.get('NEW_directional', 0)
p3_static = sum(cat_summary.get(c, 0) for c in ['LOB_vacuum','imbalance','stack_pull','flow','exhaustion','delta','price'])
print(f"\nP4 temporal: {p4_gain:.1f}%  P3 static: {p3_static:.1f}%  NEW directional: {new_gain_val:.1f}%")

MODEL_DIR = BASE_DIR / "output" / "direction_model_v1"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_data = {
    'model': best_model,
    'feature_cols': best_feats,
    'test_day': TEST_DAY,
    'core_auc': float(core_auc),
    'full_auc': float(full_auc),
    'best_model': best_name,
    'new_features': NEW_FEATURES,
    'forward_n': 30,
    'threshold_ticks': 1,
    'created_at': datetime.now().isoformat(),
}
with open(MODEL_DIR / "direction_model_test.pkl", 'wb') as f:
    pickle.dump(model_data, f, protocol=5)

meta = {k: v for k, v in model_data.items() if k != 'model'}
with open(MODEL_DIR / "direction_model_metadata.json", 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\nModel saved: {MODEL_DIR}")

print("\n" + "=" * 60)
print(" LAYER 2 DIRECTION MODEL — ONE DAY TEST (2026-03-19)")
print("=" * 60)
print(f" Labeled rows:     {len(df):,}")
print(f" Train / Val:      {split_idx:,} / {len(df)-split_idx:,}")
print(f" Features (all):   {len(all_feat_cols)}")
print(f" Features (core):  {len(core_feats)}")
print(f" New directional:  {len(NEW_FEATURES)}")
print(f" T&S data:         NO")
print("-" * 60)
print(f" LGBM CORE AUC:   {core_auc:.4f}  (core feats)")
print(f" LGBM FULL AUC:   {full_auc:.4f}  (all feats)")
print(f" Baseline (random): 0.5000")
print(f" Previous best (5d): 0.5480 [C3 OvR activity model]")
print("-" * 60)
print(f" Top category: {cat_summary.index[0]} ({cat_summary.iloc[0]:.1f}%)")
print(f" P4 temporal gain:     {p4_gain:.1f}%")
print(f" NEW directional gain: {new_gain_val:.1f}%")
print("=" * 60)
verdict = "WEAK SIGNAL" if best_auc >= 0.505 else "NO DIRECTION SIGNAL"
action = "Scale to all 20 days with core features" if best_auc >= 0.505 else "DOM features insufficient -- T&S data from P2b required"
print(f" VERDICT: {verdict} ({best_auc:.4f})")
print(f" {action}")
print("=" * 60)
