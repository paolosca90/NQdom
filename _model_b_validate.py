import pandas as pd, numpy as np
from pathlib import Path
import warnings, lightgbm as lgb, json
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

BASE_DIR = Path("C:/Users/Paolo/Desktop/NQ/NQdom")
DAYS = [
    "2026-03-13","2026-03-16","2026-03-17","2026-03-18","2026-03-19",
    "2026-03-20","2026-03-23","2026-03-24","2026-03-25","2026-03-26",
    "2026-03-27","2026-03-30","2026-03-31","2026-04-01","2026-04-02",
    "2026-04-06","2026-04-07","2026-04-08","2026-04-09","2026-04-10",
]
PS_NET_FEATURES = [
    'ps_delta_L1','ps_delta_L1_mean_1s','ps_delta_L1_mean_5s','ps_delta_L1_mean_30s',
    'ps_net_weighted','ps_net_weighted_mean_1s','ps_net_weighted_mean_5s','ps_net_weighted_mean_30s',
]

# STEP 1: Load all 20 days
all_dfs = []
for day in DAYS:
    path = BASE_DIR / "output" / day / "direction_labels_N30_1tick.csv"
    if not path.exists():
        print(f"  MISSING: {day}")
        continue
    df = pd.read_csv(path, usecols=PS_NET_FEATURES + ['ts','signed_return_label'])
    for col in PS_NET_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    df['day'] = day
    all_dfs.append(df)
    print(f"  {day}: {len(df):,} rows")

full_df = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal: {len(full_df):,} rows across {len(all_dfs)} days")
print(f"UP: {(full_df['signed_return_label']==1).sum():,}  "
      f"DOWN: {(full_df['signed_return_label']==0).sum():,}  "
      f"Balance: {(full_df['signed_return_label']==1).mean():.3f}")
for col in PS_NET_FEATURES:
    if col not in full_df.columns:
        print(f"STOP: {col} missing")
        raise SystemExit(1)
print("All 8 ps_net features verified\n")

# STEP 2: Walk-forward validation
LGBM_PARAMS = dict(
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, class_weight='balanced',
    random_state=42, verbose=-1,
)

FOLDS = [
    (DAYS[0:10],  DAYS[11:13]),
    (DAYS[2:12],  DAYS[13:15]),
    (DAYS[4:14],  DAYS[15:17]),
    (DAYS[6:16],  DAYS[17:19]),
    (DAYS[8:18],  DAYS[18:20]),
]

fold_results = []
print(f"{'Fold':<6} {'TrainD':>7} {'TrainRows':>11} {'ValRows':>10} {'UP%':>6} {'AUC':>7} {'Std':>7}")
print("-" * 60)

for fold_i, (train_days, test_days) in enumerate(FOLDS, 1):
    train_df = full_df[full_df['day'].isin(train_days)]
    test_df  = full_df[full_df['day'].isin(test_days)]
    X_train = train_df[PS_NET_FEATURES].values.astype(np.float32)
    y_train = train_df['signed_return_label'].values.astype(np.int8)
    X_test  = test_df[PS_NET_FEATURES].values.astype(np.float32)
    y_test  = test_df['signed_return_label'].values.astype(np.int8)
    if len(np.unique(y_test)) < 2:
        print(f"  Fold {fold_i}: skipped (single class)")
        continue
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    proba = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    fold_results.append({
        'fold': fold_i, 'train_days': len(train_days), 'n_train': len(X_train),
        'n_test': len(X_test), 'up_pct': float(y_test.mean()),
        'auc': float(auc), 'proba_std': float(proba.std()),
        'test_period': f"{test_days[0][5:]} to {test_days[-1][5:]}",
    })
    print(f"  {fold_i:<4} {len(train_days):>7} {len(X_train):>11,} {len(X_test):>10,} "
          f"{y_test.mean():>5.1%} {auc:>7.4f} {proba.std():>7.4f}")

# STEP 3: Aggregate
aucs  = [r['auc'] for r in fold_results]
mean_auc = np.mean(aucs)
std_auc  = np.std(aucs)
min_auc  = np.min(aucs)
max_auc  = np.max(aucs)
n_above_random = sum(a > 0.505 for a in aucs)
march_folds = [r for r in fold_results if '03' in r['test_period']]
april_folds = [r for r in fold_results if '04' in r['test_period']]
march_auc  = np.mean([r['auc'] for r in march_folds])  if march_folds  else float('nan')
april_auc  = np.mean([r['auc'] for r in april_folds])  if april_folds  else float('nan')

print(f"\n{'='*65}")
print(f"WALK-FORWARD SUMMARY -- Model B (8 ps_net features)")
print(f"{'='*65}")
print(f"  Mean AUC:   {mean_auc:.4f}")
print(f"  Std AUC:    {std_auc:.4f}")
print(f"  Min AUC:    {min_auc:.4f}")
print(f"  Max AUC:    {max_auc:.4f}")
print(f"  Folds > 0.505: {n_above_random}/{len(fold_results)}")
print(f"  March AUC:  {march_auc:.4f}")
print(f"  April AUC: {april_auc:.4f}")
print(f"  Regime gap: {abs(march_auc - april_auc):.4f}")

# STEP 4: Feature importance on all data
X_all = full_df[PS_NET_FEATURES].values.astype(np.float32)
y_all = full_df['signed_return_label'].values.astype(np.int8)
final_model = lgb.LGBMClassifier(**LGBM_PARAMS)
final_model.fit(X_all, y_all)
fi = pd.DataFrame({
    'feature': PS_NET_FEATURES,
    'gain': final_model.booster_.feature_importance(importance_type='gain')
}).sort_values('gain', ascending=False)
fi['pct'] = fi['gain'] / fi['gain'].sum() * 100

print(f"\nFeature importance (all 20 days):")
for _, row in fi.iterrows():
    bar = "#" * int(row['pct'] / 3)
    print(f"  {row['feature']:<35}: {row['pct']:>5.1f}% {bar}")

# STEP 5: Final verdict
print(f"\n{'='*65}")
print(f" MODEL B VALIDATION -- 20 DAYS WALK-FORWARD")
print(f"{'='*65}")
print(f" Features:        8 ps_net/ps_delta")
print(f" Mean AUC:        {mean_auc:.4f}")
print(f" Std AUC:         {std_auc:.4f}")
print(f" Folds > random:  {n_above_random}/{len(fold_results)}")
print(f" March AUC:       {march_auc:.4f}")
print(f" April AUC:       {april_auc:.4f}")
print(f" Regime gap:      {abs(march_auc-april_auc):.4f}")
print(f"{'='*65}")

if mean_auc >= 0.520 and n_above_random >= 4:
    print(f" VERDICT: STRONG -- ps_net signal generalizes across regimes")
elif mean_auc >= 0.510 and n_above_random >= 3:
    print(f" VERDICT: CONFIRMED WEAK SIGNAL ({mean_auc:.4f})")
    print(f" Missing amplifier: T&S aggressive flow (P2b)")
elif mean_auc >= 0.505:
    print(f" VERDICT: MARGINAL ({mean_auc:.4f})")
    print(f" P2b T&S data is the required next step")
else:
    print(f" VERDICT: NO GENERALIZABLE SIGNAL ({mean_auc:.4f})")
    print(f" ps_net on single day was overfitting")
print(f"{'='*65}")

# Save
results_path = BASE_DIR / "output" / "model_b_validation_20days.json"
with open(results_path, 'w') as f:
    json.dump({
        'fold_results': fold_results,
        'mean_auc': float(mean_auc), 'std_auc': float(std_auc),
        'march_auc': float(march_auc), 'april_auc': float(april_auc),
        'feature_importance': fi.to_dict('records'),
    }, f, indent=2)
print(f"\nSaved: {results_path}")
