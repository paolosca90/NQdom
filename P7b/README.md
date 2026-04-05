# P7b — Filtro Macro (GEX / Beta-Surprise)

**Script:** `vps_phase7b_macro_filter.py`

## Cosa fa
Filtra eventi basandosi su regimi macro e classifica il dataset per regime.

## Cosa Filtra

**GEX Zero Crossing:** Transizioni vicino a GEX=0 — "zone infette" dove le dinamiche LOB si rompono.

**Beta Shock:** Variazioni estreme del beta (market sensitivity).

## Thresholds
- `GEX_EPSILON = 50M` — entro ±50M = territorio Zero
- `BETA_SHOCK_THRESH = 3.5` — z-score moltiplicativo

## Output
- `sampled_events_gex_pos.csv` → Mean Reversion regime
- `sampled_events_gex_neg.csv` → Momentum regime

## Perché Due Modelli
Regimi diversi → distribuzione ritorni diversa → pattern diversi →
due modelli specializzati battono uno generico.

## Nota
Usa mock data se CSV esterni (GEX, VIX) non disponibili.
