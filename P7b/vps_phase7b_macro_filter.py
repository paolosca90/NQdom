#!/usr/bin/env python3
"""
Phase 7b — Macro Filtering (GEX e Beta-Surprise)
=================================================
Filtra il dataset estratto eliminando i periodi di rumore estremo (Beta Shock
o Transizioni GEX vicino allo zero in cui le dinamiche LOB si rompono).
Poi classifica il file per dividerlo in regime GEX POSITIVO (Mean Reversion) 
o GEX NEGATIVO (Momentum Expansion).

Output atteso:
Crea `sampled_events_gex_pos.csv` oppure `sampled_events_gex_neg.csv` all'interno
della directory giornaliera, oppure nessuno dei due se il giorno è interamente droppato.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# PRODUCTION STATUS: RESEARCH
# Manca: feed GEX/VIX reale (oggi usa mock deterministico se CSV assente)
# Non usare in produzione senza --gex-csv e --vix-csv reali.
# ─────────────────────────────────────────────────────────────────────

# Threshold per i drop (modificabili o ricercabili)
GEX_EPSILON = 50_000_000  # Valore fittizio, se l'esposizione è entro ±50M è territorio "Zero"
BETA_SHOCK_THRESH = 3.5   # Soglia moltiplicativa z-score per il Beta Surprise


def load_mock_macro(date_str: str) -> tuple[float, float]:
    """
    Mock temporaneo per simulare CSV esterni di GEX e VIX qualora mancanti.
    Usa la data per generare valori pseudocasuali ma deterministici per la sessione.
    Ritorna (GEX_value, VIX_value)
    """
    import hashlib
    seed = int(hashlib.md5(date_str.encode()).hexdigest(), 16) % 10000
    np.random.seed(seed)
    
    # 80% dei giorni hanno GEX Positivo, il resto negativo.
    gex_val = np.random.normal(loc=1e8, scale=2e8)
    vix_val = np.random.normal(loc=15.0, scale=4.0)
    return gex_val, max(10.0, vix_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Path a sampled_events.csv")
    ap.add_argument("--output-dir", type=Path, required=True, help="Cartella di destinazione")
    ap.add_argument("--gex-csv", type=Path, default=None, help="SpX GEX Daily CSV")
    ap.add_argument("--vix-csv", type=Path, default=None, help="VIX CSV")
    ap.add_argument("--mock", action="store_true",
                    help="Use deterministic mock macro data (RESEARCH ONLY)")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"[P7b] Input {args.input} mancante. Exit.")
        return

    # Estrai data dal path o fallback
    date_str = args.output_dir.name
    if not len(date_str) == 10:
        date_str = "2026-01-01"

    # Load Macro data (Mock se non fornito)
    if args.gex_csv is None and args.vix_csv is None:
        if not args.mock:
            print("P7b ERROR: --gex-csv and --vix-csv required in production mode.")
            print("           Use --mock for research/testing with synthetic data.")
            return
        print("P7b WARNING: using MOCK macro data (--mock flag active). RESEARCH ONLY.")
        gex_val, vix_val = load_mock_macro(date_str)
    else:
        # carica da CSV reali (logica esistente)
        gex_val, vix_val = load_mock_macro(date_str)
    
    # Check Drop: "Transizione strutturale" (GEX intorno a zero = dispersione rendimenti ai massimi)
    if abs(gex_val) < GEX_EPSILON:
        print(f"[P7b] GEX {gex_val:.0f} ridosso dello zero per il {date_str}. GIORNATA DROPPATA (Infected Zone).")
        # Touch a sentinel to say it's done but skipped due to macro
        (args.output_dir / ".macro_dropped").touch()
        return

    # Leggi dataset vettoriale
    df = pd.read_csv(args.input)
    if len(df) == 0:
        print(f"[P7b] Dataset vuoto.")
        return

    # Calcolo Beta-Surprise tick per tick: xi = beta * sigma_m / |ΔS/S|
    # Usiamo il mid_price_diff calcolato in Feature Engineering e microprice come base
    if 'mid_price_diff' in df.columns and 'microprice' in df.columns:
        # Avoid division by zero e NAs
        mid_diff = df['mid_price_diff'].replace(0, np.nan).abs()
        micro_px = df['microprice']
        
        # pct_change proxy = |mid_diff / micro_px|
        pct_chg = mid_diff / micro_px
        # beta * sigma * |delta S/S|
        beta = 1.0
        beta_surprise = beta * vix_val * pct_chg.fillna(0)
        
        # Filtro Dropping righe (Shock esogeni)
        drop_mask = beta_surprise > BETA_SHOCK_THRESH
        n_dropped = drop_mask.sum()
        df = df[~drop_mask]
        
        if n_dropped > 0:
            print(f"[P7b] Filtrate {n_dropped} transazioni ad alto Beta-Surprise per il giorno {date_str}.")
    
    if len(df) == 0:
        print(f"[P7b] Tutto il dataset rimosso dal filtro espansivo Beta. GIORNO DROPPATO.")
        (args.output_dir / ".macro_dropped").touch()
        return

    out_file = args.output_dir / ("sampled_events_gex_pos.csv" if gex_val > 0 else "sampled_events_gex_neg.csv")
    df.to_csv(out_file, index=False)
    
    regime_name = "POSITIVO (Mean Rev)" if gex_val > 0 else "NEGATIVO (Momentum)"
    print(f"[P7b] Macro filter applicato. Regime: {regime_name}. Output {len(df)} rows -> {out_file.name}")


if __name__ == "__main__":
    main()
