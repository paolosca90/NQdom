#!/usr/bin/env python3
"""
Phase 2b — LOB & Time & Sales Data Fusion
=========================================
Integra un feed Time & Sales (es. scaricato da Sierra Chart in txt/csv)
sul file snapshot con pandas.merge_asof.

La logica del LOB:
Quando il 'bid_qty_1' o altri livelli scendono, originariamente euristicava
un 'pull' indifferenziato. L'integrazione di trade_vol_bid (M) permette di
isolare la formula accademica esatta:
  Delta C (Cancellation) = Delta V_total - M

Input  : snapshots.csv (chunked per RAM safety), trades.csv (in RAM)
Output : snapshots_fused.csv (o sovrascrittura di snapshots.csv)
"""

import argparse
import gc
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

CHUNK_SIZE = 250_000

def load_trades(trades_path: Path):
    """
    Carica i trades. Il file deve avere: ['ts', 'price', 'size', 'side']
    side = 'buy' (Aggressor = Market Buy -> Puts pressure on ASK)
    side = 'sell' (Aggressor = Market Sell -> Puts pressure on BID)
    """
    try:
        # Load, map Sierra txt files or generic format
        df = pd.read_csv(trades_path)
        # Rename standardizzazione se necessario
        df.columns = [c.lower() for c in df.columns]
        
        # Converte timestamp high-perf
        if 'ts' in df.columns:
            ts_str = df['ts']
            if ts_str.dtype == object:
                ts_str = ts_str.str.replace(" UTC", "", regex=False)
            df['ts_dt'] = pd.to_datetime(ts_str, format='mixed', utc=False)
        else:
            raise ValueError("Il file trades non ha la colonna 'ts'")
            
        df = df.sort_values('ts_dt').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Errore nel parsing trade file {trades_path}: {e}")
        return pd.DataFrame()


def fuse_chunk(snap_chunk: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Effettua l'asof merge guardando al PASSATO (direction='backward') per
    eliminare ogni possibile simulation-to-reality gap o data leakage.
    I trades vengono prima cumulati, e al tempo T associamo l'ultimo stato
    del cumulativo <= T. Calcolando le differenze, otteniamo il volume
    esatto scambiato nello step intercorso senza mai guardare al futuro.
    """
    # Cast timestamp LOB
    ts_str = snap_chunk['ts'].str.replace(" UTC", "", regex=False)
    snap_chunk['ts_dt'] = pd.to_datetime(ts_str, format='mixed', utc=False)
    
    # 1. Precomputiamo i volumi cumulativi
    t_sell = trades_df[trades_df['side'].str.lower() == 'sell'].copy()
    t_buy  = trades_df[trades_df['side'].str.lower() == 'buy'].copy()
    
    # Evitiamo time duplicati aggregandoli prima del cumulativo
    t_sell = t_sell.groupby('ts_dt')['size'].sum().reset_index()
    t_buy  = t_buy.groupby('ts_dt')['size'].sum().reset_index()
    
    t_sell['cum_sell'] = t_sell['size'].cumsum()
    t_buy['cum_buy']   = t_buy['size'].cumsum()
    
    # 2. AsOf Merge BACKWARD rispetto allo snapshot (solo i trades ESATTAMENTE <= T)
    snap_ref = snap_chunk[['ts_dt']].copy()
    
    # Mergiamo il cum_sell
    merged_s = pd.merge_asof(snap_ref, t_sell[['ts_dt', 'cum_sell']], on='ts_dt', direction='backward')
    merged_s['cum_sell'] = merged_s['cum_sell'].fillna(method='ffill').fillna(0.0)
    
    # Mergiamo il cum_buy
    merged_b = pd.merge_asof(snap_ref, t_buy[['ts_dt', 'cum_buy']], on='ts_dt', direction='backward')
    merged_b['cum_buy'] = merged_b['cum_buy'].fillna(method='ffill').fillna(0.0)
    
    # 3. Differenziamo i cumulativi per trovare lo snap esatto (Delta M)
    snap_chunk['traded_vol_bid'] = merged_s['cum_sell'].diff().fillna(merged_s['cum_sell'].iloc[0])
    snap_chunk['traded_vol_ask'] = merged_b['cum_buy'].diff().fillna(merged_b['cum_buy'].iloc[0])
    
    # Drop ts_dt tmp col
    snap_chunk = snap_chunk.drop(columns=['ts_dt'])
    return snap_chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", type=Path, required=True, help="Path a snapshots.csv")
    ap.add_argument("--trades", type=Path, required=True, help="Path al file Time & Sales (.txt o .csv)")
    ap.add_argument("--output", type=Path, default=None, help="Output path (default: overwrite snapshots)")
    args = ap.parse_args()

    in_file = args.snapshots
    tr_file = args.trades
    out_file = args.output if args.output else in_file
    
    if not in_file.exists():
        print(f"Errore: {in_file} non esiste.")
        return
        
    print(f"[P2b] Loading Time & Sales: {tr_file.name} ...")
    t0 = time.time()
    trades_df = load_trades(tr_file)
    if trades_df.empty:
        print("[P2b] Nessun trade caricato. Fusione abortita.")
        return
    print(f"      {len(trades_df):,} trades memorizzati ({time.time()-t0:.1f}s).")
    
    print(f"[P2b] Fusione Chunked Async (LOB + T&S)...")
    temp_file = in_file.parent / f".fusing_{in_file.name}.tmp"
    
    t_start = time.time()
    total_snaps = 0
    total_matched = 0
    is_first = True
    
    with pd.read_csv(in_file, chunksize=CHUNK_SIZE, dtype=str) as reader:
        for chunk in reader:
            tc_start = time.time()
            fused = fuse_chunk(chunk, trades_df)
            
            # Filtriamo dal DB globale dei trade quelli già associati in modo sicuro
            # (Per evitare di iterare mille volte su milioni di trade al chunk 50)
            if not fused.empty:
                last_ts = pd.to_datetime(fused['ts'].iloc[-1].replace(" UTC", ""), format='mixed', utc=False)
                # Conserviamo solo trades che devono ancora verificarsi futuro
                trades_df = trades_df[trades_df['ts_dt'] > last_ts]
            
            mode = "w" if is_first else "a"
            header = is_first
            fused.to_csv(temp_file, index=False, mode=mode, header=header)
            
            matched = (fused['traded_vol_bid'] > 0).sum() + (fused['traded_vol_ask'] > 0).sum()
            total_matched += matched
            total_snaps += len(fused)
            
            print(f"  Chunk: {len(chunk):,} LOB snaps. Trade events fused: {matched:,} (Time: {time.time()-tc_start:.1f}s)")
            is_first = False
            
    print(f"[P2b] Overwriting target output...")
    # Sostituiamo atomicamente il file
    temp_file.replace(out_file)
    
    print(f"[P2b] DONE. Processati {total_snaps:,} snapshots.")
    print(f"      Snapshot contententi Trade Actions = {total_matched:,}")
    print(f"      Tempo Totale: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
