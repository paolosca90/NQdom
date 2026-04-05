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

def _parse_sierra_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trasforma il formato export Sierra Chart Time & Sales in formato canonico.
    Mappatura:
      ts    = Date + " " + Time
      price = Last
      size  = Volume
      side  = derivato da BidVolume/AskVolume (v. regole sotto)
    """
    # Costruisci ts da date + time
    df["ts"] = df["date"].astype(str) + " " + df["time"].astype(str)

    # price e size
    df["price"] = pd.to_numeric(df["last"], errors="coerce")
    df["size"]  = pd.to_numeric(df["volume"], errors="coerce")

    # Converti volumi per le regole side
    bv = pd.to_numeric(df["bidvolume"], errors="coerce").fillna(0)
    av = pd.to_numeric(df["askvolume"], errors="coerce").fillna(0)

    # Regole side:
    # - AskVolume > 0 AND BidVolume == 0 -> "buy"  (lifted Ask)
    # - BidVolume > 0 AND AskVolume == 0 -> "sell" (hit Bid)
    # - entrambi > 0: AskVol > BidVol -> "buy", BidVol > AskVol -> "sell", uguali -> skip
    # - entrambi == 0 -> skip
    conditions = [
        (av > 0) & (bv == 0),   # buy
        (bv > 0) & (av == 0),   # sell
        (av > bv),               # buy ( AskVol dominates )
        (bv > av),               # sell( BidVol dominates )
    ]
    choices   = ["buy", "sell", "buy", "sell"]
    side_mask = np.select(conditions, choices, default="_skip_")

    df["side"] = side_mask

    # Scarta righe
    rows_total = len(df)
    df = df[df["side"] != "_skip_"].copy()
    rows_discarded = rows_total - len(df)

    # Righe con ts/price/size null
    before_nulls = len(df)
    df = df.dropna(subset=["ts", "price", "size"])
    rows_nulls = before_nulls - len(df)

    total_discarded = rows_discarded + rows_nulls
    if total_discarded > 0:
        print(f"      [P2b] Sierra format: {total_discarded:,} righe scartate"
        f" (ambigue/invalide={rows_discarded:,}, null={rows_nulls:,})")

    if df.empty:
        return df

    # Parsa timestamp — formato Sierra: "2026/2/27 00:01:25.226"
    df["ts_dt"] = pd.to_datetime(df["ts"], format="mixed", utc=False)
    df = df.sort_values("ts_dt").reset_index(drop=True)

    return df[["ts", "ts_dt", "price", "size", "side"]]


def load_trades(trades_path: Path):
    """
    Carica i trades. Accetta DUE formati:

    Formato canonico (già usato dal codice):
      ts, price, size, side
      side = 'buy' (Aggressor = Market Buy -> Puts pressure on ASK)
      side = 'sell' (Aggressor = Market Sell -> Puts pressure on BID)

    Formato Sierra Chart Time & Sales export:
      Date, Time, Open, High, Low, Last, Volume,
      NumberOfTrades, BidVolume, AskVolume
      (mappatura automatica -> canonico)
    """
    try:
        df = pd.read_csv(trades_path)
        df.columns = [c.lower() for c in df.columns]

        # Rileva formato
        canonical_cols = {"ts", "price", "size", "side"}
        sierra_cols   = {"date", "time", "last", "volume", "bidvolume", "askvolume"}

        has_canonical = canonical_cols.issubset(set(df.columns))
        has_sierra    = sierra_cols.issubset(set(df.columns))

        if has_canonical and not has_sierra:
            # --- Formato canonico ---
            print("[P2b] Detected canonical trades format")
            if df["ts"].dtype == object:
                df["ts"] = df["ts"].str.replace(" UTC", "", regex=False)
            df["ts_dt"] = pd.to_datetime(df["ts"], format="mixed", utc=False)
            df = df.sort_values("ts_dt").reset_index(drop=True)

        elif has_sierra:
            # --- Formato Sierra Chart export ---
            print("[P2b] Detected Sierra Chart export format")
            df = _parse_sierra_format(df)

        else:
            missing = (canonical_cols | sierra_cols) - set(df.columns)
            raise ValueError(
                f"Formato trades non riconosciuto. Colonne mancanti: {missing}\n"
                f"Formato canonico richiesto : {sorted(canonical_cols)}\n"
                f"Formato Sierra Chart export : {sorted(sierra_cols)}"
            )

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


# ─── Self-test ────────────────────────────────────────────────────────────────
# Esegui direttamente:  python vps_phase2b_data_fusion.py --selftest
# Oppure importa e chiama:  from vps_phase2b_data_fusion import _parse_sierra_format
#
# Mappatura Sierra Chart Export → Formato Canonico
# ──────────────────────────────────────────────────────────────────────────────
# Input Sierra (righe di esempio):
#   Date,        Time,           Open,    High,     Low,      Last,     Vol,  #Trades, BidVol, AskVol
#   2026/2/27,   00:01:25.226,   25212.75, 25216.00, 25212.75, 25212.75,  1,    1,       1,      0
#   2026/2/27,   00:08:37.360,   25223.50, 25226.75, 25223.50, 25223.50,  1,    1,       1,      0
#   2026/2/27,   00:11:10.558,   25236.75, 25236.75, 25232.25, 25236.75,  1,    1,       0,      1
#
# Output canonico:
#   ts                          price      size  side
#   2026-02-27 00:01:25.226     25212.75   1     sell   (BidVol=1 > AskVol=0)
#   2026-02-27 00:08:37.360     25223.50   1     sell   (BidVol=1 > AskVol=0)
#   2026-02-27 00:11:10.558     25236.75   1     buy    (AskVol=1 > BidVol=0)
#
# Edge case residui:
#   - NumberOfTrades > 1: la riga NON viene espansa in micro-trade;
#     viene assunta come singola trade aggregata (size=Volume, side=dominante).
#     Questa è un'approssimazione: il trade aggregator reale dovrebbe redistribuire
#     Volume / NumberOfTrades in micro-trade equi-weight.
#   - BidVol == AskVol: riga scartata (side ambiguo).
#   - BidVol == AskVol == 0: riga scartata (nessun volume registrato).
#   - Formato timestamp Sierra "2026/2/27" vs canonico "2026-02-27": gestito da
#     pd.to_datetime(..., format="mixed") che accetta entrambi.
