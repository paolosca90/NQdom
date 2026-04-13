# P2b — Data Fusion (LOB + Time & Sales)

<claude-mem-context>
# Recent Activity

### Apr 5, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #3164 | 2:22 PM | 🔵 | VPS-Optimized Phase Scripts Across Full Pipeline | ~310 |
| #3121 | 12:51 PM | 🔴 | Date Format Normalization: Slash to Dash Conversion | ~239 |
| #3118 | 12:49 PM | 🔴 | Case-Insensitive Key Lookup Fix for CamelCase Sierra Columns | ~284 |
| #3115 | 12:47 PM | 🔴 | Complete Whitespace Normalization in CSV Row Processing | ~249 |
| #3114 | " | 🔴 | Fixed Sierra Format Detection with Leading Spaces in Column Names | ~200 |
| #3107 | 12:46 PM | 🔴 | Fixed NameError: epilog Function Call Replaced with Parentheses | ~202 |
| #3105 | 12:45 PM | 🔴 | NameError: '__' Not Defined in VPS Deployment | ~220 |
| #3095 | 12:42 PM | ✅ | Test Mode Visual Indicator Added | ~173 |
| #3094 | " | 🔴 | Fixed max_rows Parameter Wiring in main() | ~189 |
| #3093 | " | 🔴 | split_contract() Call Missing max_rows Parameter | ~167 |
| #3088 | 12:39 PM | 🟣 | Test Mode with --max-rows Limit Added | ~236 |
| #3087 | " | ✅ | P2b Path Alignment: OUTPUT_TS Remains, OUTPUT → output | ~306 |
| #3085 | 12:37 PM | 🔴 | Multiple OUTPUT_TS Path References Need Alignment | ~290 |
| #3065 | 12:24 PM | 🔵 | VPS already has INPUT_TS/ directory, not OUTPUT_TS/ | ~300 |
| #3061 | 12:15 PM | 🟣 | Created split_sierra_trades_by_day.py preprocessing script | ~493 |

### Apr 10, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #3689 | 4:50 PM | 🔵 | Deep LOB Notebook Architecture Context Discovered | ~286 |
| #3604 | 3:18 PM | 🔴 | P4 streaming CSV fix eliminates OOM on 8.4M-row dataset | ~396 |

### Apr 12, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #4080 | " | 🔴 | P2b Sierra Chart CSV leading spaces bug identified and fixed | ~350 |
</claude-mem-context>

## Script

`vps_phase2b_data_fusion.py`

## Perché Serve

Quando `bid_qty_1` scende, non sappiamo se è:
- **ΔC** (Cancellation) — ordine cancellato
- **ΔM** (Market Order) — ordine eseguito

**Formula:** `ΔC = ΔV_total − M` (dove M = trade volume da Time & Sales)

## Algoritmo

`pd.merge_asof` backward (solo trades passati, mai future):
1. Per ogni snapshot LOB, trova l'ultimo trade avvenuto ≤ T
2. Calcola `traded_vol_bid` (sell volume) e `traded_vol_ask` (buy volume) come differenze cumulative
3. Questo distingue cancellation (ΔV senza trade) da execution (ΔV con trade)

## Formato Trades Accettati

1. **Formato canonico:** `ts, price, size, side`
2. **Sierra Chart export:** `Date, Time, Open, High, Low, Last, Volume, NumberOfTrades, BidVolume, AskVolume`
   - Side derivato da BidVolume/AskVolume: AskVol>0,BidVol=0 → buy; BidVol>0,AskVol=0 → sell

## Bug Fix (Apr 12, 2026)

Sierra Chart CSV export contiene **leading spaces** nei nomi colonna (`" Time"`, `" Volume"`, ecc.).
`pd.read_csv()` standard NON rimuove gli spaces iniziali, quindi la detection del formato Sierra falliva.

**Fix:**
```python
df = pd.read_csv(trades_path, skipinitialspace=True)
df.columns = [c.lower().strip() for c in df.columns]
```

## Checkpoint

`_checkpoints/p2b_fusion.done`