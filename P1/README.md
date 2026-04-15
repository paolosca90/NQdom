# P1 — Parsing Binario

**Script:** `depth_parser.py`
**Runner:** `main.py` (anche `run_p1_to_p7_multiday.py`)

## Cosa fa

Legge i file `.depth` binari di Sierra Chart (formato proprietario) e li converte in `events.csv`.
Filtra automaticamente per la finestra UTC 13:40–19:50 (09:40–15:50 ET, EDT=UTC-4).

## Formato File .depth

```
Header (64 byte):
  00-03: Magic "SCDD"
  04-07: HeaderSize = 64
  08-11: RecordSize = 24
  12-15: Version

Record (24 byte):
  00-07: int64  DateTime (microsecondi da 1899-12-30)
  08:    uint8  Command  (0-7)
  09:    uint8  Flags
  10-11: uint16 NumOrders
  12-15: float32 Price
  16-19: uint32 Quantity
```

## 8 Comandi DOM

| Codice | Nome | Effetto |
|--------|------|---------|
| 0 | NO_COMMAND | NOP |
| 1 | CLEAR_BOOK | Reset completo |
| 2 | ADD_BID_LEVEL | Aggiunge livello bid |
| 3 | ADD_ASK_LEVEL | Aggiunge livello ask |
| 4 | MODIFY_BID_LEVEL | Modifica qty bid |
| 5 | MODIFY_ASK_LEVEL | Modifica qty ask |
| 6 | DELETE_BID_LEVEL | Rimuove bid |
| 7 | DELETE_ASK_LEVEL | Rimuove ask |

## Input / Output

- **Input:** `NQdom/INPUT/NQ*/YYYY-MM-DD.depth`
- **Output:** `NQdom/output/{date}/events.csv`
- **Checkpoint:** `_checkpoints/p1_parse.done`
- **Time filter:** 13:40–19:50 UTC (applicato in `records_to_csv_stream_filtered`)

## Utilizzo

```bash
# Single day
python3 NQdom/P1/main.py --days 2026-03-13 --force

# Multi-day (P1-P7)
python3 NQdom/run_p1_to_p7_multiday.py --days 2026-03-13
```

## Output CSV

Ogni riga = un evento di modifica del book (Add/Modify/Delete/Trade).
