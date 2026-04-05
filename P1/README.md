# P1 — Parsing Binario

**Script:** `depth_parser.py`
**Runner:** `main.py`

## Cosa fa
Legge i file `.depth` binari di Sierra Chart (formato proprietario) e li converte in `events.csv`.

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

- **Input:** `/opt/depth-dom/input/NQ*/YYYY-MM-DD.depth`
- **Output:** `/opt/depth-dom/output/{date}/events.csv`
- **Checkpoint:** `_checkpoints/p1_parse.done`

## Utilizzo

```bash
# Single day
python3 main.py --days 2026-01-08 --force

# Oppure direttamente
python3 depth_parser.py --input /opt/depth-dom/input/NQ*/2026-01-08.depth --output /opt/depth-dom/output/2026-01-08/events.csv
```

## Output CSV
Ogni riga = un evento di modifica del book (Add/Modify/Delete/Trade).
