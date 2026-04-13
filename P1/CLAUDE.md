# P1 — Binary Parser

<claude-mem-context>
# Recent Activity

### Apr 5, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #3087 | 12:39 PM | ✅ | P2b Path Alignment: OUTPUT_TS Remains, OUTPUT → output | ~306 |
| #3084 | 12:37 PM | ✅ | VPS Path Alignment: OUTPUT_TS → output_ts | ~184 |
| #3065 | 12:24 PM | 🔵 | VPS already has INPUT_TS/ directory, not OUTPUT_TS/ | ~300 |
| #3063 | 12:16 PM | ✅ | Updated P1/main.py batch runner call site for run_phase2b | ~203 |
| #3062 | " | ✅ | Modified run_phase2b() to use OUTPUT_TS/by_day/ path | ~323 |
| #3061 | 12:15 PM | 🟣 | Created split_sierra_trades_by_day.py preprocessing script | ~493 |

### Apr 10, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #3549 | 1:17 PM | 🟣 | P1 local runner created with UTC time filtering | ~258 |
| #3548 | 1:16 PM | 🔵 | Phase 1 binary format reverse-engineered | ~278 |
| #3547 | " | 🔵 | NQdom pipeline architecture: 6-phase processing pipeline | ~273 |
| #3546 | " | 🔵 | Project structure discovered: NQdom with multiple phases | ~235 |

### Apr 11, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #4037 | 9:11 PM | 🔵 | Phase 3-6 function signatures and data flow | ~338 |
| #4036 | " | 🔵 | Regular Trading Hours filtering for market data | ~341 |
| #4033 | 9:10 PM | 🔵 | Phase 1 main.py multi-phase batch orchestrator | ~336 |
</claude-mem-context>

## Script

`depth_parser.py` — core parsing module
`main.py` — batch runner (P1-P6 inline)

## Key Functions

`records_to_csv_stream(fh, writer)` — streaming, no filter (used for full parse)
`records_to_csv_stream_filtered(fh, writer, sh, sm, eh, em)` — streaming WITH UTC filter
`_decode_sierra_datetime(raw_value)` — Sierra microseconds → UTC datetime

## Time Filter

UTC window 13:40–19:50 applied via `records_to_csv_stream_filtered`:
- P1_START_UTC_HOUR = 13, P1_START_UTC_MIN = 40
- P1_END_UTC_HOUR = 19, P1_END_UTC_MIN = 50
- EDT = UTC-4 in March–November

## Binary Format

- Header: 64 bytes (SCDD magic)
- Record: 24 bytes (int64 dt, uint8 cmd, uint8 flags, uint16 num_orders, float32 price, uint32 qty, uint32 reserved)
- Epoch: 1899-12-30 (Sierra Chart convention)
