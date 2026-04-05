# SHARED — Costanti e Utilities Condivise

## _pipeline_constants.py

Costanti condivise da tutti i componenti del pipeline.

**3 Candidati Triple-Barrier:**
vb_ticks = vertical barrier in TICK-CLOCK updates (NOT seconds).
pt_ticks = profit target in ticks.
sl_ticks = stop loss in ticks.
```python
CANDIDATES = [
    {"vb_ticks": 30,  "pt_ticks": 9.5,  "sl_ticks": 9.8,  "desc": "30t/9.5/9.8"},
    {"vb_ticks": 60,  "pt_ticks": 20.0, "sl_ticks": 20.0, "desc": "60t/20/20"},
    {"vb_ticks": 120, "pt_ticks": 13.0, "sl_ticks": 14.5, "desc": "120t/13/14.5"},
]
```

**Helpers:**
```python
label_filename(vb_ticks, pt_ticks, sl_ticks)   # → "phase7_labels_30ticks_9p5_9p8"
label_filenames()                              # → lista di tutti i 3 nomi
parse_ts_to_ms(ts_str)                         # ISO ts → ms-from-midnight
```

## utils.py

```python
parse_ts_to_ms(ts_str)          # ISO ts → ms-from-midnight
memory_efficient_csv_read()     # np.loadtxt wrapper per file grandi
checksum_csv(path)               # MD5 checksum per regression testing
```

## Note
Questa directory è sincronizzata con VPS (`/opt/depth-dom/`).
Le versioni locali sono la **fonte di verità**.
