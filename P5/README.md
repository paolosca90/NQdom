# P5 — CUSUM Sampling

**Script:** `vps_cusum_sampler.py`

## Cosa fa
Riduce da milioni a ~migliaia di eventi significativi usando il filtro CUSUM.

## Algoritmo CUSUM

```
Pass 1: h = 25° percentile(|Δmid_price|) non nulli
Pass 2: accumula Δmid_price in S
         quando |S| > h → emetti evento → resetta S
```

CUSUM detecta "shift" persistenti nel prezzo — filtra il rumore browniano.

## Ottimizzazioni
- `np.percentile()` invece di `sort()` → O(n) vs O(n log n)
- `list[list[str]]` invece di `list[dict]` → elimina overhead dizionari

## Input / Output
- **Input:** `features_dom.csv` + `features_dom_agg.csv`
- **Output:** `sampled_events.csv` (~1% degli eventi originali)
- **Checkpoint:** `_checkpoints/p5_sample.done`
