# P5 — CUSUM Sampling

<claude-mem-context>

</claude-mem-context>

## Script

`vps_cusum_sampler.py`

## Algorithm

Adaptive CUSUM con calibrazione empirica:
Pass 1: h = percentile(|Δmid_price|) non nulli (floor=0.5 tick)
Pass 1b: valida emissione su sample 200k righe; se rate > 10% → raddoppia h iterativamente
Pass 2: accumula Δmid_price in S; quando |S| > h → emetti evento → resetta S

## Key Fixes

- Percentile default: 5.0 (era 25.0)
- h threshold floor: 2 × tick_size = 0.5
- Zero-sample guard nel return path
- Buffered row write (non più streaming puro)
- Adaptive calibration: se emissione > 10%, raddoppia h iterativamente su sample