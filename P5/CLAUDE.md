# P5 — CUSUM Sampling

<claude-mem-context>
# Recent Activity

### Apr 6, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #3388 | 8:57 PM | 🔴 | P5 Buffer Refactor Completed: Streaming Write → Buffered Write-After-Guard | ~375 |
| #3387 | 8:56 PM | 🔴 | P5 vps_cusum_sampler.py Broken After Buffer Refactor | ~321 |
| #3386 | 8:55 PM | 🔴 | P5 Sampler Left in Broken State: Buffer Refactor Incomplete | ~292 |
| #3385 | " | 🔄 | P5 Sampler Refactoring: Streaming Write → Buffered Rows | ~258 |
| #3384 | " | ✅ | P5 Module Docstring Updated for New Threshold Behavior | ~123 |
| #3383 | " | ✅ | P5 CLI Percentile Default Aligned to 5.0 | ~110 |
| #3382 | 8:54 PM | 🔴 | P5 Zero-Sample Guard Added in cusum_sample Return Path | ~244 |
| #3381 | " | 🔴 | P5 CUSUM h Threshold Floor Added: 2×tick_size=0.5 | ~236 |
| #3380 | " | ✅ | P5 Percentile Default Changed 25.0 → 5.0 | ~205 |
| #3379 | " | 🔵 | vps_cusum_sampler.py Pre-Fix State: percentile=25.0, No Fail-Gate | ~373 |

### Apr 10, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #3607 | 3:19 PM | 🟣 | P5 CUSUM sampler launched on aligned P3/P4 output | ~374 |
| #3604 | 3:18 PM | 🔴 | P4 streaming CSV fix eliminates OOM on 8.4M-row dataset | ~396 |

### Apr 14, 2026

| ID | Time | T | Title | Read |
|----|------|---|-------|------|
| #4621 | 5:05 PM | 🟣 | New P3→P7 orchestrator with direct module imports for real-time output | ~386 |
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

## Note

**Soglie CUSUM sono per-day** — h viene calcolato dalla distribuzione dei prezzi di ogni singolo giorno. Non usare soglie fisse: ogni giorno ha una volatilità diversa.