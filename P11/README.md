# P11 — RL Trade Execution (Live Service)

**Script:** `vps_phase11_rl_execution.py`

## Cosa fa
Agente Actor-Critic con Logistic-Normal policy per la gestione dinamica dell'allocazione.

## Tipo
**Live service** — non nel batch pipeline. Gira come demone.

## Spazio Azione (K=6)

| Azione | Descrizione |
|--------|-------------|
| a₀ | Market Order (fill istantaneo) |
| a₁..a₅ | Limit Orders a 1-5 tick di distanza |
| a₆ | Hold (nessun ordine) |

## Logistic-Normal Distribution
Mappa `R^K → Simplex(K+1)` garantendo Σa = 1.0:
```python
z ~ N(μ, σ)                           # sampling
aₖ = exp(zₖ) / (1 + Σexp(zₖ))        # softmax projection
```

## Dependencies
`torch`, `numpy`

## Integrazione
Riceve segnali da P8, emette azioni verso P12 (Sierra Chart Bridge).
