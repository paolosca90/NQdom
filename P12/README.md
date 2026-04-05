# P12 — Sierra Chart Live Bridge (Live Service)

**Script:** `vps_phase12_sierra_bridge.py`

## Cosa fa
Server TCP low-latency che comunica azioni RL a Sierra Chart ACSIL.

## Tipo
**Live service** — non nel batch pipeline. Gira come demone.

## Protocollo
```
Python RL Agent → TCP:8089 → Sierra Chart ACSIL C++
```

## Configurazione Socket
```python
TCP_NODELAY = 1    # Disabilita Nagle → latenza sub-ms
SO_REUSEADDR = 1   # Rebind rapido post-crash
```

## Protocollo JSON
```python
{
    'action': 'allocate',
    'fractions': [0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.4],  # somma = 1.0
    'flatten': False
}
```

## Integrazione ACSIL
Chiama direttamente `sc.BuyEntry`, `sc.SellEntry`, `sc.Flatten` —
**nessuno spreadsheet**, bypass completo per sub-millisecondo.
