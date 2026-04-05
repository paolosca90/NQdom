#!/usr/bin/env python3
"""
Phase 11 — RL Trade Execution & Stochastic Queue Engine
=======================================================
Actor-Critic basato su Logistic-Normal policy per l'HFT.
Trasforma decisioni di esecuzione ottima da classification (Phase 8) in 
partizionamento dinamico su un Simplesso (S_K). Include una simulazione 
stocastica dei Queue Dynamics (FIFO) e attori markoviani.

K=6:
 a_0 = Market Order (Fill istantaneo, paga spread/slippage)
 a_1...a_5 = Limit Orders a diveri Ticks away (1-5)
 a_6 = Inventario trattenuto fuori dal book (Hold outside)
 
Requires: torch, numpy
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# ─────────────────────────────────────────────────────────────────────
# PRODUCTION STATUS: RESEARCH SANDBOX
# ExecutionEnv è un mock con queue dynamics sintetiche.
# Non collegato al flusso P8→P12 reale.
# Dipendenze: torch, numpy
# ─────────────────────────────────────────────────────────────────────

# ── LOGISTIC-NORMAL DISTRIBUTION ──────────────────────────────────────────────

class LogisticNormal(nn.Module):
    """
    Gestisce la mappatura R^K -> Simplex(K+1)
    L'input mu (shape K) ed exp(std) campionano z ~ N(mu, std).
    z viene proiettato sul simplesso:
       a_k = exp(z_k) / (1 + sum(exp(z)))  per k in 1..K
       a_0 = 1 / (1 + sum(exp(z)))
    Questo garantisce SUM = 1.0 e azioni biettive per il log-prob.
    """
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim # (K_limit + 1) logit dimension
    
    def forward(self, mu: torch.Tensor, std: torch.Tensor, deterministic: bool = False):
        if deterministic:
            z = mu
        else:
            dist = Normal(mu, std)
            z = dist.rsample() # Reparameterization trick
            
        # LogSumExp trick per stabilità numerica exp(z)/(1 + sum(exp(z)))
        # z è di shape [Batch, K]
        # Aggiungiamo uno shared zero logit per a_0
        zeros = torch.zeros(z.shape[0], 1, device=z.device)
        z_extended = torch.cat([zeros, z], dim=-1) # Shape: [Batch, K+1]
        
        # Softmax è matematicamente identico alla proiezione standard del Simplesso
        # Mappa z_extended -> (a_0, a_1, ... a_K)
        a = F.softmax(z_extended, dim=-1)
        
        # Calcolo Esatto del Log-Prob (Change of Variables)
        # log P(a) = log N(z | mu, std) - sum(log a_i)
        if not deterministic:
            dist_log_prob = dist.log_prob(z).sum(dim=-1) # Log-prob del Gaussian
            # Il log determinante dello Jacobiano = -sum(log a_i)
            # Aggiungiamo un epsilon per evitare log(0)
            log_a = torch.log(a + 1e-8)
            log_det_jacobian = -log_a.sum(dim=-1)
            
            # Substracting bc variable transformation formula is log P(y) = log P(x) - log |det J|
            log_prob = dist_log_prob + log_det_jacobian
        else:
            log_prob = None
            
        return a, log_prob, z


# ── ACTOR-CRITIC NETWORKS ─────────────────────────────────────────────────────

class ExecutionActorCritic(nn.Module):
    def __init__(self, state_dim: int, num_limits: int = 5):
        super().__init__()
        # Private states + Market states
        # num_limits + 1 (MO) + 1 (Hold) = Total Actions K+2
        # LogisticNormal projection maps K logits -> K+1 actions. 
        # So we need K = num_limits + 1 (MO) logits.
        self.K_logits = num_limits + 1
        
        # ACTOR
        self.actor_base = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(), # Tanh often avoids exploding values in continuous control
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Tanh()
        )
        self.actor_out = nn.Linear(64, self.K_logits)
        
        # Inizializzazione Custom (Domanda 4)
        # Pesi vicini a 0. Bias sbilanciato sull'ultimo logit (Hold Outside)
        # Assumendo logit index 0: K_logits-1 -> ordini sul book
        # Ultimo logit: Mantenere l'inventario.
        nn.init.uniform_(self.actor_out.weight, -1e-4, 1e-4)
        nn.init.constant_(self.actor_out.bias, 0.0)
        # Il bias per l'ultimo logit (Hold) lo forziamo a +3.0 -> exp(3) ~= 20x prob
        self.actor_out.bias.data[-1] = 3.0
        
        # Inizializziamo Log-std per esplorazione
        self.log_std = nn.Parameter(torch.ones(self.K_logits) * 0.5)
        
        # CRITIC
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.logistic_normal = LogisticNormal(self.K_logits)

    def forward(self, state, deterministic=False):
        # Valore di stato
        v = self.critic(state)
        
        # Policy
        actor_features = self.actor_base(state)
        mu = self.actor_out(actor_features)
        
        # Decadimento (l'ottimizzatore modula log_std scendendo per l'annealing)
        std = torch.exp(self.log_std)
        
        action_fracs, log_prob, _ = self.logistic_normal(mu, std, deterministic)
        
        return action_fracs, log_prob, v


# ── STOCHASTIC EXECUTION ENVIRONMENT (MOCK ENGINE) ─────────────────────────────

class ExecutionEnv:
    """
    Mock Stochastic Engine per l'Execution HFT.
    Azioni: [a_MO, a_L1, a_L2, a_L3, a_L4, a_L5, a_HOLD]
    Simula Queue Depletion, Adverse Selection (Tactical Traders) e TWAP Drift.
    """
    def __init__(self, initial_inventory: int = 100, max_steps: int = 60, tick_size: float = 0.25):
        self.initial_inventory = initial_inventory
        self.max_steps = max_steps
        self.tick_size = tick_size
        self.action_dim = 7 # 1(MO) + 5(Limits) + 1(Hold)
        self.num_limits = 5
        
        self.reset()
        
    def reset(self):
        self.step_idx = 0
        self.inventory = self.initial_inventory
        self.mid_price = 4500.00
        self.initial_mark = self.mid_price
        self.realized_pnl = 0.0
        
        # Code private FIFO: per ogni livello (1..5) traccia (size, position)
        # position in coda (1.0 = in fondo, 0.0 = prossimo all'esecuzione)
        self.active_limits = {L: {"size": 0, "queue_pos": 1.0} for L in range(1, self.num_limits + 1)}
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Concatena Market States finti (imbalance) + Private States.
        Private: time, inventory, queue sizes/positions
        """
        # Feature Finte
        market_imbalance = np.random.uniform(-1, 1) # Obi mock
        volatility = np.random.uniform(0.5, 2.0)
        
        # Private
        time_rem = 1.0 - (self.step_idx / self.max_steps)
        inv_rem = self.inventory / self.initial_inventory
        
        limit_states = []
        for L in range(1, self.num_limits + 1):
            limit_states.append(self.active_limits[L]["size"] / self.initial_inventory)
            limit_states.append(self.active_limits[L]["queue_pos"])
            
        state = np.array([time_rem, inv_rem, market_imbalance, volatility] + limit_states, dtype=np.float32)
        return state

    def step(self, action_fracs: np.ndarray):
        """
        action_fracs: vettore simplesso len=7.
        """
        old_inventory = self.inventory
        
        # 1. TRADUZIONE AZIONI IN SIZE FISICA (Arrotondata a quantita minime)
        target_sizes = np.floor(action_fracs * self.inventory).astype(int)
        
        # Assicura che la somma quadrata corrisponda
        diff = self.inventory - np.sum(target_sizes)
        target_sizes[-1] += diff # Metti eventuali resti in HOLD
        
        mo_size = target_sizes[0]
        limit_sizes = target_sizes[1:6]
        held_size = target_sizes[6]
        
        # 2. MARKET ORDER EXECUTION
        if mo_size > 0:
            # Slippage base + spread
            execution_px = self.mid_price - (0.5 * self.tick_size) # Vendita
            self.realized_pnl += mo_size * execution_px
            self.inventory -= mo_size
            
        # 3. GESTIONE FIFO (PENALIZZAZIONI)
        # Se modifichiamo la size di un Limit, ci facciamo cancellare l'ordine dal motore e passiamo in fondo!
        for L in range(1, self.num_limits + 1):
            intended_size = limit_sizes[L-1]
            current_status = self.active_limits[L]
            
            if intended_size != current_status["size"]:
                # Cancellazione/Modifica -> Riavvio FIFO al 100% della coda
                if intended_size > 0:
                    self.active_limits[L] = {"size": intended_size, "queue_pos": 1.0}
                else:
                    self.active_limits[L] = {"size": 0, "queue_pos": 0.0}

        # 4. SIMULAZIONE MERCATO (Gli attori rispondono)
        obi = np.random.uniform(-1, 1) # sbilanciamento del book
        
        # Drift (Strategic Traders)
        self.mid_price += np.random.normal(0, self.tick_size)
        
        # Dinamica della coda (Queue Depletion da Noise & Tactical)
        for L in range(1, self.num_limits + 1):
            if self.active_limits[L]["size"] > 0:
                # La coda scende se i Noise trader sbattono sul nostro livello
                # I Tactical Traders sono *avversi*: se noi abbiamo size grossa, diminuiscono i fill (ci front-runnano)
                adverse_impact = 0.05 * (self.active_limits[L]["size"] / self.initial_inventory)
                base_fill_prob = max(0.01, 1.0 / (L * 2.0)) # I livelli più lontani fittano meno
                
                # Consumo fila
                queue_consumption = np.random.uniform(0.1, 0.4) - adverse_impact
                self.active_limits[L]["queue_pos"] = max(0.0, self.active_limits[L]["queue_pos"] - queue_consumption)
                
                # Se la posizione < 0, l'ordine è eseguito!
                if self.active_limits[L]["queue_pos"] <= 0.0:
                    fill_sz = self.active_limits[L]["size"]
                    # Vendo al bid
                    fill_px = self.mid_price - (L * self.tick_size)
                    self.realized_pnl += fill_sz * fill_px
                    self.inventory -= fill_sz
                    self.active_limits[L] = {"size": 0, "queue_pos": 0.0}
        
        self.step_idx += 1
        done = (self.step_idx >= self.max_steps) or (self.inventory <= 0)
        
        # 5. LIQUIDAZIONE FORZATA A FINE EPISODIO
        if done and self.inventory > 0:
            # Liquidare con pesantissimo slippage per punire il fallimento di esecuzione
            liquidation_px = self.mid_price - (5 * self.tick_size)
            self.realized_pnl += self.inventory * liquidation_px
            self.inventory = 0
            
        # 6. REWARD: IMPLEMENTATION SHORTFALL (Normalizzato)
        # Valore ideale se avessimo venduto tutto all'inizio a mid_price (o best bid)
        ideal_cashflow = self.initial_inventory * (self.initial_mark - (0.5 * self.tick_size))
        
        if done:
            # Shortfall reale (quanto abbiamo perso dal mark iniziale)
            shortfall = self.realized_pnl - ideal_cashflow
            # Normalizziamo su scala 0
            reward = shortfall / (self.initial_inventory * self.tick_size)
        else:
            # Step reward zero (sparse reward), PPO handle accumulo di ritorni
            reward = 0.0
            
        return self._get_state(), reward, done, {}


# ── TEST DELLA DISTRIBUZIONE E DELL'AMBIENTE ────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("WARNING: vps_phase11_rl_execution.py — RESEARCH SANDBOX")
    print("ExecutionEnv uses MOCK queue dynamics.")
    print("Do NOT use in live trading without real environment integration.")
    print("=" * 60)

    print("Test Logistic-Normal Initialization & Action Pipeline...")
    
    # 4 (Market) + 2(Time/Inv) + 5*2(Limit Size/Pos) = 16 state dim
    state_dim = 16 
    
    agent = ExecutionActorCritic(state_dim=state_dim)
    env = ExecutionEnv()
    
    state = env.reset()
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        a_fracs, log_p, val = agent(state_t, deterministic=False)
        a_numpy = a_fracs.squeeze(0).numpy()
    
    print("\nAction Fraction Sampled (Simplex, Sum=1.0):")
    print(f"  MO Fraction    : {a_numpy[0]*100:.1f}%")
    for i in range(1, 6):
         print(f"  Limit {i} Fract : {a_numpy[i]*100:.1f}%")
    # Ci aspettiamo che Hold Outside (ultimo) sia privilegiato dal bias!
    print(f"  Hold Outside   : {a_numpy[-1]*100:.1f}%")
    print("\nLog-Prob computation valid:", log_p is not None and not torch.isnan(log_p).any())
    
    print("\nStepping in Environment...")
    next_s, r, done, info = env.step(a_numpy)
    print("Next State Shape:", next_s.shape)
    print("Inventory Remaining:", env.inventory)
    print("Done:", done)
    
    print("\n[✓] RL Architecture basata su PILASTRO 1 (Logistic-Normal/Bias) e PILASTRO 2 (Mock Engine FIFO) testata.")
