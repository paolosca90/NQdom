#!/usr/bin/env python3
"""
Phase 12 — Live Bridge Python <-> Sierra Chart (ACSIL)
=====================================================
Server TCP a latenza ultrabassa per connettere l'agente Actor-Critic RL 
al motore di instradamento degli ordini C++ (Sierra Chart ACSIL).

Il server resta in ascolto passivo in localhost. Sierra Chart stabilisce 
una connessione persistente e riceve il vettore d'azione a ogni iterazione
per agganciarlo direttamente a sc.BuyEntry / sc.SellEntry bypassing Excel.
"""

import socket
import json
import struct
import time
import threading

# ─────────────────────────────────────────────────────────────────────
# PRODUCTION STATUS: PARTIAL
# Server TCP con TCP_NODELAY/SO_REUSEADDR: production-ready.
# Loop __main__: mock con payload sintetico — non collegato a P11/P8.
# Per produzione: rimuovere il mock loop e integrare model.forward().
# ─────────────────────────────────────────────────────────────────────

HOST = '127.0.0.1'
PORT = 8089 # Porta arbitraria per la comunicazione locale

class SierraTCPBridge:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Disabilita l'algoritmo di Nagle per minimizzare la latenza (invia immediatamente! fondamentale in HFT)
        self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Permette il bind rapido post-crash
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.client_conn = None
        self.running = False
        
    def start(self):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.running = True
        print(f"[Bridge] TCP Server in ascolto su {self.host}:{self.port} (TCP_NODELAY attivo)")
        
        # Thread parallelo per accettare connessioni da ACSIL (non blocca inferenza)
        self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.accept_thread.start()
        
    def _accept_loop(self):
        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                print(f"[Bridge] Sierra Chart Study ACSIL connesso da: {addr}")
                self.client_conn = conn
            except Exception as e:
                if self.running:
                    print(f"Errore Socket Accept: {e}")
                
    def send_action_vector(self, actions_dict: dict):
        """
        Codifica le azioni per Sierra. 
        actions_dict: {'action': 'allocate', 'fractions': [0.1, 0.5, ...], 'flatten': False}
        """
        if not self.client_conn:
            return # Nessun ACSIL collegato, agiamo in dry-run
            
        try:
            payload = json.dumps(actions_dict).encode('utf-8')
            # Pack strutturale: lunghezza fissa header + payload
            msg = struct.pack('<I', len(payload)) + payload
            self.client_conn.sendall(msg)
        except Exception as e:
            print(f"[Bridge] Disconnessione ACSIL rilevata: {e}")
            self.client_conn = None

    def close(self):
        self.running = False
        if self.client_conn:
            self.client_conn.close()
        self.server_socket.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true",
                        help="Run with synthetic mock payloads (RESEARCH ONLY, default=False)")
    args = parser.parse_args()

    bridge = SierraTCPBridge()
    bridge.start()

    try:
        if args.mock:
            print("WARNING: Bridge running in MOCK mode. Synthetic payloads only.")
            while True:
                time.sleep(1)
                mock_action = {
                    "epoch_time": time.time(),
                    "flatten_all": False,
                    "mo_size": 2,
                    "limit_bids": [{"depth": 1, "size": 5}, {"depth": 3, "size": 10}],
                    "limit_asks": [{"depth": 2, "size": 8}]
                }
                bridge.send_action_vector(mock_action)
        else:
            print("Bridge running in PRODUCTION mode. Waiting for real model integration.")
            print("Integrate model.forward() here before using in live trading.")
            while True:
                time.sleep(1)  # production: solo keepalive, nessun mock payload

    except KeyboardInterrupt:
        bridge.close()
        print("Bridge Terminata.")
