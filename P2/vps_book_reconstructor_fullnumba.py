"""
Phase 2: Book Reconstructor - FULLY NUMBA-ACCELERATED (PRACTICAL VERSION)
===========================================================================

Strategy:
1. Read CSV in chunks with pandas
2. Process each chunk with FULLY JIT-compiled function
3. Write snapshots incrementally

The JIT function processes ALL events in a chunk natively - no Python loop overhead.
"""

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

# Command codes
CMD_NO_COMMAND = 0
CMD_CLEAR_BOOK = 1
CMD_ADD_BID_LEVEL = 2
CMD_ADD_ASK_LEVEL = 3
CMD_MODIFY_BID_LEVEL = 4
CMD_MODIFY_ASK_LEVEL = 5
CMD_DELETE_BID_LEVEL = 6
CMD_DELETE_ASK_LEVEL = 7

# Output field names
FIELDNAMES = [
    "ts", "best_bid", "best_ask", "spread", "mid_price",
    "bid_px_1", "bid_px_2", "bid_px_3", "bid_px_4", "bid_px_5",
    "bid_px_6", "bid_px_7", "bid_px_8", "bid_px_9", "bid_px_10",
    "bid_qty_1", "bid_qty_2", "bid_qty_3", "bid_qty_4", "bid_qty_5",
    "bid_qty_6", "bid_qty_7", "bid_qty_8", "bid_qty_9", "bid_qty_10",
    "ask_px_1", "ask_px_2", "ask_px_3", "ask_px_4", "ask_px_5",
    "ask_px_6", "ask_px_7", "ask_px_8", "ask_px_9", "ask_px_10",
    "ask_qty_1", "ask_qty_2", "ask_qty_3", "ask_qty_4", "ask_qty_5",
    "ask_qty_6", "ask_qty_7", "ask_qty_8", "ask_qty_9", "ask_qty_10",
]

MAX_LEVELS = 20
CHUNK_SIZE = 200_000  # Events per chunk - balanced for memory/speed


# ══════════════════════════════════════════════════════════════════════════════
# CORE NUMBA FUNCTIONS - These run ENTIRELY in native code
# ══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def update_level(prices: np.ndarray, qtys: np.ndarray,
                  price: float, qty: float,
                  n_levels: int, ascending: bool) -> int:
    """Update a single price level in sorted book array."""
    # Find existing level
    idx = -1
    for i in range(n_levels):
        if prices[i] == price:
            idx = i
            break

    if qty == 0.0:  # Delete
        if idx >= 0:
            for j in range(idx, n_levels - 1):
                prices[j] = prices[j + 1]
                qtys[j] = qtys[j + 1]
            prices[n_levels - 1] = 0.0
            qtys[n_levels - 1] = 0.0
            return n_levels - 1
        return n_levels
    else:  # Add/update
        if idx >= 0:
            qtys[idx] = qty
            return n_levels
        else:
            insert_pos = n_levels
            for i in range(n_levels):
                if (ascending and price < prices[i]) or \
                   (not ascending and price > prices[i]):
                    insert_pos = i
                    break
            max_l = len(prices)
            new_n = min(n_levels + 1, max_l)
            for j in range(new_n - 1, insert_pos, -1):
                prices[j] = prices[j - 1]
                qtys[j] = qtys[j - 1]
            if insert_pos < max_l:
                prices[insert_pos] = price
                qtys[insert_pos] = qty
            return new_n


@njit(cache=True)
def process_chunk_numba(
    command_codes: np.ndarray,
    prices: np.ndarray,
    quantities: np.ndarray,
) -> tuple:
    """
    FULLY JIT-COMPILED chunk processor.

    Takes raw numpy arrays of events, processes ALL of them in native code,
    returns arrays of snapshot features.

    Returns:
        snapshot_features: (n_snapshots, 40) array of [bid,ask,spread,mid,10bid_px,10bid_q,10ask_px,10ask_q]
        snapshot_timestamps: (n_snapshots,) array of timestamp indices
        n_snapshots: actual number of snapshots
    """
    n_events = len(command_codes)

    # Pre-allocate max possible snapshots (worst case: every event is valid)
    # Features: 4 (bid,ask,spread,mid) + 20 (bid px/qty) + 20 (ask px/qty) = 44
    max_snaps = n_events
    features = np.zeros((max_snaps, 44), dtype=np.float64)
    ts_indices = np.zeros(max_snaps, dtype=np.int64)
    n_snapshots = 0

    # Book state - fixed numpy arrays (Numba can optimize these)
    bid_prices = np.zeros(MAX_LEVELS, dtype=np.float64)
    bid_qtys = np.zeros(MAX_LEVELS, dtype=np.float64)
    ask_prices = np.zeros(MAX_LEVELS, dtype=np.float64)
    ask_qtys = np.zeros(MAX_LEVELS, dtype=np.float64)
    n_bid = 0
    n_ask = 0

    # Process EVERY event in this chunk - all native code, no Python
    for i in range(n_events):
        cmd = command_codes[i]
        price = prices[i]
        qty = quantities[i]

        if cmd == CMD_CLEAR_BOOK:
            bid_prices.fill(0.0)
            bid_qtys.fill(0.0)
            ask_prices.fill(0.0)
            ask_qtys.fill(0.0)
            n_bid = 0
            n_ask = 0
            continue

        if cmd == CMD_NO_COMMAND:
            continue

        # These are the HOT PATH - JIT-compiled to native loop
        if cmd == CMD_ADD_BID_LEVEL:
            n_bid = update_level(bid_prices, bid_qtys, price, float(qty), n_bid, False)
        elif cmd == CMD_ADD_ASK_LEVEL:
            n_ask = update_level(ask_prices, ask_qtys, price, float(qty), n_ask, True)
        elif cmd == CMD_MODIFY_BID_LEVEL:
            n_bid = update_level(bid_prices, bid_qtys, price, float(qty), n_bid, False)
        elif cmd == CMD_MODIFY_ASK_LEVEL:
            n_ask = update_level(ask_prices, ask_qtys, price, float(qty), n_ask, True)
        elif cmd == CMD_DELETE_BID_LEVEL:
            n_bid = update_level(bid_prices, bid_qtys, price, 0.0, n_bid, False)
        elif cmd == CMD_DELETE_ASK_LEVEL:
            n_ask = update_level(ask_prices, ask_qtys, price, 0.0, n_ask, True)

        # Check if valid snapshot
        if n_bid > 0 and n_ask > 0:
            best_bid = bid_prices[0]
            best_ask = ask_prices[0]

            if best_bid < best_ask:  # Valid (not crossed)
                ts_indices[n_snapshots] = i

                # Basic: bid, ask, spread, mid
                features[n_snapshots, 0] = best_bid
                features[n_snapshots, 1] = best_ask
                features[n_snapshots, 2] = best_ask - best_bid
                features[n_snapshots, 3] = (best_bid + best_ask) / 2.0

                # Bid prices (cols 4-13)
                for j in range(10):
                    features[n_snapshots, 4 + j] = bid_prices[j] if j < n_bid else 0.0

                # Bid qtys (cols 14-23)
                for j in range(10):
                    features[n_snapshots, 14 + j] = bid_qtys[j] if j < n_bid else 0.0

                # Ask prices (cols 24-33)
                for j in range(10):
                    features[n_snapshots, 24 + j] = ask_prices[j] if j < n_ask else 0.0

                # Ask qtys (cols 34-43)
                for j in range(10):
                    features[n_snapshots, 34 + j] = ask_qtys[j] if j < n_ask else 0.0

                n_snapshots += 1

    return features[:n_snapshots], ts_indices[:n_snapshots], n_snapshots


# ══════════════════════════════════════════════════════════════════════════════
# RECONSTRUCT FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct(input_path: Path, output_path: Path) -> dict[str, Any]:
    """
    Reconstruct order book snapshots from events.csv using FULLY NUMBA processing.

    Each chunk is processed entirely in JIT-compiled native code.
    This gives 10-50x speedup over pure Python for the event processing loop.
    """
    import time

    stats = {
        'snapshots_generated': 0,
        'clear_book_count': 0,
        'crossed_book_count': 0,
        'first_snapshot_ts': None,
        'last_snapshot_ts': None,
    }

    t_start = time.time()

    # Warm up Numba JIT (first call is slow due to compilation)
    print("    [P2] Warming up Numba JIT...")
    dummy_codes = np.array([CMD_ADD_BID_LEVEL, CMD_DELETE_BID_LEVEL], dtype=np.int32)
    dummy_prices = np.array([100.0, 100.0], dtype=np.float64)
    dummy_qtys = np.array([10.0, 0.0], dtype=np.float64)
    _ = process_chunk_numba(dummy_codes, dummy_prices, dummy_qtys)
    print("    [P2] JIT warmup complete.")

    # Open output file
    with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(FIELDNAMES)

        # Read and process CSV in chunks
        chunk_num = 0
        total_events = 0
        total_snapshots = 0

        for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE,
                                  dtype={'record_index': np.int64,
                                         'command_code': np.int32,
                                         'price': np.float64,
                                         'quantity': np.int64}):
            chunk_num += 1
            t0 = time.time()

            # Extract arrays (zero-copy from pandas)
            command_codes = chunk['command_code'].values.astype(np.int32)
            prices = chunk['price'].values.astype(np.float64)
            quantities = chunk['quantity'].values.astype(np.float64)
            timestamps = chunk['datetime_utc'].values

            # Process ENTIRE chunk in JIT-compiled native code
            snapshot_features, ts_indices, n_snapshots = process_chunk_numba(
                command_codes, prices, quantities
            )

            # Write snapshots for this chunk
            for i in range(n_snapshots):
                ts_idx = ts_indices[i]
                row = [timestamps[ts_idx]]

                # Basic: bid, ask, spread, mid
                row.extend(snapshot_features[i, 0:4].tolist())

                # Bid prices (convert 0 to empty)
                for j in range(10):
                    v = snapshot_features[i, 4 + j]
                    row.append(int(v) if v != 0 else '')

                # Bid qtys
                for j in range(10):
                    v = snapshot_features[i, 14 + j]
                    row.append(int(v) if v != 0 else '')

                # Ask prices
                for j in range(10):
                    v = snapshot_features[i, 24 + j]
                    row.append(int(v) if v != 0 else '')

                # Ask qtys
                for j in range(10):
                    v = snapshot_features[i, 34 + j]
                    row.append(int(v) if v != 0 else '')

                writer.writerow(row)

            total_events += len(chunk)
            total_snapshots += n_snapshots

            t1 = time.time()
            print(f"    Chunk {chunk_num}: {len(chunk):,} events -> {n_snapshots:,} snapshots in {t1-t0:.2f}s")

            # Track first/last timestamps
            if n_snapshots > 0 and stats['first_snapshot_ts'] is None:
                stats['first_snapshot_ts'] = timestamps[ts_indices[0]]
            if n_snapshots > 0:
                stats['last_snapshot_ts'] = timestamps[ts_indices[-1]]

            stats['snapshots_generated'] = total_snapshots

    t_end = time.time()

    # Count events for stats
    df_count = pd.read_csv(input_path, usecols=['command_code'])
    stats['clear_book_count'] = int(np.sum(df_count['command_code'].values == CMD_CLEAR_BOOK))

    print(f"\n    [P2] Total: {total_events:,} events -> {total_snapshots:,} snapshots in {t_end-t_start:.1f}s")
    print(f"    [P2] Throughput: {total_events / (t_end - t_start):,.0f} events/sec")

    return stats


def print_report(stats: dict[str, Any]) -> None:
    """Print the Phase 2 terminal report."""
    print("\n" + "=" * 60)
    print("BOOK RECONSTRUCTOR - PHASE 2 (FULLY NUMBA)")
    print("=" * 60)

    print(f"\n[Snapshots]")
    print(f"  Generated            : {stats['snapshots_generated']:,}")
    print(f"  CLEAR_BOOK events   : {stats['clear_book_count']:,}")
    print(f"  Crossed book events : {stats['crossed_book_count']:,}")

    print(f"\n[TIME RANGE]")
    print(f"  First snapshot : {stats['first_snapshot_ts'] or 'N/A'}")
    print(f"  Last snapshot  : {stats['last_snapshot_ts'] or 'N/A'}")

    print("\n" + "=" * 60)
    print("Phase 2 completed.")
    print("=" * 60 + "\n")
