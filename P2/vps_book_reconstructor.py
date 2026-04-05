"""
Phase 2: Book Reconstructor (Memory-Optimized, Numba-Accelerated)
Reads events.csv in chunks, reconstructs bid/ask book, emits snapshots.csv.
"""

import csv
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from numba import njit

# Command codes (same as depth_parser.py)
CMD_NO_COMMAND = 0
CMD_CLEAR_BOOK = 1
CMD_ADD_BID_LEVEL = 2
CMD_ADD_ASK_LEVEL = 3
CMD_MODIFY_BID_LEVEL = 4
CMD_MODIFY_ASK_LEVEL = 5
CMD_DELETE_BID_LEVEL = 6
CMD_DELETE_ASK_LEVEL = 7

# CSV field names
FIELDNAMES = [
    "ts",
    "best_bid", "best_ask", "spread", "mid_price",
    "bid_px_1", "bid_px_2", "bid_px_3", "bid_px_4", "bid_px_5",
    "bid_px_6", "bid_px_7", "bid_px_8", "bid_px_9", "bid_px_10",
    "bid_qty_1", "bid_qty_2", "bid_qty_3", "bid_qty_4", "bid_qty_5",
    "bid_qty_6", "bid_qty_7", "bid_qty_8", "bid_qty_9", "bid_qty_10",
    "ask_px_1", "ask_px_2", "ask_px_3", "ask_px_4", "ask_px_5",
    "ask_px_6", "ask_px_7", "ask_px_8", "ask_px_9", "ask_px_10",
    "ask_qty_1", "ask_qty_2", "ask_qty_3", "ask_qty_4", "ask_qty_5",
    "ask_qty_6", "ask_qty_7", "ask_qty_8", "ask_qty_9", "ask_qty_10",
]

# How many events to process before printing progress
PROGRESS_EVERY = 200_000

# How many rows to read per chunk (balance: too small = overhead, too big = memory)
CHUNK_SIZE = 100_000

# PERF: fixed-size arrays for numba book representation
MAX_LEVELS = 20


# ── numba-accelerated book update primitives ───────────────────────────────────

@njit(cache=True)
def update_book_level(prices: np.ndarray, qtys: np.ndarray,
                      price: float, qty: float,
                      n_levels: int, ascending: bool) -> int:
    """
    Update a single price level in sorted book array.
    Returns updated n_levels count.
    - qty == 0: delete level
    - qty > 0: add or update
    Array is kept sorted (ascending for asks, descending for bids).
    """
    # find existing level
    idx = -1
    for i in range(n_levels):
        if prices[i] == price:
            idx = i
            break

    if qty == 0.0:
        if idx >= 0:
            # remove: shift left
            for j in range(idx, n_levels - 1):
                prices[j] = prices[j+1]
                qtys[j]   = qtys[j+1]
            prices[n_levels-1] = 0.0
            qtys[n_levels-1]   = 0.0
            return n_levels - 1
        return n_levels
    else:
        if idx >= 0:
            qtys[idx] = qty
            return n_levels
        else:
            # insert new level maintaining sort order
            insert_pos = n_levels
            for i in range(n_levels):
                if (ascending and price < prices[i]) or \
                   (not ascending and price > prices[i]):
                    insert_pos = i
                    break
            # shift right to make room (cap at MAX_LEVELS)
            max_l = len(prices)
            new_n = min(n_levels + 1, max_l)
            for j in range(new_n - 1, insert_pos, -1):
                prices[j] = prices[j-1]
                qtys[j]   = qtys[j-1]
            if insert_pos < max_l:
                prices[insert_pos] = price
                qtys[insert_pos]   = qty
            return new_n


@njit(cache=True)
def compute_snapshot_features(bid_p: np.ndarray, bid_q: np.ndarray,
                               ask_p: np.ndarray, ask_q: np.ndarray,
                               n_bid: int, n_ask: int, n_levels: int = 5) -> tuple:
    """Compute mid, spread, top-N imbalance from book arrays."""
    if n_bid == 0 or n_ask == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0  # mid, spread, bid_vol, ask_vol, imbalance

    mid     = (bid_p[0] + ask_p[0]) / 2.0
    spread  = ask_p[0] - bid_p[0]

    bid_vol = 0.0
    ask_vol = 0.0
    for i in range(min(n_levels, n_bid)):
        bid_vol += bid_q[i]
    for i in range(min(n_levels, n_ask)):
        ask_vol += ask_q[i]

    denom = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / denom if denom > 0 else 0.0

    return mid, spread, bid_vol, ask_vol, imbalance


class BookState:
    """Holds the current bid/ask book state using numpy arrays for numba acceleration."""

    def __init__(self) -> None:
        # PERF: numpy arrays instead of dicts — enables numba JIT acceleration
        self.bid_prices = np.zeros(MAX_LEVELS, dtype=np.float64)
        self.bid_qtys   = np.zeros(MAX_LEVELS, dtype=np.float64)
        self.ask_prices = np.zeros(MAX_LEVELS, dtype=np.float64)
        self.ask_qtys   = np.zeros(MAX_LEVELS, dtype=np.float64)
        self._n_bid = 0
        self._n_ask = 0

    def clear(self) -> None:
        self.bid_prices.fill(0.0); self.bid_qtys.fill(0.0)
        self.ask_prices.fill(0.0); self.ask_qtys.fill(0.0)
        self._n_bid = 0; self._n_ask = 0

    def add_bid(self, price: float, quantity: int) -> None:
        self._n_bid = update_book_level(
            self.bid_prices, self.bid_qtys, price, float(quantity),
            self._n_bid, ascending=False)

    def add_ask(self, price: float, quantity: int) -> None:
        self._n_ask = update_book_level(
            self.ask_prices, self.ask_qtys, price, float(quantity),
            self._n_ask, ascending=True)

    def modify_bid(self, price: float, quantity: int) -> None:
        self._n_bid = update_book_level(
            self.bid_prices, self.bid_qtys, price, float(quantity),
            self._n_bid, ascending=False)

    def modify_ask(self, price: float, quantity: int) -> None:
        self._n_ask = update_book_level(
            self.ask_prices, self.ask_qtys, price, float(quantity),
            self._n_ask, ascending=True)

    def delete_bid(self, price: float, record_index: int) -> bool:
        new_n = update_book_level(
            self.bid_prices, self.bid_qtys, price, 0.0,
            self._n_bid, ascending=False)
        if new_n == self._n_bid:
            print(f"  WARNING: DELETE_BID_LEVEL at index {record_index} price {price} not in book", file=sys.stderr)
            return False
        self._n_bid = new_n
        return True

    def delete_ask(self, price: float, record_index: int) -> bool:
        new_n = update_book_level(
            self.ask_prices, self.ask_qtys, price, 0.0,
            self._n_ask, ascending=True)
        if new_n == self._n_ask:
            print(f"  WARNING: DELETE_ASK_LEVEL at index {record_index} price {price} not in book", file=sys.stderr)
            return False
        self._n_ask = new_n
        return True

    def is_valid(self) -> bool:
        return self._n_bid > 0 and self._n_ask > 0

    def best_bid(self) -> float | None:
        return self.bid_prices[0] if self._n_bid > 0 else None

    def best_ask(self) -> float | None:
        return self.ask_prices[0] if self._n_ask > 0 else None

    def n_bid_levels(self) -> int:
        return self._n_bid

    def n_ask_levels(self) -> int:
        return self._n_ask


def build_snapshot(book: BookState, ts: str) -> dict[str, Any]:
    """Build a snapshot row from current book state (numpy-accelerated)."""
    best_bid = book.best_bid()
    best_ask = book.best_ask()

    if best_bid is None or best_ask is None:
        return {}

    spread = best_ask - best_bid
    mid_price = (best_bid + best_ask) / 2.0

    row: dict[str, Any] = {
        "ts": ts,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "mid_price": mid_price,
    }

    # PERF: arrays are already sorted (bids descending, asks ascending) — O(1) slice vs heapq
    n_bid = book.n_bid_levels()
    n_ask = book.n_ask_levels()

    for i in range(10):
        if i < n_bid:
            row[f"bid_px_{i+1}"] = book.bid_prices[i]
            row[f"bid_qty_{i+1}"] = int(book.bid_qtys[i])
        else:
            row[f"bid_px_{i+1}"] = None
            row[f"bid_qty_{i+1}"] = None

    for i in range(10):
        if i < n_ask:
            row[f"ask_px_{i+1}"] = book.ask_prices[i]
            row[f"ask_qty_{i+1}"] = int(book.ask_qtys[i])
        else:
            row[f"ask_px_{i+1}"] = None
            row[f"ask_qty_{i+1}"] = None

    return row


def iter_csv_chunks(filepath: Path, chunk_size: int = CHUNK_SIZE) -> Iterator[list[dict]]:
    """
    Read events.csv in chunks to avoid loading the entire file into memory.
    Yields lists of rows (dicts).
    """
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        chunk: list[dict] = []
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def reconstruct(input_path: Path, output_path: Path) -> dict[str, Any]:
    """
    Read events.csv in chunks, reconstruct book state, write snapshots.csv.
    Writes snapshots incrementally to avoid memory exhaustion.
    Returns stats dict for reporting.
    """
    book = BookState()
    stats = {
        "snapshots_generated": 0,
        "clear_book_count": 0,
        "crossed_book_count": 0,
        "first_snapshot_ts": None,
        "last_snapshot_ts": None,
    }

    total_bid_levels_sum = 0
    total_ask_levels_sum = 0
    total_records_processed = 0

    # Write snapshots incrementally — file stays open for the whole run
    with open(output_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
        writer.writeheader()

        # Process events.csv in chunks
        for chunk_num, chunk in enumerate(iter_csv_chunks(input_path)):
            for row in chunk:
                record_index = int(row["record_index"])
                command_code = int(row["command_code"])
                price = float(row["price"])
                quantity = int(row["quantity"])
                ts = row["datetime_utc"]

                if command_code == CMD_CLEAR_BOOK:
                    book.clear()
                    stats["clear_book_count"] += 1
                    continue

                if command_code == CMD_NO_COMMAND:
                    continue

                if command_code == CMD_ADD_BID_LEVEL:
                    book.add_bid(price, quantity)
                elif command_code == CMD_ADD_ASK_LEVEL:
                    book.add_ask(price, quantity)
                elif command_code == CMD_MODIFY_BID_LEVEL:
                    book.modify_bid(price, quantity)
                elif command_code == CMD_MODIFY_ASK_LEVEL:
                    book.modify_ask(price, quantity)
                elif command_code == CMD_DELETE_BID_LEVEL:
                    book.delete_bid(price, record_index)
                elif command_code == CMD_DELETE_ASK_LEVEL:
                    book.delete_ask(price, record_index)
                else:
                    continue

                if not book.is_valid():
                    continue

                snapshot = build_snapshot(book, ts)

                best_bid = snapshot.get("best_bid")
                best_ask = snapshot.get("best_ask")
                if best_bid is not None and best_ask is not None and best_bid >= best_ask:
                    stats["crossed_book_count"] += 1
                    # CRITICAL FIX: skip invalid snapshots — best_bid must be < best_ask
                    continue

                writer.writerow(snapshot)
                stats["snapshots_generated"] += 1

                total_bid_levels_sum += book.n_bid_levels()
                total_ask_levels_sum += book.n_ask_levels()

                if stats["first_snapshot_ts"] is None:
                    stats["first_snapshot_ts"] = ts
                stats["last_snapshot_ts"] = ts

                total_records_processed += 1
                if total_records_processed % PROGRESS_EVERY == 0:
                    print(f"    ... {total_records_processed:,} events processed, "
                          f"{stats['snapshots_generated']:,} snapshots written")

            # End of chunk — let GC clean up references before next chunk
            del chunk

    if stats["snapshots_generated"] > 0:
        stats["avg_bid_levels"] = total_bid_levels_sum / stats["snapshots_generated"]
        stats["avg_ask_levels"] = total_ask_levels_sum / stats["snapshots_generated"]
    else:
        stats["avg_bid_levels"] = 0.0
        stats["avg_ask_levels"] = 0.0

    return stats


def print_report(stats: dict[str, Any]) -> None:
    """Print the Phase 2 terminal report."""
    print("\n" + "=" * 60)
    print("BOOK RECONSTRUCTOR - PHASE 2 REPORT")
    print("=" * 60)

    print(f"\n[Snapshots]")
    print(f"  Generated            : {stats['snapshots_generated']:,}")
    print(f"  CLEAR_BOOK events   : {stats['clear_book_count']:,}")
    print(f"  Crossed book events : {stats['crossed_book_count']:,}")

    print(f"\n[TIME RANGE]")
    print(f"  First snapshot : {stats['first_snapshot_ts'] or 'N/A'}")
    print(f"  Last snapshot  : {stats['last_snapshot_ts'] or 'N/A'}")

    avg_bid = stats.get("avg_bid_levels", 0.0)
    avg_ask = stats.get("avg_ask_levels", 0.0)
    print(f"\n[AVG LEVELS]")
    print(f"  Avg bid levels  : {avg_bid:.1f}")
    print(f"  Avg ask levels  : {avg_ask:.1f}")

    print("\n" + "=" * 60)
    print("Phase 2 completed.")
    print("=" * 60 + "\n")
