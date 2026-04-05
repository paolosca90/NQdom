"""
Sierra Chart .depth Binary Parser - Phase 1

Binary format (derived from reverse engineering):
  Header (64 bytes):
    0-3:   uint32  FileTypeUniqueHeaderID ("SCDD")
    4-7:   uint32  HeaderSize (64)
    8-11:  uint32  RecordSize (24)
    12-15: uint32  Version
    16-63: 48 bytes reserved

  Record (24 bytes):
    0-7:   int64   DateTime (microseconds since 1899-12-30)
    8:     uint8   Command (0-7)
    9:     uint8   Flags
    10-11: uint16  NumOrders
    12-15: float32 Price
    16-19: uint32  Quantity
    20-23: uint32  Reserved
"""

import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO

# Binary format constants
HEADER_SIZE = 64
RECORD_SIZE = 24
# Header layout (64 bytes):
#   0-3:   uint32  FileTypeUniqueHeaderID ("SCDD")
#   4-7:   uint32  HeaderSize (64)
#   8-11:  uint32  RecordSize (24)
#   12-15: uint32  Version
#   16-63: 48 bytes reserved
HEADER_FORMAT = "<4s III 48s"  # Little-endian: 4+4+4+4+48 = 64 bytes

# File identification
MAGIC_HEADER = b"SCDD"

# Command codes
CMD_NO_COMMAND = 0
CMD_CLEAR_BOOK = 1
CMD_ADD_BID_LEVEL = 2
CMD_ADD_ASK_LEVEL = 3
CMD_MODIFY_BID_LEVEL = 4
CMD_MODIFY_ASK_LEVEL = 5
CMD_DELETE_BID_LEVEL = 6
CMD_DELETE_ASK_LEVEL = 7

COMMAND_NAMES = {
    CMD_NO_COMMAND: "NO_COMMAND",
    CMD_CLEAR_BOOK: "CLEAR_BOOK",
    CMD_ADD_BID_LEVEL: "ADD_BID_LEVEL",
    CMD_ADD_ASK_LEVEL: "ADD_ASK_LEVEL",
    CMD_MODIFY_BID_LEVEL: "MODIFY_BID_LEVEL",
    CMD_MODIFY_ASK_LEVEL: "MODIFY_ASK_LEVEL",
    CMD_DELETE_BID_LEVEL: "DELETE_BID_LEVEL",
    CMD_DELETE_ASK_LEVEL: "DELETE_ASK_LEVEL",
}

# Flags
FLAG_END_OF_BATCH = 0x01


@dataclass
class DepthHeader:
    """Parsed header from a .depth file."""
    file_type_id: bytes
    header_size: int
    record_size: int
    version: int
    reserved: bytes
    file_size: int


@dataclass
class DepthRecord:
    """Single event record from a .depth file."""
    record_index: int
    datetime_raw: int
    command_code: int
    flags: int
    num_orders: int
    price: float
    quantity: int
    reserved: int


class DepthParseError(Exception):
    """Raised when parsing fails due to invalid data."""
    pass


def _decode_sierra_datetime(raw_value: int) -> datetime:
    """
    Convert Sierra Chart datetime (microseconds since 1899-12-30) to UTC datetime.
    Falls back to epoch if conversion fails.
    """
    if raw_value == 0:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

    try:
        sierra_epoch = datetime(1899, 12, 30, 0, 0, 0, tzinfo=timezone.utc)
        return sierra_epoch + timedelta(microseconds=raw_value)
    except (OverflowError, OSError):
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def read_header(fh: BinaryIO) -> DepthHeader:
    """Read and validate the 64-byte header from a .depth file."""
    raw = fh.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise DepthParseError(
            f"Header truncated: read {len(raw)} bytes, expected {HEADER_SIZE}"
        )

    # Unpack: 4s III 48s (file_type, header_size, record_size, version, reserved)
    try:
        data = struct.unpack("<4s III 48s", raw)
    except struct.error as e:
        raise DepthParseError(f"Header unpacking failed: {e}")

    file_type_id = data[0]
    file_size = fh.seek(0, 2)  # Get actual file size

    return DepthHeader(
        file_type_id=file_type_id,
        header_size=data[1],
        record_size=data[2],
        version=data[3],
        reserved=data[4],
        file_size=file_size,
    )


def validate_header(header: DepthHeader) -> list[str]:
    """
    Validate header fields and return list of warnings.
    Raises DepthParseError on critical failures.
    """
    warnings_list = []

    # Check magic header
    if header.file_type_id != MAGIC_HEADER:
        raise DepthParseError(
            f"Invalid magic header: {header.file_type_id!r} != {MAGIC_HEADER!r}"
        )

    # Check header size
    if header.header_size != HEADER_SIZE:
        raise DepthParseError(
            f"Invalid header size: {header.header_size} != {HEADER_SIZE}"
        )

    # Check record size
    if header.record_size != RECORD_SIZE:
        raise DepthParseError(
            f"Invalid record size: {header.record_size} != {RECORD_SIZE}"
        )

    # Check file length
    data_bytes = header.file_size - HEADER_SIZE
    if data_bytes < 0:
        raise DepthParseError(
            f"File too small: {header.file_size} bytes < {HEADER_SIZE} header"
        )

    remainder = data_bytes % RECORD_SIZE
    if remainder != 0:
        warnings_list.append(
            f"File size mismatch: {data_bytes} data bytes not divisible by "
            f"{RECORD_SIZE} (remainder: {remainder})"
        )

    return warnings_list


def read_records(fh: BinaryIO) -> tuple[list[DepthRecord], list[str]]:
    """
    Read all 24-byte records from the file.
    Layout: int64 datetime, uint8 cmd, uint8 flags, uint16 num_orders,
            float32 price, uint32 quantity, uint32 reserved
    Returns (records, warnings).
    """
    records = []
    warnings_list = []
    index = 0
    unknown_commands = set()

    # Seek to start of records
    fh.seek(HEADER_SIZE)

    # Record format: q (datetime) + B (cmd) + B (flags) + H (num_orders) + f (price) + I (qty) + I (reserved)
    RECORD_FORMAT = "<q BB H f I I"

    while True:
        raw = fh.read(RECORD_SIZE)
        if len(raw) == 0:
            break  # EOF

        if len(raw) < RECORD_SIZE:
            warnings_list.append(
                f"Truncated record at index {index}: "
                f"read {len(raw)} bytes, expected {RECORD_SIZE}"
            )
            break

        try:
            data = struct.unpack(RECORD_FORMAT, raw)
        except struct.error as e:
            warnings_list.append(f"Unpack error at index {index}: {e}")
            break

        record = DepthRecord(
            record_index=index,
            datetime_raw=data[0],
            command_code=data[1],
            flags=data[2],
            num_orders=data[3],
            price=data[4],
            quantity=data[5],
            reserved=data[6],
        )

        if record.command_code not in COMMAND_NAMES:
            unknown_commands.add(record.command_code)

        records.append(record)
        index += 1

    # Report unknown commands
    for cmd in sorted(unknown_commands):
        warnings_list.append(f"Unknown command code: {cmd}")

    return records, warnings_list


def records_to_csv_stream(
    fh_in: BinaryIO,
    writer,
    progress_every: int = 500_000,
) -> tuple[int, list[str]]:
    """
    Stream-parse a .depth file and write CSV rows directly without
    accumulating records in memory. Writes to an already-open CSV writer.
    Returns (record_count, warnings).
    """
    warnings_list = []
    unknown_commands = set()
    index = 0

    # Record format: q (datetime) + B (cmd) + B (flags) + H (num_orders) + f (price) + I (qty) + I (reserved)
    RECORD_FORMAT = "<q BB H f I I"

    fh_in.seek(HEADER_SIZE)

    rec_dict = {}

    while True:
        raw = fh_in.read(RECORD_SIZE)
        if len(raw) == 0:
            break

        if len(raw) < RECORD_SIZE:
            warnings_list.append(
                f"Truncated record at index {index}: "
                f"read {len(raw)} bytes, expected {RECORD_SIZE}"
            )
            break

        try:
            data = struct.unpack(RECORD_FORMAT, raw)
        except struct.error as e:
            warnings_list.append(f"Unpack error at index {index}: {e}")
            break

        command_code = data[1]
        if command_code not in COMMAND_NAMES:
            unknown_commands.add(command_code)

        dt_utc = _decode_sierra_datetime(data[0])
        rec_dict["record_index"] = index
        rec_dict["datetime_raw"] = data[0]
        rec_dict["datetime_utc"] = dt_utc.strftime("%Y-%m-%d %H:%M:%S.%f") + " UTC"
        rec_dict["command_code"] = command_code
        rec_dict["command_name"] = COMMAND_NAMES.get(command_code, f"UNKNOWN_{command_code}")
        rec_dict["flags"] = data[2]
        rec_dict["end_of_batch"] = bool(data[2] & FLAG_END_OF_BATCH)
        rec_dict["num_orders"] = data[3]
        rec_dict["price"] = round(data[4], 4)
        rec_dict["quantity"] = data[5]
        writer.writerow(rec_dict)
        index += 1

        if index % progress_every == 0:
            print(f"    ... parsed {index:,} records")

    for cmd in sorted(unknown_commands):
        warnings_list.append(f"Unknown command code: {cmd}")

    return index, warnings_list


def records_to_csv_rows(records: list[DepthRecord]) -> list[dict]:
    """Convert records to CSV-friendly dictionaries."""
    rows = []
    for rec in records:
        dt_utc = _decode_sierra_datetime(rec.datetime_raw)
        rows.append({
            "record_index": rec.record_index,
            "datetime_raw": rec.datetime_raw,
            "datetime_utc": dt_utc.strftime("%Y-%m-%d %H:%M:%S.%f") + " UTC",
            "command_code": rec.command_code,
            "command_name": COMMAND_NAMES.get(rec.command_code, f"UNKNOWN_{rec.command_code}"),
            "flags": rec.flags,
            "end_of_batch": bool(rec.flags & FLAG_END_OF_BATCH),
            "num_orders": rec.num_orders,
            "price": round(rec.price, 4),
            "quantity": rec.quantity,
        })
    return rows


def count_by_command(records: list[DepthRecord]) -> dict[str, int]:
    """Count records by command name."""
    counts = {name: 0 for name in COMMAND_NAMES.values()}
    for rec in records:
        name = COMMAND_NAMES.get(rec.command_code, f"UNKNOWN_{rec.command_code}")
        counts[name] = counts.get(name, 0) + 1
    return counts


def count_end_of_batch(records: list[DepthRecord]) -> int:
    """Count records with FLAG_END_OF_BATCH set."""
    return sum(1 for rec in records if rec.flags & FLAG_END_OF_BATCH)


def parse_depth_file(filepath: Path) -> tuple[list[DepthRecord], DepthHeader, list[str]]:
    """
    Main parsing function. Reads and validates a .depth file.
    Returns (records, header, warnings).
    """
    with open(filepath, "rb") as fh:
        header = read_header(fh)
        warnings_list = validate_header(header)
        records, record_warnings = read_records(fh)
        warnings_list.extend(record_warnings)

    return records, header, warnings_list