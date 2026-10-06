import json
from pathlib import Path


def read_engine_plan(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) < 5:
        return data

    header_size = int.from_bytes(data[:4], "little")
    header_end = 4 + header_size
    if header_size < 1 or header_end >= len(data):
        return data
    try:
        header = json.loads(data[4:header_end])
    except (UnicodeDecodeError, json.JSONDecodeError):
        return data
    if not isinstance(header, dict):
        return data
    return data[header_end:]