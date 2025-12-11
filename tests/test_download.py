import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.download import compute_sha256, download_data


def test_download_creates_files_and_metadata(tmp_path):
    symbol = "TEST"
    timeframe = "15m"
    start = "2024-01-01"
    end = "2024-01-02"
    out_dir = tmp_path / "data" / "raw"

    metadata = download_data(symbol, timeframe, start, end, out_dir=out_dir)

    csv_path = out_dir / symbol / f"{symbol}_{timeframe}.csv"
    meta_path = out_dir / symbol / f"{symbol}_{timeframe}_metadata.json"

    assert csv_path.exists(), "CSV file should be created"
    assert meta_path.exists(), "Metadata file should be created"

    df = pd.read_csv(csv_path)
    required_columns = {"timestamp", "open", "high", "low", "close", "volume"}
    assert required_columns.issubset(set(df.columns))
    assert len(df) > 0

    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    assert timestamps.is_monotonic_increasing
    assert timestamps.is_unique

    loaded_meta = json.loads(meta_path.read_text())
    assert loaded_meta["rows"] == len(df)
    assert loaded_meta["rows"] == metadata["rows"]
    assert loaded_meta["checksum_sha256"] == compute_sha256(csv_path)

    # verify checksum independently
    recalculated = compute_sha256(csv_path)
    assert loaded_meta["checksum_sha256"] == recalculated
