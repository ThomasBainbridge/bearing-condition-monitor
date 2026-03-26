from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class DatasetFileSpec:
    file_name: str
    label: str
    health_state: str
    fault_type: str | None
    fault_size_in: float | None
    load_id: int
    sensor_location: str = "drive_end"
    sample_rate_hz: float = 12000.0
    outer_race_position: str | None = None

    def to_record(self) -> dict:
        return asdict(self)


def _make_normal_specs(load_ids: Iterable[int]) -> list[DatasetFileSpec]:
    return [
        DatasetFileSpec(
            file_name=f"normal_{load_id}.mat",
            label="normal",
            health_state="normal",
            fault_type=None,
            fault_size_in=None,
            load_id=load_id,
        )
        for load_id in load_ids
    ]


def _make_fault_specs(
    *,
    prefix: str,
    label: str,
    health_state: str,
    fault_type: str,
    fault_sizes_in: Iterable[float],
    load_ids: Iterable[int],
    outer_race_position: str | None = None,
) -> list[DatasetFileSpec]:
    specs: list[DatasetFileSpec] = []

    for fault_size_in in fault_sizes_in:
        size_code = f"{int(round(fault_size_in * 1000)):03d}"
        for load_id in load_ids:
            if outer_race_position is None:
                file_name = f"{prefix}{size_code}_{load_id}.mat"
            else:
                file_name = f"{prefix}{size_code}_{outer_race_position}_{load_id}.mat"

            specs.append(
                DatasetFileSpec(
                    file_name=file_name,
                    label=label,
                    health_state=health_state,
                    fault_type=fault_type,
                    fault_size_in=fault_size_in,
                    load_id=load_id,
                    outer_race_position=outer_race_position,
                )
            )

    return specs


def get_project_file_specs() -> list[DatasetFileSpec]:
    load_ids = [0, 1, 2, 3]
    fault_sizes_in = [0.007, 0.014, 0.021]

    return [
        *_make_normal_specs(load_ids),
        *_make_fault_specs(
            prefix="ir",
            label="inner_race",
            health_state="fault",
            fault_type="inner_race",
            fault_sizes_in=fault_sizes_in,
            load_ids=load_ids,
        ),
        *_make_fault_specs(
            prefix="b",
            label="ball",
            health_state="fault",
            fault_type="ball",
            fault_sizes_in=fault_sizes_in,
            load_ids=load_ids,
        ),
        *_make_fault_specs(
            prefix="or",
            label="outer_race",
            health_state="fault",
            fault_type="outer_race",
            fault_sizes_in=fault_sizes_in,
            load_ids=load_ids,
            outer_race_position="6",
        ),
    ]


def specs_to_dataframe(file_specs: Iterable[DatasetFileSpec]) -> pd.DataFrame:
    return pd.DataFrame([spec.to_record() for spec in file_specs])


def get_raw_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "raw"


def validate_dataset_files(
    file_specs: Iterable[DatasetFileSpec],
    raw_dir: str | Path | None = None,
) -> dict:
    raw_dir = Path(raw_dir) if raw_dir is not None else get_raw_data_dir()
    file_specs = list(file_specs)

    expected_files = [spec.file_name for spec in file_specs]
    present_files = sorted(path.name for path in raw_dir.glob("*.mat")) if raw_dir.exists() else []
    present_set = set(present_files)

    missing_files = [file_name for file_name in expected_files if file_name not in present_set]
    unexpected_files = [file_name for file_name in present_files if file_name not in set(expected_files)]

    return {
        "raw_dir": raw_dir,
        "expected_count": len(expected_files),
        "present_expected_count": len(expected_files) - len(missing_files),
        "missing_count": len(missing_files),
        "missing_files": missing_files,
        "unexpected_count": len(unexpected_files),
        "unexpected_files": unexpected_files,
        "all_present": len(missing_files) == 0,
    }