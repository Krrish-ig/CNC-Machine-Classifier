import csv
from pathlib import Path

from cnc_classifier.domain import (
    Condition,
    Machine,
    MachineFeatures,
    MaintenancePriority,
    TrainingRecord,
)

_REQUIRED_COLUMNS = (
    "machine_id",
    "machine_type",
    "coolant_leak",
    "maintenance_priority",
    "temperature_c",
    "vibration_mm_s",
    "spindle_hours",
    "oil_particle_ppm",
    "condition",
)


def _parse_row(row: dict[str | None, str | None], line_number: int) -> TrainingRecord:
    if None in row:
        raise ValueError(f"Line {line_number} has more columns than the CSV header")

    def value(column: str) -> str:
        raw_value = row.get(column)
        if raw_value is None or not raw_value.strip():
            raise ValueError(f"Line {line_number} has an empty {column!r} value")
        return raw_value.strip()

    try:
        leak_value = value("coolant_leak").lower()
        if leak_value not in {"yes", "no"}:
            raise ValueError("coolant_leak must be Yes or No")

        features = MachineFeatures(
            machine_type=value("machine_type"),
            coolant_leak=leak_value == "yes",
            maintenance_priority=MaintenancePriority(value("maintenance_priority")),
            temperature_c=float(value("temperature_c")),
            vibration_mm_s=float(value("vibration_mm_s")),
            spindle_hours=float(value("spindle_hours")),
            oil_particle_ppm=float(value("oil_particle_ppm")),
        )
        return TrainingRecord(
            machine=Machine(machine_id=value("machine_id"), features=features),
            condition=Condition(value("condition")),
        )
    except ValueError as error:
        raise ValueError(f"Invalid CSV record on line {line_number}: {error}") from error


def load_machines(path: str | Path) -> list[TrainingRecord]:
    """Load and validate labeled machine records from a CSV file."""
    csv_path = Path(path)
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty or has no CSV header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{csv_path} has duplicate CSV column names")

        missing_columns = set(_REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        records = [
            _parse_row(row, line_number)
            for line_number, row in enumerate(reader, start=2)
        ]

    if not records:
        raise ValueError(f"{csv_path} contains no machine records")

    machine_ids = [record.machine.machine_id for record in records]
    if len(machine_ids) != len(set(machine_ids)):
        raise ValueError(f"{csv_path} contains duplicate machine_id values")

    return records
