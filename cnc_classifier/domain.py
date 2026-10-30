from dataclasses import dataclass
from enum import Enum
from math import isfinite


class MaintenancePriority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

    @property
    def rank(self) -> int:
        return {
            MaintenancePriority.LOW: 0,
            MaintenancePriority.MEDIUM: 1,
            MaintenancePriority.HIGH: 2,
        }[self]


class Condition(str, Enum):
    NORMAL = "Normal"
    WARNING = "Warning"
    CRITICAL = "Critical"


@dataclass(frozen=True, slots=True)
class MachineFeatures:
    machine_type: str
    coolant_leak: bool
    maintenance_priority: MaintenancePriority
    temperature_c: float
    vibration_mm_s: float
    spindle_hours: float
    oil_particle_ppm: float

    def __post_init__(self) -> None:
        if not self.machine_type.strip():
            raise ValueError("machine_type must not be empty")
        if not isinstance(self.coolant_leak, bool):
            raise TypeError("coolant_leak must be a bool")
        if not isinstance(self.maintenance_priority, MaintenancePriority):
            raise TypeError("maintenance_priority must be a MaintenancePriority")
        for name, value in (
            ("temperature_c", self.temperature_c),
            ("vibration_mm_s", self.vibration_mm_s),
            ("spindle_hours", self.spindle_hours),
            ("oil_particle_ppm", self.oil_particle_ppm),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class Machine:
    machine_id: str
    features: MachineFeatures

    def __post_init__(self) -> None:
        if not self.machine_id.strip():
            raise ValueError("machine_id must not be empty")


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    machine: Machine
    condition: Condition


@dataclass(frozen=True, slots=True)
class Neighbor:
    record: TrainingRecord
    distance: float
    similarity: float


@dataclass(frozen=True, slots=True)
class Prediction:
    condition: Condition
    neighbors: tuple[Neighbor, ...]
    class_votes: dict[Condition, int]
