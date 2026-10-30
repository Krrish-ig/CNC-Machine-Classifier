from dataclasses import dataclass
from collections.abc import Sequence

from cnc_classifier.domain import Machine, MaintenancePriority


@dataclass(frozen=True, slots=True)
class GowerSimilarity:
    """Gower-style distance for the CNC machine's mixed feature types."""

    _numeric_ranges: dict[str, tuple[float, float]]

    @classmethod
    def fit(cls, machines: Sequence[Machine]) -> GowerSimilarity:
        if not machines:
            raise ValueError("At least one machine is required to fit feature ranges")

        numeric_values = {
            "temperature_c": tuple(m.features.temperature_c for m in machines),
            "vibration_mm_s": tuple(m.features.vibration_mm_s for m in machines),
            "spindle_hours": tuple(m.features.spindle_hours for m in machines),
            "oil_particle_ppm": tuple(m.features.oil_particle_ppm for m in machines),
        }
        ranges = {
            name: (min(values), max(values))
            for name, values in numeric_values.items()
        }
        return cls(_numeric_ranges=ranges)

    def distance(self, first: Machine, second: Machine) -> float:
        left = first.features
        right = second.features
        differences = [
            float(left.machine_type != right.machine_type),
            float(left.coolant_leak != right.coolant_leak),
            abs(left.maintenance_priority.rank - right.maintenance_priority.rank)
            / (MaintenancePriority.HIGH.rank - MaintenancePriority.LOW.rank),
        ]
        numeric_pairs = (
            ("temperature_c", left.temperature_c, right.temperature_c),
            ("vibration_mm_s", left.vibration_mm_s, right.vibration_mm_s),
            ("spindle_hours", left.spindle_hours, right.spindle_hours),
            ("oil_particle_ppm", left.oil_particle_ppm, right.oil_particle_ppm),
        )
        for name, left_value, right_value in numeric_pairs:
            minimum, maximum = self._numeric_ranges[name]
            feature_range = maximum - minimum
            if feature_range:
                differences.append(
                    min(abs(left_value - right_value) / feature_range, 1.0)
                )

        return sum(differences) / len(differences)

    def similarity(self, first: Machine, second: Machine) -> float:
        return 1.0 - self.distance(first, second)
