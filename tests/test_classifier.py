import unittest
from dataclasses import replace
from math import nan
from pathlib import Path

from cnc_classifier import (
    Condition,
    GowerSimilarity,
    KNNClassifier,
    Machine,
    MachineFeatures,
    MaintenancePriority,
    TrainingRecord,
    load_machines,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def machine(
    machine_id: str = "test",
    *,
    machine_type: str = "Lathe",
    coolant_leak: bool = False,
    priority: MaintenancePriority = MaintenancePriority.LOW,
    temperature_c: float = 10.0,
    vibration_mm_s: float = 1.0,
    spindle_hours: float = 100.0,
    oil_particle_ppm: float = 2.0,
) -> Machine:
    return Machine(
        machine_id=machine_id,
        features=MachineFeatures(
            machine_type=machine_type,
            coolant_leak=coolant_leak,
            maintenance_priority=priority,
            temperature_c=temperature_c,
            vibration_mm_s=vibration_mm_s,
            spindle_hours=spindle_hours,
            oil_particle_ppm=oil_particle_ppm,
        ),
    )


class GowerSimilarityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = machine("first")
        self.middle = machine("middle", temperature_c=15.0)
        self.last = machine("last", temperature_c=20.0)
        self.similarity = GowerSimilarity.fit([self.first, self.middle, self.last])

    def test_numeric_difference_is_scaled_by_training_range(self) -> None:
        self.assertAlmostEqual(
            self.similarity.distance(self.first, self.middle), 0.5 / 4
        )

    def test_nominal_binary_and_ordinal_attributes_are_handled(self) -> None:
        different_type = replace(
            self.first,
            features=replace(self.first.features, machine_type="Milling"),
        )
        different_leak = replace(
            self.first,
            features=replace(self.first.features, coolant_leak=True),
        )
        different_priority = replace(
            self.first,
            features=replace(
                self.first.features,
                maintenance_priority=MaintenancePriority.HIGH,
            ),
        )

        self.assertAlmostEqual(
            self.similarity.distance(self.first, different_type), 1 / 4
        )
        self.assertAlmostEqual(
            self.similarity.distance(self.first, different_leak), 1 / 4
        )
        self.assertAlmostEqual(
            self.similarity.distance(self.first, different_priority), 1 / 4
        )

    def test_numeric_values_outside_training_range_are_capped(self) -> None:
        extreme = replace(
            self.first,
            features=replace(self.first.features, temperature_c=1_000.0),
        )
        self.assertLessEqual(self.similarity.distance(self.first, extreme), 1.0)
        self.assertGreaterEqual(self.similarity.similarity(self.first, extreme), 0.0)

    def test_non_finite_sensor_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature_c must be finite"):
            machine(temperature_c=nan)


class KNNClassifierTests(unittest.TestCase):
    def test_fit_and_prediction_return_closest_labeled_records(self) -> None:
        normal_machine = machine("normal", temperature_c=10.0)
        critical_machine = machine("critical", temperature_c=20.0)
        records = [
            TrainingRecord(normal_machine, Condition.NORMAL),
            TrainingRecord(critical_machine, Condition.CRITICAL),
        ]
        classifier = KNNClassifier(k=1).fit(records)

        prediction = classifier.predict_with_neighbors(normal_machine)

        self.assertEqual(prediction.condition, Condition.NORMAL)
        self.assertEqual(prediction.class_votes[Condition.NORMAL], 1)
        self.assertEqual(prediction.neighbors[0].record.machine.machine_id, "normal")
        self.assertEqual(prediction.neighbors[0].similarity, 1.0)

    def test_invalid_k_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            KNNClassifier(k=0)
        with self.assertRaises(ValueError):
            KNNClassifier(k=2).fit(
                [TrainingRecord(machine("one"), Condition.NORMAL)]
            )

    def test_unfitted_classifier_cannot_predict(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Call fit"):
            KNNClassifier().predict(machine())


class DatasetTests(unittest.TestCase):
    def test_project_dataset_loads_as_complete_labeled_records(self) -> None:
        records = load_machines(_PROJECT_ROOT / "data" / "cnc.csv")

        self.assertEqual(len(records), 240)
        self.assertEqual(
            {record.condition for record in records},
            {Condition.NORMAL, Condition.WARNING, Condition.CRITICAL},
        )

    def test_dataset_can_train_the_public_classifier_api(self) -> None:
        records = load_machines(_PROJECT_ROOT / "data" / "cnc.csv")
        classifier = KNNClassifier(k=5).fit(records)

        result = classifier.predict_with_neighbors(records[0].machine)

        self.assertIn(result.condition, set(Condition))
        self.assertEqual(
            result.neighbors[0].record.machine.machine_id,
            records[0].machine.machine_id,
        )
        self.assertEqual(len(result.neighbors), 5)


if __name__ == "__main__":
    unittest.main()
