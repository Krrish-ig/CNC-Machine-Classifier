from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from cnc_classifier.domain import (
    Condition,
    Machine,
    Neighbor,
    Prediction,
    TrainingRecord,
)
from cnc_classifier.similarity import GowerSimilarity


@dataclass(slots=True)
class KNNClassifier:
    k: int = 5
    _records: tuple[TrainingRecord, ...] = field(default=(), init=False, repr=False)
    _similarity: GowerSimilarity | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be at least 1")

    def fit(self, records: Sequence[TrainingRecord]) -> KNNClassifier:
        if not records:
            raise ValueError("At least one training record is required")
        if self.k > len(records):
            raise ValueError("k cannot be larger than the number of training records")

        machine_ids = [record.machine.machine_id for record in records]
        if len(machine_ids) != len(set(machine_ids)):
            raise ValueError("Training records must have unique machine IDs")

        self._records = tuple(records)
        self._similarity = GowerSimilarity.fit(
            [record.machine for record in self._records]
        )
        return self

    def find_similar(
        self, machine: Machine, k: int | None = None
    ) -> tuple[Neighbor, ...]:
        if self._similarity is None:
            raise RuntimeError("Call fit() before finding similar machines")

        count = self.k if k is None else k
        if count < 1:
            raise ValueError("The number of neighbors must be at least 1")
        if count > len(self._records):
            raise ValueError("The number of neighbors exceeds the training set size")

        neighbors = []
        for record in self._records:
            distance = self._similarity.distance(machine, record.machine)
            neighbors.append(
                Neighbor(
                    record=record,
                    distance=distance,
                    similarity=1.0 - distance,
                )
            )
        return tuple(
            sorted(
                neighbors,
                key=lambda neighbor: (
                    neighbor.distance,
                    neighbor.record.machine.machine_id,
                ),
            )[:count]
        )

    def predict_with_neighbors(self, machine: Machine) -> Prediction:
        neighbors = self.find_similar(machine)
        votes = Counter(neighbor.record.condition for neighbor in neighbors)
        class_votes = {condition: votes[condition] for condition in Condition}
        similarity_totals = {
            condition: sum(
                neighbor.similarity
                for neighbor in neighbors
                if neighbor.record.condition == condition
            )
            for condition in Condition
        }
        winner = max(
            Condition,
            key=lambda condition: (class_votes[condition], similarity_totals[condition]),
        )
        return Prediction(condition=winner, neighbors=neighbors, class_votes=class_votes)

    def predict(self, machine: Machine) -> Condition:
        return self.predict_with_neighbors(machine).condition
