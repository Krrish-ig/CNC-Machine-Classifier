import argparse
from pathlib import Path

from cnc_classifier import (
    KNNClassifier,
    Machine,
    MachineFeatures,
    MaintenancePriority,
    load_machines,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a CNC machine and inspect its closest training examples."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "cnc.csv",
        help="Path to the labeled machine CSV file.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors to use.")
    parser.add_argument("--machine-id", default="NEW-MACHINE")
    parser.add_argument("--machine-type", required=True)
    parser.add_argument("--coolant-leak", choices=("Yes", "No"), required=True)
    parser.add_argument(
        "--maintenance-priority",
        choices=tuple(priority.value for priority in MaintenancePriority),
        required=True,
    )
    parser.add_argument("--temperature-c", type=float, required=True)
    parser.add_argument("--vibration-mm-s", type=float, required=True)
    parser.add_argument("--spindle-hours", type=float, required=True)
    parser.add_argument("--oil-particle-ppm", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    machine = Machine(
        machine_id=args.machine_id,
        features=MachineFeatures(
            machine_type=args.machine_type,
            coolant_leak=args.coolant_leak == "Yes",
            maintenance_priority=MaintenancePriority(args.maintenance_priority),
            temperature_c=args.temperature_c,
            vibration_mm_s=args.vibration_mm_s,
            spindle_hours=args.spindle_hours,
            oil_particle_ppm=args.oil_particle_ppm,
        ),
    )
    classifier = KNNClassifier(k=args.k).fit(load_machines(args.data))
    prediction = classifier.predict_with_neighbors(machine)

    print(f"Predicted condition: {prediction.condition.value}")
    print(
        "Neighbor votes: "
        + ", ".join(
            f"{condition.value}={count}"
            for condition, count in prediction.class_votes.items()
        )
    )
    print("Closest machines:")
    for neighbor in prediction.neighbors:
        record = neighbor.record
        print(
            f"  {record.machine.machine_id}: {record.condition.value} "
            f"(similarity={neighbor.similarity:.3f}, distance={neighbor.distance:.3f})"
        )


if __name__ == "__main__":
    main()
