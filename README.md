# CNC Machine Classifier

A small, dependency-free CNC condition classifier built with a Gower-style mixed-attribute distance and k-nearest neighbors (k-NN).

## Features and similarity

The CSV columns used for comparison are `machine_type` (nominal), `coolant_leak` (binary), `maintenance_priority` (ordinal: Low < Medium < High), and `temperature_c`, `vibration_mm_s`, `spindle_hours`, and `oil_particle_ppm` (numeric). `machine_id` identifies a record and `condition` is its training label; neither is used as a feature.

Numeric differences are divided by each feature's range in the training data. Constant numeric features are omitted from the average, and differences beyond a training range are capped at one. Categorical and ordinal differences are normalized to the same 0–1 scale. Similarity is one minus the average distance.

The classifier returns the majority condition among the nearest `k` records. A vote tie is resolved by the total similarity of each tied class among those neighbors. The prediction also includes the neighbors and vote counts so maintenance decisions can be reviewed alongside the supporting examples.

On one deterministic, stratified 48-record holdout from the included dataset, the default classifier correctly classified 33 records (68.8%). This is only an initial check; validate on representative, time-separated production data before operational use, and treat predictions as decision support rather than a replacement for engineering judgment.

## Run a prediction

Install/sync the project with [uv](https://docs.astral.sh/uv/), then provide the new machine's readings:

```powershell
uv sync
uv run python main.py --machine-type Milling --coolant-leak No --maintenance-priority Medium --temperature-c 60 --vibration-mm-s 1.2 --spindle-hours 5000 --oil-particle-ppm 12
```

Use `--k` to change the number of neighbors (default: 5), or `--data` to select another labeled CSV file with the same columns.

## Python API

```python
from cnc_classifier import KNNClassifier, Machine, MachineFeatures, MaintenancePriority, load_machines

training_records = load_machines("data/cnc.csv")
classifier = KNNClassifier(k=5).fit(training_records)
new_machine = Machine(
    machine_id="CNC-NEW",
    features=MachineFeatures(
        machine_type="Milling",
        coolant_leak=False,
        maintenance_priority=MaintenancePriority.MEDIUM,
        temperature_c=60.0,
        vibration_mm_s=1.2,
        spindle_hours=5000,
        oil_particle_ppm=12.0,
    ),
)

prediction = classifier.predict_with_neighbors(new_machine)
print(prediction.condition.value)
for neighbor in prediction.neighbors:
    print(neighbor.record.machine.machine_id, neighbor.record.condition.value, neighbor.similarity)
```

Run the test suite with `uv run python -m unittest discover -s tests -v`.