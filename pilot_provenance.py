"""Distinguish original generation provenance from the corrected offline replay.

Only the current evaluator sources are distributed in this package. Historical
source hashes identify the fixed release-e archive in which feedback was made.
"""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parent
GENERATION_REFERENCE = {
    "release": "e",
    "commit": "c5a4fc997dfd5b90fb5d45b13879f083adeaedf5",
    "doi": "10.5281/zenodo.23219351",
    "evaluator_sha256": {
        "creoh_llm_pilot.py": "7aa7309eafca7b15960cf817ac36a4cf706567ae1a203d4a547fef8c321e250b",
        "creoh_scheduling.py": "3629ffcb2ec802994b884235d164b5c61c2ac8797322b8fd238bdf4bfb48f87d",
        "creoh_routing.py": "826ec9c2e83f5e212d489c0a5294b637177eedffc569abe8e9b19615f64819bd",
    },
}


def verify_replay(pilot_name, backbone, original):
    assert backbone["evaluator_sha256"] == GENERATION_REFERENCE["evaluator_sha256"]
    replay_dir = ROOT / "data" / pilot_name
    replay = json.loads((replay_dir / "llm_pilot_metadata.json").read_text())
    current_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in GENERATION_REFERENCE["evaluator_sha256"]}
    assert replay["replay_evaluator_sha256"] == current_hashes
    assert replay["backbone"] == backbone
    for key in ("summary", "execution_status", "programs_generated",
                "programs_admitted", "executions", "repair_rate",
                "feasible_before_repair_rate", "feasible_after_repair_rate",
                "per_generation_best"):
        assert replay[key] == original[key], f"Replay differs: {pilot_name}/{key}"
    original_dir = ROOT / pilot_name / "experiment" / "evaluation_gen3"
    def records(directory):
        with (directory / "llm_pilot_executions.csv").open() as stream:
            return [{k: v for k, v in row.items() if k != "runtime_ms"}
                    for row in csv.DictReader(stream)]
    assert records(replay_dir) == records(original_dir)
    return {"generation_reference": GENERATION_REFERENCE,
            "replay_evaluator_sha256": current_hashes,
            "replay_scientific_results_match_generation_record": True}
