"""Offline evaluation of the released Qwen-7B pool (Qwen2.5-Coder 7B); never queries Ollama."""
from pathlib import Path
import json
import creoh_llm_pilot as pilot

HERE = Path(__file__).resolve().parent

def run(outdir="data"):
    root = HERE / "ollama_pilot" / "experiment"
    old_dir, old_backbone = pilot.CANDIDATE_DIR, pilot.BACKBONE
    try:
        pilot.CANDIDATE_DIR = str(root / "candidates")
        pilot.BACKBONE = json.loads((root / "backbone.json").read_text())
        return pilot.run(seeds=30, outdir=str(Path(outdir) / "ollama_pilot"))
    finally:
        pilot.CANDIDATE_DIR, pilot.BACKBONE = old_dir, old_backbone

if __name__ == "__main__":
    run(outdir=str(HERE / "data"))
