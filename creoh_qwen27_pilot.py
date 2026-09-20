"""Offline evaluation of the released Qwen-27B pool (local tag qwen3.8:27b-mlx); never queries Ollama."""
from pathlib import Path
import json
import csv
from collections import Counter
import creoh_llm_pilot as pilot

HERE = Path(__file__).resolve().parent

class LoggedPilot(pilot.LLMProposerPilot):
    """Add raw instance-level metrics without changing evaluation or selection."""
    def _report(self, progs, loaded, per, exec_rows, attempts, feas_pre,
                feas_post, repaired_n, runtimes, pool_sizes, outdir, per_gen=None):
        result = super()._report(progs, loaded, per, exec_rows, attempts, feas_pre,
                                feas_post, repaired_n, runtimes, pool_sizes, outdir, per_gen)
        success = Counter(r['seed'] for r in exec_rows if r['status'] == 'ok')
        evaluated = [seed for seed in range(self.seeds) if success[seed] >= 2]
        with (Path(outdir)/'llm_pilot_raw_metrics.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            metrics = ('f1','f2','f3','p95','hv','ood')
            writer.writerow(['seed','evaluator'] + list(metrics))
            for label, values in per.items():
                assert all(len(values[k]) == len(evaluated) for k in metrics)
                for i, seed in enumerate(evaluated):
                    writer.writerow([seed, label] + [values[k][i] for k in metrics])
        return result


def run(outdir="data"):
    root = HERE / "qwen27_pilot" / "experiment"
    old_dir, old_backbone = pilot.CANDIDATE_DIR, pilot.BACKBONE
    try:
        pilot.CANDIDATE_DIR = str(root / "candidates")
        pilot.BACKBONE = json.loads((root / "backbone.json").read_text())
        return LoggedPilot(seeds=30).run(outdir=str(Path(outdir) / "qwen27_pilot"))
    finally:
        pilot.CANDIDATE_DIR, pilot.BACKBONE = old_dir, old_backbone

if __name__ == "__main__":
    run(outdir=str(HERE / "data"))
