#!/usr/bin/env python3
"""Regenerate every released result from scratch (no LLM key required).

Runs the primary scheduling benchmark and the routing contrast, writing all
CSV/JSON artefacts into data/. Deterministic given the fixed seeds.
"""
import importlib, time

def main():
    t0 = time.time()
    print("[1/2] scheduling (primary benchmark) ...")
    importlib.import_module("creoh_scheduling").run(outdir="data")
    print("[2/2] routing (contrast benchmark) ...")
    importlib.import_module("creoh_routing").run(seeds=30, n=25, budget=80, outdir="data")
    print(f"done in {time.time()-t0:.1f}s; artefacts in data/")

if __name__ == "__main__":
    main()
