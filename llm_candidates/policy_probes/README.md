# Policy probes

These files are **not** language-model output and are **not** part of the pilot
results.  They are author-written adversarial probes whose only purpose is to
demonstrate, in `test_smoke.py`, that the static policy, the execution timeout
and the deterministic repair policy of `creoh_llm_pilot.py` actually fire.  The
pilot loader ignores this directory (it globs `gen*_*.py` in the parent folder).
