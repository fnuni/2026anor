# Third pilot: Qwen-27B (local tag qwen3.8:27b-mlx)

The exact local Ollama tag is `qwen3.8:27b-mlx`; in the manuscript this pilot is called **Qwen-27B**. Ollama reports 27,781,081,984
parameters, architecture `qwen3_5`, safetensors format and NVFP4 quantization.
The upstream release identity is not inferred from the local tag. Model digest,
full configuration and Ollama version are recorded in `experiment/backbone.json`.

The main campaign uses three generations of six fresh-context requests with
`think=false`, temperature 0.7, top_p 0.9, num_ctx 8192, num_predict 4096,
and seeds 1000 * generation + candidate index. No tools are supplied.
The original base prompt, evaluator and 30 instance seeds are unchanged.
Feedback is drafted by the assisting AI from actual execution outcomes and
training summaries; no OOD metrics are fed back. The same instances inform
feedback and final evaluation, so this is a descriptive adaptive pilot.

## Preliminary technical batch

The first default-thinking response exhausted 4096 tokens without returning
code. The batch was stopped while its second request was pending. Both request
records and the available completed response are preserved under
`preliminary_default_thinking/`, excluded in full from the main campaign.
The switch to `think=false` was made for all 18 main requests before examining
any main-campaign performance. There is no selective replacement of candidates.

## Offline reproduction

From the repository root:

```bash
python3 creoh_qwen27_pilot.py
python3 make_qwen27_macros.py
python3 qwen27_pilot/verify_results.py
python3 qwen27_pilot/verify_statistics.py
```

Offline replay does not require Ollama. Original records remain in `experiment`;
regenerated final tables are written under `data/qwen27_pilot/`.
The original generator is `pilot_ollama.py`; saved requests and responses are
preserved and existing evaluations are never silently overwritten.
