# Example output — MOCK ENGINE, NOT A MEASUREMENT

`mock-runs/<run_id>/` is what one `g4bench run` writes, produced on CPU with the fake
OpenAI-compatible server in `src/g4bench/mockserver.py` (engine `mock`). It exists only to show
the report/CSV/JSON layout. **None of these numbers describe Gemma 4, vLLM, SGLang or any GPU.**
`g4bench analyze` excludes engine `mock` by default; this directory is outside `results/runs/` anyway.
