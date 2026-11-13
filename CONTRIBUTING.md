# Contributing

This is experimental research code; issues and pull requests are welcome, especially reproductions on other GPUs.

- Keep changes small and measurable: a baseline, a metric, an acceptance criterion.
- Performance changes must state hardware, precision, workload, concurrency and the measurement definition, and must
  not change tool calls or spoken output (check with the multi-session gate and an ASR pass).
- Run the CPU tests (`cd v2 && PYTHONPATH=src python -m pytest -q tests/cpu`) before opening a PR.
- Do not commit model weights, audio, traces or experiment outputs (see `.gitignore`).
