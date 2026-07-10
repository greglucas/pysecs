# Contributing to pysecs

Contributions are welcome! The full contributing guide lives in the
documentation: https://greglucas.github.io/pysecs/development/

## Quick start

```bash
git clone https://github.com/greglucas/pysecs
cd pysecs
pip install -e ".[dev]"
pre-commit install
```

Run the test suite (including the physics-invariant tests):

```bash
pytest
```

Run the benchmarks before and after performance-related changes:

```bash
python benchmarks/bench_secs.py
```

Build the documentation:

```bash
pip install -e ".[doc]"
cd docs && make html
```

Please open an issue at https://github.com/greglucas/pysecs/issues for
bugs or feature discussions.
