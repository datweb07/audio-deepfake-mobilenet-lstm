"""Backward-compatible baseline efficiency entry point.

Prefer: python -m benchmark.runner --models mobilenetv3_lstm --suite efficiency
"""

from benchmark.runner import main


if __name__ == "__main__":
    main(["mobilenetv3_lstm"], "efficiency", limit=None, warmup=10, runs=50)
