"""CIRCE evaluation entry point; does not load a language model."""
from circe.cli import evaluate_main

if __name__ == "__main__":
    raise SystemExit(evaluate_main())
