"""Command entrypoint for ``python -m conference_pipeline.costs``."""

from conference_pipeline.costs.estimator import main


if __name__ == "__main__":
    raise SystemExit(main())
