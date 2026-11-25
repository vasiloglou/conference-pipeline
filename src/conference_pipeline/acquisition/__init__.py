"""Conference-general acquisition primitives.

Modules here provide reusable acquisition pipelines used by per-conference
production runners (`scripts/run_<slug>_production.py`). Conference-specific
code should import from this package rather than redefining the logic.
"""

from conference_pipeline.acquisition.workshop_pipeline import (
    run_workshop_discovery_and_download,
    slugify_venueid,
)

__all__ = [
    "run_workshop_discovery_and_download",
    "slugify_venueid",
]
