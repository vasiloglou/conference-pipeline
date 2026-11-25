"""Compatibility alias: ``conference_pipeline.acquisition.cvpr_program_spine`` is now
:mod:`conference_pipeline.acquisition.program_spine` (named by capability, not by
conference year; see :mod:`conference_pipeline.acquisition.lane_catalog`).

Importing this path yields the very same module object, so historical code,
tests and monkeypatches written against the old path keep working.
"""

import sys

from conference_pipeline.acquisition import program_spine as _module

sys.modules[__name__] = _module
