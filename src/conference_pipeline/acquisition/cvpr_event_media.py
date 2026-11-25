"""Compatibility alias: ``conference_pipeline.acquisition.cvpr_event_media`` is now
:mod:`conference_pipeline.acquisition.event_media` (named by capability, not by
conference year; see :mod:`conference_pipeline.acquisition.lane_catalog`).

Importing this path yields the very same module object, so historical code,
tests and monkeypatches written against the old path keep working.
"""

import sys

from conference_pipeline.acquisition import event_media as _module

sys.modules[__name__] = _module
