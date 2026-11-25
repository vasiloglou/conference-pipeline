"""Synthetic conference-local component (no RECOMMENDED_FOR annotation)."""


def load_program_spine(path):
    return {"path": str(path), "items": []}


def run_program_spine_discovery(workspace, run_id):
    spine = load_program_spine(workspace)
    return {"status": "succeeded", "receipt": {"run_id": run_id, "spine": spine}, "summary": {}}
