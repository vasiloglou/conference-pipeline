"""Synthetic shared analysis stage."""


def run_analysis_stage(ctx, *, resume=False, hooks=None):
    receipt = {"stage": "analysis", "run_id": ctx.run_id, "resumed": bool(resume)}
    return {"status": "succeeded", "receipt": receipt, "summary": {"stage": "analysis"}}
