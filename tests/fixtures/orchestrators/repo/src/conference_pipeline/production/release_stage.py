"""Synthetic shared release stage."""


def run_release_stage(ctx, *, resume=False, hooks=None):
    receipt = {"stage": "release", "run_id": ctx.run_id, "resumed": bool(resume)}
    return {"status": "succeeded", "receipt": receipt, "summary": {"stage": "release"}}
