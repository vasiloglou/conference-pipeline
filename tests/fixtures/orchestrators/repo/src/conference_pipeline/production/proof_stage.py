"""Synthetic shared proof stage."""


def run_proof_stage(ctx, *, resume=False, hooks=None):
    receipt = {"stage": "proof", "run_id": ctx.run_id, "resumed": bool(resume)}
    return {"status": "succeeded", "receipt": receipt, "summary": {"stage": "proof"}}
