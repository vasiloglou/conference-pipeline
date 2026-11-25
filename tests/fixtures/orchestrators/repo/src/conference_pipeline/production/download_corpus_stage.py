"""Synthetic shared download_corpus stage."""


def run_download_corpus_stage(ctx, *, resume=False, hooks=None):
    receipt = {"stage": "download_corpus", "run_id": ctx.run_id, "resumed": bool(resume)}
    return {"status": "succeeded", "receipt": receipt, "summary": {"stage": "download_corpus"}}
