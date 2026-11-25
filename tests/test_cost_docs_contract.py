from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


def test_public_docs_reference_canonical_cost_estimator():
    docs = [
        "docs/workflows/pipeline-workflow.md",
        "docs/cli.md",
        "src/conference_pipeline/README.md",
    ]
    for doc in docs:
        text = _read(doc)
        assert "python -m conference_pipeline.costs estimate" in text
        assert "src/conference_pipeline/costs/pricing_catalog.json" in text


def test_specialized_docs_reference_cost_registry():
    docs = [
        "docs/extraction/exhaustive-extraction-guide.md",
        "docs/integrations/gpt-researcher/integration-guide.md",
        "docs/integrations/gpt-researcher/quickstart.md",
        "docs/video-generation.md",
    ]
    for doc in docs:
        text = _read(doc)
        assert "pricing_catalog.json" in text
        assert "conference_pipeline.costs" in text
