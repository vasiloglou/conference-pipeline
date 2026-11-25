from __future__ import annotations

from conference_pipeline.figures.grounding import filter_components, filter_edges, is_grounded


def _slide() -> dict:
    """Deck 02 slide 8 as the blueprint carries it: two sources, no harness."""

    return {
        "title": "Orchestration as generation: automatic sub-agent creation",
        "visible_text": ["AOrchestra automates sub-agent creation", "OpenSage generates topology, toolset and memory"],
        "claims": ["Both systems generate agent structure instead of hand-authoring it."],
        "narration_beats": ["AOrchestra, by Zhang and colleagues, creates sub-agents automatically; OpenSage's LLM writes its own topology, toolsets, and hierarchical memory."],
        "speaker_author_work_mentions": [
            {"title": "AOrchestra: Automating Sub-Agent Creation for Agents", "authors": ["Wei Zhang"]},
            {"title": "OpenSage: Self-programming Agent Generation Engine", "authors": []},
        ],
        "source_plan": [{"label": "SRC-007", "supports": "OpenSage's LLM generates its own topology, toolsets, and hierarchical memory."}],
    }


def test_comparison_items_must_each_be_named_by_the_slide() -> None:
    """Deck 02 slide 8: the brief's 'Hand-designed harness' row was invented."""

    kept, dropped = filter_components(
        ["Hand-designed harness", "AOrchestra", "OpenSage", "Sub-agent creation", "Topology", "Toolset",
         "Hierarchical memory", "Policy update", "Physics simulation"],
        _slide(), depiction="comparison_matrix",
    )
    assert dropped == ["Hand-designed harness", "Policy update", "Physics simulation"]
    assert kept == ["AOrchestra", "OpenSage", "Sub-agent creation", "Topology", "Toolset", "Hierarchical memory"]


def test_a_mechanism_figure_keeps_generic_stages_but_not_foils_or_unknown_names() -> None:
    kept, dropped = filter_components(
        ["User goal", "Observation", "Final report", "Hand-designed harness", "Single-agent baseline",
         "CatFlow slab+adsorbate co-generation", "Jittered patch grid (Walrus)", "Fixed patch grid",
         "Noise injector (AgentHijack-style)"],
        {"narration_beats": ["AOrchestra creates sub-agents. CatFlow co-generates slabs. Walrus jitters the patch grid."]},
        depiction="mechanism_pipeline",
    )
    assert kept == ["User goal", "Observation", "Final report", "CatFlow slab+adsorbate co-generation",
                    "Jittered patch grid (Walrus)"], kept
    assert dropped == ["Hand-designed harness", "Single-agent baseline", "Fixed patch grid",
                       "Noise injector (AgentHijack-style)"], dropped


def test_short_phrases_need_the_phrase_not_just_the_words() -> None:
    """'Policy update' matches 'policy' and 'update' word by word on any RL slide."""

    pool = "the policy is updated after each rollout; rollout error compounds"
    assert not is_grounded("Policy update", pool)
    assert is_grounded("rollout error", pool)
    assert is_grounded("Rollout-error", pool), "hyphens normalise to spaces"
    assert is_grounded("Reward-hacked candidates", "reward hacking is common", strict=True), "stems meet"


def test_long_phrases_pass_on_most_content_words_and_parentheticals_are_ignored() -> None:
    pool = "exact checkers such as numerics and proofs decide the loop; a wet lab experiment is the final checker"
    assert is_grounded("Exact checkers (numerics, proofs)", pool)
    assert is_grounded("Final checker: the wet-lab experiment", pool)
    assert not is_grounded("Learned cellular-response predictor", pool)


def test_edges_survive_only_between_kept_components() -> None:
    kept = ["AOrchestra", "Sub-agent creation", "OpenSage"]
    edges = ["AOrchestra -> Sub-agent creation", "Hand-designed harness -> Sub-agent creation", "OpenSage -> Topology"]
    assert filter_edges(edges, kept) == ["AOrchestra -> Sub-agent creation"]
