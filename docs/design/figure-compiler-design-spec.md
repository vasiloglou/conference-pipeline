# Figure Compiler — Taxonomy and Design Spec

Status: BUILT (2026-09-09). Phases 0-4 are implemented; this document is
now the design record plus an implementation note (§12) covering where the
build departed from the proposal and why. Grounded in the CVPR 2026
two-deck canary.
Owner: presentation platform.
Code: `src/conference_pipeline/figures/` (the compiler),
`scripts/presentations/generate_figure_specs.py` (spec authoring, paid),
`scripts/presentations/compile_figures.py` (compile, $0),
`tests/test_figure_compiler*.py`, `tests/test_figure_ledger_and_selection.py`.
Run it: `python scripts/run_cvpr2026_production.py presentation --figure-route compiler`.

## 1. Why a compiler, and why a taxonomy first

The canary produced two full decks (49 content slides) with Opus-authored
drawsvg figures. The vision judge's 57 findings were almost entirely geometric —
text overlap (12), box overlap (11), text overflow (10), crowded layout (7),
clipped text (5) — and two redraw rounds moved the failing-slide count from
18→16 and 20→14→13→12. The ICML topic-02 rebuild saw the same oscillation.
An LLM writing absolute coordinates cannot measure a rendered string or see a
connector cross a label; a stronger model has the same blindness.

The second problem is monotony. Both decks use exactly five `visual_role`
archetypes and all five are drawn as boxes-and-arrows:

| visual_role (current) | slides | drawn as |
|---|---|---|
| mechanism_pipeline | 17 | boxes + arrows |
| tradeoff_map | 13 | boxes + arrows |
| comparison_matrix | 11 | boxes in a grid |
| architecture | 7 | boxes + arrows |
| evidence_chart | 1 | boxes with bars |

Yet 11 of the 49 slides carry real numbers in their approved narration beats
(80.2 % tokens pruned / 96.0 % retained / 1.34× speedup; 21.1 % latency
reduction / 1.49×; up to 5× speedup; 84.96 % watermark recovery; 55 %+ of
failures from insufficient exploration), many claims are about the *shape* of a
behaviour (confidence should fall as blur increases; drift accumulates across
denoising steps), and — this being computer vision — many are about images,
video frames, 3-D scenes, masks and modalities.

So the design splits **what** to draw (the LLM's job: entities, relations,
numbers with sources, emphasis, a takeaway) from **where** to draw it (a
deterministic layout compiler with real text metrics and collision-free
placement), and it starts from a taxonomy of figure families, because each
family needs a different layout engine and a different *grounding* rule.

## 2. Principles

1. **Claim first.** Every figure makes exactly one claim legible. The claim
   type (mechanism, comparison, magnitude, shape, structure, sequence, space,
   visual evidence) selects the family; the family selects the engine.
2. **What/where split.** The model never emits coordinates, fonts, colours or
   marker geometry. It emits a validated declarative spec; the compiler owns
   layout, styling and geometry.
3. **Grounding is explicit and rendered.** A figure is `data` (every plotted
   value maps to a numeric-claims ledger row with a source span), `schematic`
   (illustrates a behaviour's shape; no numeric ticks; carries a visible
   "schematic" badge), `hybrid` (schematic layout with data callouts), or
   `evidence_crop` (a licensed paper figure, gated by FIG-07 triage). The
   artifact-truth validator enforces the pairing: numeric axes ⇒ ledger refs;
   schematic ⇒ badge and no numeric ticks.
4. **Correct by construction.** Node sizes come from measured text (brand font
   metrics), connectors are routed by a layout engine, attribution chips live
   outside the figure, arrow tips sit at the reference point. The geometry gate
   is a unit-tested invariant, not a retry loop; the vision judge becomes a
   sampled regression check.
5. **Deterministic and cheap.** Same spec ⇒ byte-identical SVG; no model call
   in the layout path; Sonnet-class models suffice for specs; Opus only for
   dense slides.
6. **Variety is a policy, not luck.** Per-deck family quotas and archetype
   rotation are enforced at selection time, and quantitative archetypes are
   *demoted* deterministically when no data exists (radar → hierarchy tree;
   line curve → schematic curve; bar → stat callout).

## 3. Figure taxonomy

Seven families, 36 archetypes (34 in the tables below plus `composite` and
`key_points`). Each
entry lists: when the claim calls for it, the engine, the grounding rule, and
where it would have been used on the two canary decks.

### Family A — Topology (structure and flow)

Things connected by relationships. Engine: **D2 (ELK)** as primary, graphviz
`dot` as second engine; nodes sized from measured labels; orthogonal routing.

| archetype | use when | engine | grounding | canary examples |
|---|---|---|---|---|
| `flow` | a directed process with stages (DAG), L→R or T→B | D2 ELK layered | schematic | concept-aware batch sampling → contrastive loss → embedding (01/sl_14); quantize + cache per step (02/sl_18) |
| `loop` | a cyclic process: self-play, reflection, iterative denoising | dot (cycle-aware) or ring layout | schematic | VisPlay proposer→solver→reward→update (01/sl_16); VISTA test-time self-improvement (02/sl_13) |
| `nested_architecture` | containment: modules inside systems; frozen vs adapted parts | D2 containers / dot clusters; frozen glyph | schematic | surrogate LLM adapting the encoder inside a frozen LVLM (01/sl_18); OrthoFuse adapters on a frozen diffusion model (02/sl_06) |
| `layered_stack` | ordered layers with a quantity that changes over depth | deterministic stack + side annotation | schematic/hybrid | visual-token homogenization across LLM layers (01/sl_19) |
| `branching_paths` | two alternatives from one input, one emphasised | dot with two lanes + emphasis path | schematic | perception path vs recall path from an illusion image (01/sl_06); calibrated vs blind overconfidence (01/sl_08) |
| `hierarchy_tree` | a taxonomy or decomposition | dot `rankdir=TB` / radial ELK | schematic | AVA-Bench's 14 atomic abilities by category (01/sl_07) |
| `swimlane_sequence` | actors exchanging steps over time (agent ↔ tools) | D2 sequence diagram | schematic | tool-integrated RL: model ↔ crop/zoom/query tools (01/sl_17) |
| `hub_spoke` | one hub with anchored satellites (thesis → clusters → anchor works) | radial layout | schematic | S0 thesis map on both decks |

### Family B — Comparison and classification

Discrete categories against attributes. Engine: deterministic grid/column
layouts with measured cells; no graph engine.

| archetype | use when | engine | grounding | canary examples |
|---|---|---|---|---|
| `feature_matrix` | methods × properties with ✓ / ✗ / partial | measured grid, wrapped headers, glyph cells | data-lite (each cell citable) | open weights / data / recipe / non-distilled across OpenMMReasoner, Molmo2, closed checkpoint (01/sl_03); CLIP vs hard-negative vs C2-CLIP (01/sl_15) |
| `contrast_pair` | A vs B on aligned attributes; before/after | two aligned columns | schematic | per-frame enhancement vs joint temporal (02/sl_10); pairwise vs higher-order fusion (02/sl_22); retrieval shortcut vs agentic RL (01/sl_17) |
| `quadrant_map` | methods positioned on two qualitative axes | 2-D placement + label repulsion | schematic (ordinal) unless data | width vs depth reasoning (01/sl_09); four efficiency levers (02/sl_20); supervision-target map (02/sl_09) |
| `ranked_list` | an ordering by one attribute | sorted ladder / horizontal bars | data if numbers, else schematic ranks | most-referenced papers (bookend); leaderboard-style claims |
| `decision_tree` | "which method when" routing | dot | schematic | per-domain release gate for evidence sensitivity (01/sl_13) |
| `venn_overlap` | set relationships (optional, rare) | fixed geometry | schematic | — |

### Family C — Quantitative (data-grounded charts)

Requires numbers from the numeric-claims ledger (§5). Engine: **Vega-Lite via
vl-convert** (already an allowed renderer) with the brand theme; custom SVG
for stat callouts. If the ledger has no rows for the claim, the archetype is
demoted (see §6) — never faked with "illustrative" ticks.

| archetype | use when | engine | demotes to | canary examples |
|---|---|---|---|---|
| `bar_compare` | one metric across methods/conditions | Vega-Lite bar | `stat_callout` | speculative decoding up to 5× (02/sl_19); EVATok −24.4 % tokens (02/sl_20) |
| `paired_delta` | before → after per method with delta labels | Vega-Lite slope / dumbbell | `stat_callout` | HAWK: 80.2 % tokens pruned, 96.0 % accuracy retained (01/sl_20); SegQuant 21.1 % latency, 1.49× (02/sl_18) |
| `stat_callout` | 1–3 headline numbers with context and source | typographic tiles | — | 84.96 % watermark recovery under motion (02/sl_21); 55 %+ failures from exploration (01/sl_09) |
| `line_curve` | a metric vs a parameter, ≥ 3 points per series | Vega-Lite line | `schematic_curve` | quality vs sampling steps (02/sl_05) if measured |
| `scatter_frontier` | two metrics per method, Pareto frontier | Vega-Lite + frontier hull | `quadrant_map` | speed vs quality across efficiency methods (02/sl_18–20) |
| `radar_profile` | a multi-axis profile (≥ 5 axes) | Vega-Lite radial | `hierarchy_tree` | AVA-Bench per-ability scores (01/sl_07) if extracted |
| `distribution_breakdown` | shares of a whole | stacked bar / waffle (never pie) | `stat_callout` | failure types: exploration vs backtracking vs perception (01/sl_09) |
| `heatmap_grid` | numeric matrix (domain × model) | Vega-Lite rect | `feature_matrix` | per-domain grounding scores, Qwen 3VL vs Gemini on GUI grounding (01/sl_13) |
| `timeline_data` | dated events, versions, lineage | measured horizontal timeline | `flow` | CLIP → SigLIP → C2-CLIP lineage (01/sl_14–15) |
| `magnitude_ladder` | orders of magnitude (7B vs 70B params) | log-scale bar | `stat_callout` | OctoMed 7B vs larger reasoners (01/sl_24) |

### Family D — Schematic shapes (conceptual curves and spaces)

Illustrate how something behaves as something else varies. No numeric ticks;
axes carry quantity names; a visible "schematic" badge. Engine: parametric
splines from a closed shape vocabulary, with computed label anchors.

| archetype | use when | shape vocabulary | canary examples |
|---|---|---|---|
| `schematic_curve` | a qualitative function with regimes | monotone ↑/↓, saturating, U, inverted-U, knee/threshold, oscillating; up to 3 curves; regime bands | confidence should fall as blur rises, observed stays flat (01/sl_08); drift accumulating across denoising steps (02/sl_03); homogenization over layers (01/sl_19) |
| `trajectory_on_manifold` | paths through a space, on/off a surface | manifold band + trajectories + tangent/normal decomposition | ManifoldGD tangent updates (02/sl_02); one-step vs multi-step flow (02/sl_05) |
| `landscape_regions` | where a method applies; conditional behaviour | region blobs on two axes | domain-conditional evidence sensitivity (01/sl_13) |
| `tug_of_war` | two forces competing for an outcome, with a regime shift | balance diagram with weighted sides | language priors vs pixel evidence under coarse vs fine queries (01/sl_10) |

### Family E — Visual-domain illustrations (computer vision subjects)

The corpus is about images, video, 3-D and modalities; boxes labelled "image"
are a poor stand-in. Engine: a procedural **glyph library** (brand-styled SVG
symbols drawn from parameters: photo frame, blurred photo, mask grid, video
frame strip, camera frustum, splat cloud, text bubble, event stream, fMRI
volume, GPU, lock/frozen), composed by deterministic strip/grid layouts.

| archetype | use when | composition | canary examples |
|---|---|---|---|
| `image_transform_strip` | input → operation → output on an image | horizontal panels + arrows, glyphs with parameters (blur σ, mask, crop) | blur → confidence (01/sl_08); modified illusion (01/sl_06); watermark embed → paraphrase attack → recovery (02/sl_21–22); photo → fake video defence (02/sl_23); HierEdit low-res → high-res (02/sl_09) |
| `video_frame_strip` | temporal claims: flicker, consistency, motion editing | frame row with time axis and consistency links | per-frame vs joint temporal coherence (02/sl_10); MotionV2V (02/sl_11); long-video OOD correction (02/sl_12) |
| `spatial_3d_schematic` | scenes, cameras, splats, actions | isometric primitives (box, frustum, point cloud, path) | LangRef3DGS, Diff4Splat, Sculpt4D, VLA in 3-D (01/sl_25; 02/sl_16–17) |
| `token_grid_overlay` | patches/tokens kept, pruned, attended | patch grid with masks and counts | TransPrune / HAWK pruning (01/sl_20); FINER fine-grained region (01/sl_10) |
| `modality_fusion` | modalities converging or missing; competing mechanisms; several results | modality glyphs → fusion node, dashed for missing; `fusion.alternatives` (2-3) draw as a junction bus into stacked mechanisms each with its own output, winners by emphasis; `fusion.output` as a list fans out to up to three results | higher-order fusion, event + image (02/sl_22); incomplete modalities (01/sl_23); fMRI-LM (01/sl_24) |
| `evidence_crop` | a licensed real figure adds what a redraw cannot | crop + attribution caption; FIG-07 triage decides | rare for CVPR (CVF open-access copyright is not an adaptable licence) |

### Family F — Composite

Two or three panels from the families above under one header (mechanism +
headline number; frame strip + schematic curve). Each panel is itself a spec;
the compiler allocates the canvas by `size_class` and keeps a shared baseline.

### Family G — Text

`key_points` typesets the slide's own title, the first sentence of its first
claim and its visible-text lines, verbatim, with no takeaway. It is not a
picture and is never selected or suggested by a model: `ARCHETYPE_ACCEPTS`
gives it no claim type and the critic's catalog omits it. It exists for the
operator outcome "replace the figure with the slide's text layout"
(`docs/operations/figure-fidelity-policy.md`, item 9), built by
`patch_figure_spec.py --outcome replace --text-layout` from
`presentation_deck.json` (`conference_pipeline.figures.text_layout`). The
family is in `QUOTA_EXEMPT_FAMILIES`, so `deck_variety_report` reports these
slides under `exempt_from_quotas` and computes every quota without them.

## 4. The declarative figure spec

```json
{
  "spec_version": "figure-spec/1",
  "figure_id": "sl_20_pruning-visual-tokens-visual",
  "slide_id": "sl_20_pruning-visual-tokens-without-paying-full",
  "archetype": "paired_delta",
  "claim": "HAWK prunes most visual tokens while keeping almost all accuracy.",
  "takeaway": "80 % fewer tokens, 96 % of the accuracy",
  "grounding": {
    "kind": "data",
    "ledger_rows": ["nc-01-0417", "nc-01-0418", "nc-01-0419"]
  },
  "emphasis": {"highlight": ["hawk"]},
  "size_class": "full",
  "content": {
    "x": {"label": "Visual tokens kept", "unit": "%"},
    "y": {"label": "Accuracy retained", "unit": "%"},
    "series": [
      {"id": "hawk", "label": "HAWK", "before": {"x": 100, "y": 100}, "after": {"x": 19.8, "y": 96.0}, "source": "SRC-014"},
      {"id": "transprune", "label": "TransPrune", "before": {"x": 100, "y": 100}, "after": {"x": 34, "y": 97.1}, "source": "SRC-013"}
    ],
    "annotations": [{"series": "hawk", "text": "1.34× end-to-end speedup", "source": "SRC-014"}]
  }
}
```

Envelope fields (all archetypes): `spec_version`, `figure_id`, `slide_id`,
`section`, `archetype`, `claim`, `takeaway` (≤ 12 words; becomes
`visual_takeaway`), `grounding`, `emphasis` (`path`, `highlight`, `dim`),
`reading_direction`, `size_class` (`full | half | third`), `content`, optional
`constraints` (defaults per archetype: `max_primary`, `max_secondary`,
`max_label_chars`).

`section` is the blueprint's `section_id`, stamped from the deck contract rather
than authored by the model — it exists so the per-section variety quota in §6
can ask whether each section shows a number. `emphasis.path` accepts either
explicit edge ids (`"encode->fuse"`) or the sequence of nodes the path runs
through (`["encode", "fuse", "decode"]`).

Content payloads by family:

- **Topology** — `nodes[{id, label, sublabel?, kind: input|process|model|data|decision|output|frozen|adapted, group?}]`, `edges[{from, to, label?, kind: flow|feedback|contrast|dashed}]`, `groups[{id, label, members}]`, `lanes?`.
- **Comparison** — `rows[]`, `cols[]`, `cells[{row, col, value: yes|no|partial|text|number, note?, source?}]`; `contrast_pair` uses `left/right {title, items[]}`; `quadrant_map` uses `x_axis{low, high}`, `y_axis{low, high}`, `points[{label, x∈[0,1], y∈[0,1], source}]`.
- **Quantitative** — `series[{id, label, points[{x, y}], source}]`, `x{label, unit, scale}`, `y{...}`, `annotations[]`, `baseline?`; `stat_callout` uses `items[{value, unit, label, comparator?, source}]`.
- **Schematic** — `x{label}`, `y{label}`, `curves[{id, label, shape, style: ideal|observed|proposed, regimes[{from, to, label}]}]`, `annotations[]`; `trajectory_on_manifold` adds `manifold{label}`, `trajectories[{label, on_manifold: bool}]`, `vectors[]`.
- **Visual-domain** — `panels[{glyph, params{...}, caption, badge?}]`, `connectors[{from, to, label?}]`; `video_frame_strip` adds `frames`, `highlight[]`, `links[]`; `spatial_3d_schematic` adds `primitives[{kind, label, position: {row, col, level}}]`.
- **Composite** — `panels[{spec, weight}]`, `layout: row|column`.

What the model must not emit: coordinates, sizes, colours, fonts, marker
geometry, renderer choices. What it must emit: the claim, every required entity
verbatim, every number with a ledger row, and a takeaway.

## 5. Grounding: the numeric-claims ledger

A `$0`-to-cheap extraction pass over each deck's bound sources (≈ 35 sources
per deck, not the corpus) records structured numeric claims:

```
{id, source_id, span, metric, value, unit, comparator?, baseline?, condition?, confidence}
```

Sources of spans, in order: the ledger `contribution_note` (already carries
"21.1 % latency reduction and 1.49× speedup"), the abstract text in the
narrative source pack, and the full-read extraction. Quantitative archetypes
may bind only to ledger rows; the figure manifest lists the rows it used; the
artifact-truth validator gains two checks (numeric ticks ⇒ rows; schematic ⇒
badge, no ticks). This closes the "fake chart ticks" pattern the current
`forbidden_patterns` list can only forbid in prose.

## 6. Selection policy

The blueprint's `visual_role` becomes a **claim type** plus an archetype
proposal; the selector is deterministic:

| claim type | first choice | if no ledger rows | if entity budget exceeded |
|---|---|---|---|
| mechanism / process | `flow`, `loop`, `nested_architecture` | — | split into `composite` or drop sublabels |
| comparison | `feature_matrix`, `contrast_pair` | — | `contrast_pair` |
| magnitude / result | `paired_delta`, `bar_compare` | `stat_callout` | `stat_callout` |
| behaviour shape | `line_curve` | `schematic_curve` | — |
| profile / taxonomy | `radar_profile` | `hierarchy_tree` | `ranked_list` |
| tradeoff / positioning | `scatter_frontier` | `quadrant_map` | `contrast_pair` |
| temporal (video) | `video_frame_strip` | — | fewer frames |
| image operation | `image_transform_strip` | — | fewer panels |
| spatial / 3-D | `spatial_3d_schematic` | — | — |
| multimodal | `modality_fusion` | — | — |

Per-deck variety quotas (enforced at selection, reported in the deck manifest):
no family above 45 % of slides; at least one Family C or `stat_callout` per
section when ledger rows exist; at least two Family E figures on a
vision deck; no archetype on three consecutive slides (the archetype rotation
already on `main`).

Three rules added after deck 03 (2026-09-11), all in `selection.py`:

- **A subject cue never outranks a relation-heavy brief.** Every gloss on a
  geometry deck says "point cloud", so the strong cue put 8 of 21 figures on
  `spatial_3d_schematic`; 3 of 4 compile refusals and 5 of 9 first-round judge
  failures were those scenes. Each picture archetype now publishes what it can
  place and say (`PICTURE_CAPACITY`: a scene, 5 parts and 4 relations; a strip,
  its panel budget and no more relations than a chain); a brief beyond that is
  about its relations and the claim type's structural archetype carries them,
  with the reason recorded.
- **Family share is judged while choosing, not after paying.** Once eight
  slides are chosen, a pick that would keep a family over 45 % yields to the
  first archetype from another family that accepts the claim type -- structural
  shapes before pictures, pictures only within capacity, and never a
  subject-only archetype (`modality_fusion`, `tug_of_war`, ...) the brief did
  not name. When nothing else carries the claim the pick stands and says so:
  a blueprint that assigns 14 topology roles to 25 slides has decided the
  share, and whether to accept it is the operator's call at compile time.
- **The critic may change the archetype.** A `wrong_archetype` finding that
  names a known archetype re-authors under it (`named_archetype_switch`),
  unless it names a picture the brief outgrows; before this the re-author
  regenerated the spec under the same archetype the selector had fixed, which
  made that one finding impossible to act on.

## 7. Compiler pipeline

```
spec (JSON) ─▶ validate ─▶ shape text ─▶ layout ─▶ style ─▶ geometry gate ─▶ emit
                schema      font metrics   per-family  brand    invariants     SVG +
                budgets     wrap/shrink    engine      tokens   + remedies     manifest
                grounding
```

1. **Validate** — JSON schema per archetype; label budgets; required entities
   present verbatim; grounding rules (§5); demotion applied (§6).
2. **Shape text** — measure with the brand font (Pillow `ImageFont` /
   fontTools, same face the reveal theme uses); wrap to a per-archetype column
   width; shrink to a floor of 16 px at 1280 px canvas; never truncate a word;
   trim in a fixed priority order (edge labels → sublabels → annotations).
3. **Layout** — Family A: D2/ELK with measured node sizes, orthogonal routing,
   `ranksep`/`nodesep` from the size class; `dot`/`circo` for loops and trees.
   Family B: grid solver. Family C: Vega-Lite → SVG via vl-convert with the
   brand theme. Family D: parametric splines with label anchors on the curve.
   Family E: strip/grid composition of glyphs. Family F: panel allocation.
4. **Style** — one token set (palette, faces, stroke widths, node-kind shapes,
   frozen/adapted encodings, badge styles); a single marker library with tips
   at the reference point; attribution chips outside the figure by default.
5. **Geometry gate** — invariants checked on the emitted SVG: no text bbox
   intersects another text or box bbox; no connector segment crosses a label;
   arrowheads never intrude into boxes; everything inside the canvas with
   margin; minimum font size; contrast against the theme ground. Two more are
   held on what the compiler recorded rather than on the SVG, because the
   shared checker cannot see them: every arrow shows at least
   `MIN_VISIBLE_SHAFT` of line behind its head (`arrowhead_without_shaft`),
   and no arrow crosses a drawn picture -- a glyph, an isometric block, a
   polygon node -- it neither starts nor ends at (`connector_through_glyph`).
   Layouts honour the first by construction: `min_arrow_run()` is the floor
   for any separation an edge crosses (ranks, linked group members, tree
   drops, branch steps, loop neighbours, scene primitives). Failures
   trigger bounded, deterministic remedies (increase separation → wrap →
   shrink to floor → drop lowest-priority annotation → switch orientation),
   then a hard error — never a model call.
6. **Emit** — the SVG plus `figure_manifest.json`: archetype, grounding kind
   and ledger rows, entities rendered (verbatim strings), numbers rendered,
   glyphs used, engine versions, spec hash. The semantic reviewer and the
   artifact-truth validator read the manifest instead of scraping SVG text.
7. **Regression** — the vision judge runs on a sample (or on demand), records
   findings against the archetype, and never drives a per-figure retry.

Determinism: same spec + same engine versions ⇒ byte-identical SVG (pinned D2,
graphviz, vl-convert; fixed seeds for label repulsion). Cache key = spec hash.

## 8. The model's role after the split

- **Spec author** (Sonnet-class by default; Opus for dense slides): classify the
  claim type, fill the archetype payload from the blueprint (components,
  edges, required entities), attach ledger rows to every number, choose
  emphasis, write the takeaway. Prompted with the schema and two examples per
  family; validated by the compiler; one repair attempt on validation errors.
- **Spec critic** (optional, Sonnet vision, sampled): does the rendered figure
  make the claim legible? Findings feed the *selection policy* (e.g. this
  claim type wants a different archetype), not a redraw loop.
- **Nothing else.** No coordinate authoring, no colour choices, no marker
  paths.

## 9. Quality gates and tests

- Unit tests per archetype over a fuzz corpus: 1–12 nodes, labels 4–60 chars,
  unicode and math, RTL-safe wrapping, every size class; the geometry gate
  must report zero findings for every generated case.
- Golden SVG snapshots per archetype for regression of style and layout.
- Grounding tests: a `data` spec without ledger rows is rejected; a
  `schematic` spec renders the badge and no numeric ticks.
- Manifest contract tests: entities and numbers in the manifest are present in
  the SVG text nodes.
- Performance: < 2 s per figure on the build machine; no network.
- Deck-level: variety quotas enforced and reported; vision-judge findings per
  archetype tracked over time as a health metric.

## 10. Phasing

| phase | scope | covers | effort |
|---|---|---|---|
| 0 | merge `main`'s geometry gate, marker-tip fix, chips-only attribution, archetype rotation into this branch; re-run figures for decks 01/02 | biggest two defect classes today | 1 day, ≈ $5/deck |
| 1 | spec schema + validator + text shaping + Family A (`flow`, `loop`, `nested_architecture`) on D2 + Family C (`paired_delta`, `bar_compare`, `stat_callout`) on Vega-Lite + numeric-claims ledger + geometry gate + manifest | ≈ 70 % of current slides | 1 week |
| 2 | Family B (`feature_matrix`, `contrast_pair`, `quadrant_map`, `ranked_list`, `decision_tree`), `hierarchy_tree`, `swimlane_sequence`, `timeline_data`, `heatmap_grid` | remaining structure/comparison slides | 1 week |
| 3 | Family D (schematic curves, manifold trajectories, region maps) + Family E glyph library and strips + `spatial_3d_schematic` | the vision-specific and behaviour-shape slides | 1–2 weeks |
| 4 | Family F composites, selection policy + demotion, retire freeform drawsvg to a gated fallback, validator checks for grounding | platform-wide | 3 days |

## 11. Open questions

1. **Glyph library source** — generate the CV glyphs procedurally in brand
   style (proposed), or adopt an icon set? Procedural keeps them parametric
   (blur radius, mask density) and licence-free.
2. **Evidence crops for CVPR** — CVF open-access papers are not under an
   adaptable licence; keep `evidence_crop` gated off for this conference unless
   a source is CC BY.
3. **Schematic badge wording and placement** — proposed: a small "schematic —
   illustrates the claim, not measured data" tag in the figure corner and in the
   narration's visual takeaway.
4. **Numeric-claims ledger as its own stage** — run per deck at binding time
   (proposed) versus as a corpus-wide analysis stage.
5. **Quota values** — the 45 % / two-Family-E / no-three-in-a-row defaults above
   are starting points to tune after the first compiled deck.

## 12. Implementation note (what was built, and where it differs)

Phases 0-4 are implemented, and §9's test battery with them. Six decisions
differ from the proposal above; each is a deliberate change, not an omission.

**Layout engines are pure Python, not D2/ELK, graphviz and Vega-Lite (§3, §7.3).**
The proposal named external engines. The build implements a layered
(Sugiyama-style) topology layout, a measured grid solver, chart drawing and
parametric splines directly, for three reasons the proposal's own requirements
force. Determinism: §7 asks for byte-identical SVG from the same spec, which a
pinned third-party engine cannot promise across versions or platforms.
Agreement with the gate: correctness here rests on layout and checking using
*the same* text metric, and an external engine measures text with its own
fonts, so a node it sizes can still overflow under the checker that judges the
result. Failure surface: three subprocess dependencies (one of which,
vl-convert, is not installed here) is three ways for a build to fail for a
reason that has nothing to do with the figure. The layout code is behind one
registry (`figures/layouts/__init__.py`), so an external backend can be added
per archetype later without touching the spec contract.

**Thirty-five archetypes, not thirty.** The extra five are `venn_overlap`
(listed as optional in §3), plus the composite panel type counted separately,
and the split of a few families' entries into distinct payload contracts.

**Rules are not connectors.** An axis, a gridline, a lifeline and a timeline
spine are emitted as thin rects rather than `<line>` elements. The geometry
gate holds every connector to landing on a shape and to clearing every label;
scenery has nothing to land on, and treating it as a connector produced false
"dangling arrow" defects on the first pass. Direction, where scenery needs it,
is a small triangle polygon.

**The variety quota fails closed.** §6 reports the mix in the deck manifest.
The build blocks on it, with `--allow-figure-variety-findings` as the named
escape, because a non-blocking quota is exactly what let the canary ship 27
box-and-arrow slides.

**Budget keys are per-family, not per-node-type (§4).** The proposal named
`max_nodes`, `max_edges` and `max_labels`, which only read naturally for
topology: a feature matrix has rows and cells, a chart has series and points.
The build uses `max_primary` (nodes / rows / series, whichever the family
counts), `max_secondary` (edges / cells / points) and `max_label_chars`, so one
budget vocabulary covers all six families instead of three names that mean
something different in each.

**The font floor is 20 px, not 16 (§7.2).** `FONT_FLOOR` is 20 with
`FONT_ABSOLUTE_FLOOR` at 18, above the proposal's 16, because the renderer's own
legibility gate rejects text below its minimum and a figure that shrank to a
legal-but-unreadable 16 px would pass the compiler and fail the deck. Shrinking
stops earlier and the figure fails loudly instead.

### What the build proves

- 35 archetypes, 35 worked examples from real canary slides, all compiling
  clean against the measured geometry gate, twice, byte-identically.
- A topology fuzz corpus (1-10 nodes x labels from one character to sixty,
  unicode, an unbreakable token) either compiles clean or fails loudly. There
  is no third outcome where a bad figure reaches a slide.
- The numeric-claims ledger, run on the two canary decks' own bound sources,
  finds exactly the measured claims this document cites: 80.2% / 96.0% / 1.34x
  on deck 01, and 21.1% / 1.49x / 5x / 84.96% / 24.4% on deck 02. A spec citing
  a row the deck's ledger does not hold is blocked before anything is drawn.
- The whole free half of the route (ledger -> selection -> prompt -> spec ->
  compile -> manifest -> receipt) runs end to end on deck 01's real deck
  contract: 25 slides, 25 specs, 25 figures, zero geometry findings, no spend.
- §9's battery in full: a golden SVG snapshot per archetype; every archetype at
  every size class; every archetype against hostile labels (one character, an
  unbreakable token, math notation, RTL scripts, CJK with nothing to wrap on);
  all three emphasis modes; the schematic badge; a per-figure time budget; and
  the grounding, manifest-contract and variety-quota checks. 687 tests.
- The two §5 validator checks are wired into `artifact_truth.py` as the
  `figures` check, reading figure manifests rather than scraping SVG text. To
  tell an axis tick from a number in a claim sentence the manifest now records
  `numeric_ticks` separately from `numbers_rendered`.
- The spec-hash cache of §7 is real: an unchanged spec is not recompiled, an
  edited spec recompiles only itself, and a hand-edited SVG is rebuilt rather
  than served (the cache re-hashes the file it is about to reuse).
- The remedy ladder trims in §7.2's stated order. It previously had a single
  `drop_annotations` step that took edge labels and annotations together; the
  ladder now drops edge labels, then sublabels, then annotations, so a figure
  that only needs to lose its edge labels keeps the text that carries facts.
- `emphasis.path` accepts a node sequence as well as explicit edge ids. Only
  the edge-id form was ever read, so a spec naming the nodes on the path
  emphasised nothing and reported nothing.

### What the paid run and the judge found

The paid half has now run, and so has the vision judge. Both are recorded here
because what they found is worth more than the artefacts they produced.

**Authoring.** 49 specs across the two canary decks, Sonnet, $3.34 over 90
calls ($0.068 a spec). Every spec valid at the schema gate. Four defects
surfaced that `--dry-run` could not: a receipt-boundary attribute error on the
very first live call; `skip_existing` adopting dry-run placeholders as authored
content; a one-word ledger match convincing the selector a slide had data; and
the closed vocabularies missing from the prompt, so the author had to guess a
glyph name and a guess is a rejected spec.

**The judge.** Three rounds over the 36 figures that compiled: 30 -> 27 -> 25
findings (0.83 -> 0.69 per figure) against the hand-drawn route's 57 over 53
slides (1.08). `text_overflow` and `text_clipped` went to zero and stayed
there. $2.73 over 108 calls, about $0.91 a round. The judge is not
deterministic: single-category moves of one or two between rounds are noise,
the three-round trend is not.

**What the remaining findings turned out to be.** Not, as first reported, two
classes beyond geometry's reach. Re-attributing every round-3 finding to its
figure and reading the spec behind it gave five causes, three of them in the
compiler:

- Layout defects behind 12 of 13 refusals, every one in an archetype the fuzz
  corpus never covered because the corpus only ever fuzzed `flow`.
- Archetype misuse the validator could not see: a pipeline authored as a
  filmstrip because the word "frames" appeared in its gloss; alternatives drawn
  as a sequence; a frontier through two incommensurable points.
- Silent fallbacks: four layouts discarded a label when placement failed, and
  the checker's path parser could not see a loop's arrows at all.
- Decorative glyphs, because the default picture for a panel was a landscape.
- No fuzz coverage for 33 of 35 archetypes, which is why real specs kept
  finding defects the worked examples never did.

### What was built in response

- **A label-accounting contract.** Every label is drawn, relocated, or dropped
  with a named reason, and a drop that is not a ladder trim is a gate error.
  It found 45 unaccounted labels on the canary decks on its first run,
  including all three labels of `spatial_3d_schematic`'s own worked example.
- **A fuzz corpus per archetype**, as §9 always asked for: ~1,200 generated
  cases over all 35, structural where the shape varies and label-substituted
  elsewhere.
- **Per-family remedy ladders.** One ladder served every archetype and five of
  its eleven rungs turned a lever only `flow` reads; for trees the one lever
  that applied was turned the wrong way.
- **A structure lint** (`figures/lint.py`): eleven deterministic rules that
  refuse or demote a spec whose shape misstates its claim. Fail-closed at
  compile with no escape flag -- the remedy is re-authoring.
- **Strong and weak subject cues** in selection, so a common word in a gloss no
  longer overrides the claim type.
- **`depicts` and glyph modifiers**: a panel names its subject and the compiler
  picks the drawing, with fifteen parametric modifiers over the base scenes.
- **Judge context**: the reviewer now receives the claim, the archetype, the
  grounding kind, the compiler's own conventions, and a definition for every
  check id. The enum is unchanged so counts stay comparable; the report carries
  a `rubric_version`.

### Still open

- The 10 specs the structure lint refuses have not been re-authored. They are
  the ones whose pictures contradicted their claims, and re-authoring them is
  the next paid step.
- The optional **spec critic** of §8 is still unbuilt. Everything it would
  catch that a deterministic rule can catch is now a lint rule; what remains
  for it is the judgement call, and it should be run sampled and budget-capped.
- §11's five questions are answered as proposed, with one revision: the glyph
  library is procedural AND compositional. A base scene plus a named modifier
  is what makes a panel depict its subject rather than merely differ from its
  neighbour, and varying the scene compositions alone -- which was tried --
  moved no finding at all.
